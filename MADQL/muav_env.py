import numpy as np
from math import pi, cos, sin
from constant_muav import num_gu,num_uav,num_channel,bandwidth,N0,d_min,rcell_max,height,tau_min,power_levels,phi_max,omega
from scipy.cluster.vq import kmeans2
from gym import spaces

class MultiAgentEnv:
    def __init__(self):
        self.M = num_uav  # Number of drones (agents)
        self.N = num_gu  # Number of ground users
        self.num_channel = num_channel
        self.rcell_max = rcell_max  # Square area side length
        self.d_min = d_min  # Minimum distance between drones

        self.height = height
        self.bandwidth = bandwidth
        self.N0 = N0
        self.omega = omega # wW to W
        self.tau_min = tau_min
        self.phi_max = phi_max

        self.power_levels = power_levels

        self.agent_num = self.M
        self.action_dim = 3
        self.obs_dim = 4
        self.share_obs_dim = self.agent_num * self.obs_dim

        self.drones = None
        self.phm = None
        self.channel_matrix = None
        self.drones_power = None
        self.n = None

        self.users = None

        # Initialize environment
        self.reset()

    def reset(self):
        self.drones_power = self.power_levels[1] * np.ones((self.M,))
        self.users = generate_user_coordinates_2(self.N,self.rcell_max,self.rcell_max)
        self.drones = place_drones_scipy(self.users,self.M,self.height)
        new_column = np.full((self.M, 1), self.height)
        drones_3d = np.hstack((self.drones, new_column))
        sub_G = calc_channel_gain(drones_3d, self.users, self.num_channel)

        self.channel_matrix = generate_matrix(self.M, self.num_channel)

        output, user_count = calculate_throughput_and_users_2(sub_G[:, :, 0], self.channel_matrix, self.tau_min, self.drones_power, self.N0,
                                         self.bandwidth)
        self.phm = output
        self.n = user_count


        return self._get_state()

    def _initialize_drones(self):
        drones = np.zeros((self.M, 2))
        for i in range(self.M):
            angle_tmp = 2 * pi * i / self.M
            while True:
                # pos = np.random.uniform(0, self.rcell_max, 2)
                pos = np.array([int(round(self.rcell_max / 2 * cos(angle_tmp))), int(round(self.rcell_max / 2 * sin(angle_tmp)))])
                if all(np.linalg.norm(pos - drones[:i, :], axis=1) >= self.d_min):
                    drones[i, :] = pos
                    break
        return drones

    def _get_state(self):
        state = np.concatenate([self.drones, self.n[:, None], self.phm], axis=1)
        return state

    def step(self, actions):

        movement = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])
        self.channel_matrix = np.zeros((self.M, self.num_channel))
        for i, (direction, channel_choice, power_index) in enumerate(actions):
            # Update drone position
            self.drones[i, :] += movement[int((1+direction)/2*4)]
            self.channel_matrix[i][int((1+channel_choice)/2*(self.num_channel-1))] = 1
            self.drones_power[i] = self.power_levels[int((1+power_index)/2*4)]

        Din_flg = check_distances(self.drones,self.d_min)

        self.drones = np.clip(self.drones, 0, self.rcell_max)

        # Calculate reward(when train)
        reward = self.calcul_reward_train()

        # Calculate reward(when test)
        # reward = self.calcul_reward_test()

        return self._get_state(), reward, self.calcul_dones(), ''

    def calcul_dones(self):
        dones = False
        return dones

    def calcul_reward_train(self):

        Din_flg = check_distances(self.drones, self.d_min)

        self.drones = np.clip(self.drones, 0, self.rcell_max)

        # Calculate reward
        reward = np.zeros((self.M, 1))
        reward_tmp = self._calculate_reward()
        for i in range(self.M):
            self.phm[i] = reward_tmp[i]
            if Din_flg:
                reward[i][0] = -2
            elif reward_tmp[i] >= self.phi_max:
                reward[i][0] = -1
            else:
                reward[i][0] = reward_tmp[i] - 2 * (num_gu - np.sum(self.n)) - self.omega * self.drones_power[i]
        return reward

    def calcul_reward_test(self):

        self.drones = np.clip(self.drones, 0, self.rcell_max)
        # Calculate reward
        reward = np.zeros((self.M, 1))
        reward_tmp = self._calculate_reward()
        for i in range(self.M):
            reward[i][0] = reward_tmp[i]
            self.phm[i] = reward_tmp[i]

        return reward


    def _calculate_reward(self):

        new_column = np.full((self.M, 1), self.height)
        # 将原数组和新列在列维度上拼接
        drones_3d = np.hstack((self.drones, new_column))

        sub_G = calc_channel_gain(drones_3d,self.users,self.num_channel)
        output, user_count = calculate_throughput_and_users_2(sub_G[:,:,0], self.channel_matrix, self.tau_min,self.drones_power,self.N0,self.bandwidth)
        self.n = user_count

        return output

    def calculate_reward_final(self):

        new_column = np.full((self.M, 1), self.height)
        # 将原数组和新列在列维度上拼接
        drones_3d = np.hstack((self.drones, new_column))

        sub_G = calc_channel_gain(drones_3d,self.users,self.num_channel)
        output, user_count = calculate_throughput_and_users_2(sub_G[:,:,0], self.channel_matrix, self.tau_min, self.drones_power,self.N0,self.bandwidth)
        self.n = user_count

        return output


def generate_positions(num_entities, area_size, fixed_height=None):

    positions = np.random.uniform(low=0.0, high=area_size, size=(num_entities, 2))
    if fixed_height is not None:
        heights = np.full((num_entities, 1), fixed_height)
    else:
        heights = np.zeros((num_entities, 1))
    return np.hstack((positions, heights))


def generate_matrix(M, K):
    A = np.zeros((M, K), dtype=int)
    for m in range(M):
        # 随机选择一个列索引设置为1
        k = np.random.randint(K)
        A[m, k] = 1

    return A



def calc_channel_gain(loc_uav_list, loc_iot_list, num_channels):
    N = loc_iot_list.shape[0]
    M = loc_uav_list.shape[0]
    h_matr = np.zeros((M, N))
    theta = np.zeros((M, N))
    distance = np.zeros((M, N))
    prob_los = np.zeros((M, N))
    prob_nlos = np.zeros((M, N))
    g_los = np.zeros((M, N))
    g_nlos = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            distance[i, j] = np.linalg.norm(loc_iot_list[j, :] - loc_uav_list[i, :])
            theta[i, j] = np.degrees(np.arcsin(loc_uav_list[i, 2] / distance[i, j]))  # 仰角计算
            prob_los[i, j] = 1 / (1 + 9.6177 * np.exp(-0.1581 * (theta[i, j] - 9.6177)))
            prob_nlos[i, j] = 1 - prob_los[i, j]
            g_los[i, j] = 20 * np.log10(distance[i, j]) + 1
            g_nlos[i, j] = 20 * np.log(distance[i, j]) + 20
            pl_mean_db = prob_los[i, j] * g_los[i, j] + prob_nlos[i, j] * g_nlos[i, j]
            h_matr[i, j] = 1 / (10 ** (pl_mean_db / 10))

    # 扩展 h_matr 到 M x N x K 的矩阵
    h_matr_expanded = np.repeat(h_matr[:, :, np.newaxis], num_channels, axis=2)

    return h_matr_expanded


def generate_user_coordinates_2(N, xmax, ymax):

    # np.random.seed(seed)
    # 生成用户的x和y坐标，范围在[0, xmax]和[0, ymax]之间
    x_coords = np.random.uniform(0, xmax, N)
    y_coords = np.random.uniform(0, ymax, N)
    z_coords = np.zeros(N)
    coordinates = np.vstack((x_coords, y_coords, z_coords)).T

    return coordinates



def place_drones_scipy(user_coordinates, M, H):
    # 只使用x, y坐标进行K-means聚类
    user_coordinates_xy = user_coordinates[:, :2]
    centroids, _ = kmeans2(user_coordinates_xy, M, iter=100, minit='points')
    drone_coordinates = centroids
    return drone_coordinates


def calculate_throughput_and_users_2(G, C, tau, p, a, B):
    M, N = G.shape
    _, K = C.shape

    # 计算SINR
    SINR = np.zeros((M, N))
    interference = np.zeros((M, N))
    for m in range(M):
        k = np.argmax(C[m,:])
        for n in range(N):
            # 计算干扰
            interference[m,n] = sum(G[i, n] * p[i] * C[i][k] for i in range(M) if i != m)

            # 信噪比公式
            SINR[m, n] = p[m] * G[m, n] / (interference[m,n] + 0.0002 + B * a)

    # 确保用户只能接入一个无人机，选择SINR最大的那个
    user_association = np.argmax(SINR, axis=0)

    # 计算每个无人机的吞吐量和接入用户数
    total_throughput = np.zeros((M, 1))
    users_connected = np.zeros(M, dtype=int)

    for m in range(M):
        for n in range(N):
            if user_association[n] == m and SINR[m, n] > tau:
                # 用户n接入无人机m，并且SINR[m,n] > tau
                users_connected[m] += 1
                total_throughput[m] += np.log2(1 + SINR[m, n]) / 10

    return total_throughput, users_connected


def check_distances(positions, dmin):
    M = positions.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < dmin:
                # print("distance:" + str(distance) + ". ")
                return 1
    return 0

