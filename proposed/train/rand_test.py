import numpy as np

from environment2.constant_muav import *
from environment2.muav_env import *
import os

reward_eps = []
energy = []



for test_num in range(10):

    users = generate_user_coordinates_2(num_gu, rcell_max, rcell_max)
    drones_power = power_levels[1] * np.ones((num_uav,))
    drones = initialize_drones(rcell_max, num_uav)
    new_column = np.full((num_uav, 1), height)
    drones_3d = np.hstack((drones, new_column))
    sub_G = calc_channel_gain(drones_3d, users, num_channel)

    channel_matrix = generate_matrix(num_uav, num_channel)

    output, user_count = calculate_throughput_and_users_2(sub_G[:, :, 0], channel_matrix, tau_min,drones_power, N0, bandwidth)
    phm = output
    n = user_count

    movement = np.array([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]])


    reward = np.zeros((num_uav, 1))
    output = np.zeros((num_uav, 1))

    for a in range(100):
        actions = np.random.uniform(-1, 1, (num_uav, 3))

        for m in range(num_uav):
            action = actions[m,:]
            direction, channel_choice, power_index = action

            drones[m, :] += movement[int((1 + direction) / 2 * 4)]
            channel_matrix[m][int((1 + channel_choice) / 2 * (num_channel - 1))] = 1
            drones_power[m] = power_levels[int((1 + power_index) / 2 * 4)]

        Din_flg = check_distances(drones, d_min)

        drones = np.clip(drones, 0, rcell_max)



        new_column = np.full((num_uav, 1), height)
        # 将原数组和新列在列维度上拼接
        drones_3d = np.hstack((drones, new_column))

        sub_G = calc_channel_gain(drones_3d, users, num_channel)
        output, user_count = calculate_throughput_and_users_2(sub_G[:, :, 0], channel_matrix, tau_min,
                                                                  drones_power, N0, bandwidth)

        reward_tmp = output
        n = user_count

        for i in range(num_uav):
            if Din_flg:
                reward[i][0] = -2
            elif reward_tmp[i] >= phi_max:
                reward[i][0] = -1
            else:
                reward[i][0] = reward_tmp[i] - 2 * (num_gu - np.sum(n)) - omega * drones_power[i]

    reward_eps.append(np.sum(reward))
    energy.append(np.sum(output))

# print(np.mean(np.array(reward_eps)))
print('mean_phm = ' + str(sum(energy)/10))
# np.savetxt(os.getcwd() + '/' + 're_vec_rand.csv', np.array(reward_eps), delimiter=',')
# np.savetxt(os.getcwd() + '/' + 'phm_vec_rand.csv', np.array(energy), delimiter=',')