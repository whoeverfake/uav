import numpy as np

def dBm_to_mW(dBm):
    """
    将dBm转换为瓦特（W）

    :param dBm: 功率以dBm为单位
    :return: 功率以mW为单位
    """
    return 10 ** ((dBm ) / 10)


num_gu = 200
"""用户数"""
num_uav = 6
"""UAV数量"""
num_channel = 4
"""信道数"""
bandwidth = 1000000
"""带宽1MHz"""
rcell_max = 500
"""分布范围"""
height = 50
"""无人机高度"""
d_min = 50
"""安全间隔"""
N0 = dBm_to_mW(-174)
"""噪声功率谱密度"""
omega = 20 / 1000 #(w with mW)
"""能耗开销"""
tau_min = dBm_to_mW(-10)
"""最小SINR"""
phi_max = 30
"""最大能耗"""
power_levels = np.arange(0,dBm_to_mW(20),20,dtype = int)
"""功率等级"""

x_range = 1000
y_range = 1000

