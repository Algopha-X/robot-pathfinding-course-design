import numpy as np
import matplotlib.pyplot as plt

def advanced_env():
    # 创建 20x20 的复杂地图
    grid = np.zeros((20, 20))
    grid[5:15, 10] = 1 # 垂直长墙
    grid[10, 5:15] = 1 # 水平长墙
    start = (2, 2)
    end = (18, 18)
    return grid, start, end

def artificial_potential_field(current, end, obstacles):
    # 基础引力增益
    k_att = 1.0
    # 基础斥力增益
    k_rep = 100.0
    # 斥力作用范围
    rho_0 = 2.0
    
    # 1. 计算引力 (指向终点)
    f_att = k_att * (np.array(end) - np.array(current))
    
    # 2. 计算斥力 (背离障碍物)
    f_rep = np.array([0.0, 0.0])
    for obs in obstacles:
        dist = np.linalg.norm(np.array(current) - np.array(obs))
        if dist < rho_0:
            # 斥力公式: k * (1/dist - 1/rho_0) * (1/dist^2)
            f_rep += k_rep * (1.0/dist - 1.0/rho_0) * (1.0/(dist**2)) * (np.array(current) - np.array(obs)) / dist
            
    return f_att + f_rep

# 这里仅展示原理逻辑，后续可集成到之前的可视化引擎中