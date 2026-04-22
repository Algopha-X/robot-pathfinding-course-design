import numpy as np
import matplotlib.pyplot as plt

def get_map():
    # 严格按照图片创建 9行 x 8列 的矩阵
    grid = np.zeros((9, 8))
    
    # 终点与起点
    start_pos = (2, 3)  # S
    end_pos = (4, 7)    # E
    
    # 障碍物 (1 表示蓝色墙壁)
    grid[1, 1:5] = 1    # 顶部横梁 (行1, 列1到4)
    grid[2:4, 1] = 1    # 左侧短臂 (行2到3, 列1)
    grid[1:8, 4] = 1    # 右侧贯穿长臂 (行1到7, 列4)
    grid[7, 1:5] = 1    # 底部横梁 (行7, 列1到4)
    
    return grid, start_pos, end_pos

# --- 绘图验证模块 ---
if __name__ == "__main__":
    grid, start, end = get_map()
    
    plt.figure(figsize=(8, 9))
    # 使用 origin='upper' 确保 (0,0) 在左上角
    plt.imshow(grid, cmap='Blues', origin='upper')
    
    # 标注 S 和 E
    plt.text(start[1], start[0], 'S', color='red', ha='center', va='center', fontsize=20, weight='bold')
    plt.text(end[1], end[0], 'E', color='black', ha='center', va='center', fontsize=20, weight='bold')
    
    # 绘制严格的网格线以供对比
    plt.xticks(np.arange(-0.5, 8, 1), [])
    plt.yticks(np.arange(-0.5, 9, 1), [])
    plt.grid(color='gray', linestyle='-', linewidth=1)
    plt.title("Robot Navigation Grid (9x8) - Pixel Perfect", fontsize=14)
    
    plt.show()