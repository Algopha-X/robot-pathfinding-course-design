import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. 复杂迷宫生成 (30x30)
# ==========================================
def create_complex_maze(size=30):
    grid = np.zeros((size, size))
    # 随机生成密集障碍物
    np.random.seed(42) 
    grid = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3])
    
    # 确保起点和终点是通路
    start, end = (1, 1), (size-2, size-2)
    grid[start] = 0
    grid[end] = 0
    
    # 挖出一些必要的通道确保连通性 (简化演示)
    grid[1:size-1, 1] = 0
    grid[size-2, 1:size-1] = 0
    
    return grid, start, end

# ==========================================
# 2. Q-Learning 算法类
# ==========================================
class QLearningAgent:
    def __init__(self, states_n, actions_n, lr=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states_n[0], states_n[1], actions_n))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3) # 探索
        return np.argmax(self.q_table[state[0], state[1]]) # 利用

    def learn(self, s, a, r, s_):
        q_predict = self.q_table[s[0], s[1], a]
        q_target = r + self.gamma * np.max(self.q_table[s_[0], s_[1]])
        self.q_table[s[0], s[1], a] += self.lr * (q_target - q_predict)

# ==========================================
# 3. 训练与可视化
# ==========================================
def train_agent():
    size = 30
    grid, start, end = create_complex_maze(size)
    agent = QLearningAgent((size, size), 4)
    
    episodes = 500
    for ep in range(episodes):
        state = start
        steps = 0
        while state != end and steps < 200:
            action = agent.choose_action(state)
            # 执行动作
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            next_state = (state[0] + moves[action][0], state[1] + moves[action][1])
            
            # 边界与障碍检查
            if 0 <= next_state[0] < size and 0 <= next_state[1] < size and grid[next_state] == 0:
                reward = 100 if next_state == end else -1
            else:
                reward = -10
                next_state = state # 撞墙原地不动
            
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
            
        if ep % 100 == 0:
            print(f"Episode {ep} finished.")

    # 展示最终学到的路径
    show_path(grid, agent, start, end)

def show_path(grid, agent, start, end):
    path = [start]
    curr = start
    while curr != end and len(path) < 100:
        action = np.argmax(agent.q_table[curr[0], curr[1]])
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        curr = (curr[0] + moves[action][0], curr[1] + moves[action][1])
        path.append(curr)
    
    plt.imshow(grid, cmap='binary')
    py, px = zip(*path)
    plt.plot(px, py, color='red', label='Q-Learning Path')
    plt.scatter([start[1]], [start[0]], c='g', s=100)
    plt.scatter([end[1]], [end[0]], c='b', s=100)
    plt.title("30x30 Complex Maze: Q-Learning Result")
    plt.show()

if __name__ == "__main__":
    train_agent()