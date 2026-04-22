import os
import heapq
from collections import deque
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
from matplotlib import animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ["U", "D", "L", "R"]


@dataclass
class SearchResult:
    name: str
    path: list
    visited_order: list
    path_cost: float
    success: bool
    extra: dict | None = None


def get_basic_map():
    """原始 9x8 栅格地图。"""
    grid = np.zeros((9, 8), dtype=int)
    start_pos = (2, 3)
    end_pos = (4, 7)
    grid[1, 1:5] = 1
    grid[2:4, 1] = 1
    grid[1:8, 4] = 1
    grid[7, 1:5] = 1
    return grid, start_pos, end_pos


def get_challenge_map(size=30):
    """
    自主设计的 30x30 高难度场景:
    - 多条长墙和狭窄通道
    - U 型陷阱与局部极小值区域
    - 起点终点分处对角，要求算法具备更强的全局规划能力
    """
    grid = np.zeros((size, size), dtype=int)

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    vertical_walls = {
        4: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        8: [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
        12: [2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27],
        16: [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 26, 27, 28],
        20: [2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        24: [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27],
    }
    for col, rows in vertical_walls.items():
        for row in rows:
            grid[row, col] = 1

    horizontal_walls = {
        5: [5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
        9: [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        14: [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        18: [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26],
        22: [4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        26: [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26],
    }
    for row, cols in horizontal_walls.items():
        for col in cols:
            grid[row, col] = 1

    # U 型障碍和假通道，增加人工势场和贪心启发的难度
    grid[6:12, 21] = 1
    grid[6, 18:22] = 1
    grid[11, 18:22] = 1
    grid[17:24, 6] = 1
    grid[17, 6:10] = 1
    grid[23, 6:10] = 1

    # 明确保留关键门洞，保证地图可达
    gates = [
        (4, 2), (13, 4), (9, 8), (20, 8), (13, 12), (23, 12),
        (7, 16), (24, 16), (17, 20), (9, 24), (21, 24), (26, 20),
        (5, 8), (14, 14), (22, 10), (26, 24), (18, 6), (20, 6),
    ]
    for gate in gates:
        grid[gate] = 0

    start = (1, 1)
    end = (28, 28)
    grid[start] = 0
    grid[end] = 0
    return grid, start, end


def in_bounds(grid, node):
    r, c = node
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def get_neighbors(grid, node):
    neighbors = []
    for move in DIRECTIONS:
        nxt = (node[0] + move[0], node[1] + move[1])
        if in_bounds(grid, nxt) and grid[nxt] == 0:
            neighbors.append(nxt)
    return neighbors


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def nearby_obstacle_penalty(grid, node):
    penalty = 0.0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = node[0] + dr, node[1] + dc
            if not in_bounds(grid, (nr, nc)) or grid[nr, nc] == 1:
                penalty += 0.08
    return penalty


def compute_path_cost(grid, path):
    if len(path) < 2:
        return 0.0
    cost = 0.0
    prev_dir = None
    for idx in range(1, len(path)):
        curr = path[idx]
        prev = path[idx - 1]
        direction = (curr[0] - prev[0], curr[1] - prev[1])
        cost += 1.0
        cost += nearby_obstacle_penalty(grid, curr)
        if prev_dir is not None and direction != prev_dir:
            cost += 0.2
        prev_dir = direction
    return cost


def bfs_search(grid, start, end):
    queue = deque([start])
    visited = {start}
    came_from = {}
    visited_order = []

    while queue:
        current = queue.popleft()
        visited_order.append(current)
        if current == end:
            path = reconstruct_path(came_from, current)
            return SearchResult("BFS", path, visited_order, len(path) - 1, True)

        for nxt in get_neighbors(grid, current):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                queue.append(nxt)

    return SearchResult("BFS", [], visited_order, float("inf"), False)


def a_star_search(grid, start, end, heuristic_weight=1.0, improved=False):
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, 0.0, start, None))
    came_from = {}
    best_cost = {(start, None): 0.0}
    node_best = {start: 0.0}
    visited_order = []
    expanded = set()

    while open_set:
        _, heuristic_value, g_cost, current, prev_dir = heapq.heappop(open_set)
        state = (current, prev_dir)
        if state in expanded:
            continue
        expanded.add(state)
        if current not in visited_order:
            visited_order.append(current)

        if current == end:
            path = reconstruct_path(came_from, current)
            name = "Improved A*" if improved else "A*"
            return SearchResult(name, path, visited_order, compute_path_cost(grid, path), True)

        for nxt in get_neighbors(grid, current):
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            step_cost = 1.0
            if improved:
                step_cost += nearby_obstacle_penalty(grid, nxt)
                if prev_dir is not None and direction != prev_dir:
                    step_cost += 0.1

            tentative_g = g_cost + step_cost
            nxt_state = (nxt, direction)
            if tentative_g >= best_cost.get(nxt_state, float("inf")):
                continue

            best_cost[nxt_state] = tentative_g
            if tentative_g < node_best.get(nxt, float("inf")):
                node_best[nxt] = tentative_g
                came_from[nxt] = current

            heuristic = manhattan(nxt, end)
            if improved:
                heuristic += 0.2 * nearby_obstacle_penalty(grid, nxt)
            f_score = tentative_g + heuristic_weight * heuristic
            heapq.heappush(open_set, (f_score, heuristic, tentative_g, nxt, direction))

    name = "Improved A*" if improved else "A*"
    return SearchResult(name, [], visited_order, float("inf"), False)


class QLearningPlanner:
    def __init__(self, grid, start, goal, alpha=0.2, gamma=0.95, epsilon=1.0):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((grid.shape[0], grid.shape[1], len(DIRECTIONS)), dtype=float)

    def valid_move(self, state, action_idx):
        dr, dc = DIRECTIONS[action_idx]
        nxt = (state[0] + dr, state[1] + dc)
        if not in_bounds(self.grid, nxt) or self.grid[nxt] == 1:
            return state, False
        return nxt, True

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(DIRECTIONS))
        return int(np.argmax(self.q_table[state[0], state[1]]))

    def step_reward(self, state, nxt, valid, seen):
        if not valid:
            return -8.0
        if nxt == self.goal:
            return 120.0

        reward = -1.0
        reward += 0.35 * (manhattan(state, self.goal) - manhattan(nxt, self.goal))
        reward -= 0.25 * nearby_obstacle_penalty(self.grid, nxt)
        if nxt in seen:
            reward -= 1.2
        return reward

    def train(self, episodes=2500, max_steps=400, epsilon_decay=0.996, min_epsilon=0.05):
        rewards = []
        for _ in range(episodes):
            state = self.start
            episode_reward = 0.0
            seen = {state}

            for _ in range(max_steps):
                action = self.choose_action(state)
                nxt, valid = self.valid_move(state, action)
                reward = self.step_reward(state, nxt, valid, seen)
                best_next = np.max(self.q_table[nxt[0], nxt[1]])
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state[0], state[1], action]
                self.q_table[state[0], state[1], action] += self.alpha * td_error
                episode_reward += reward
                state = nxt
                seen.add(state)
                if state == self.goal:
                    break

            self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)
            rewards.append(episode_reward)
        return rewards

    def extract_policy_path(self, max_steps=400):
        current = self.start
        path = [current]
        visited_order = [current]
        seen = {current}
        for _ in range(max_steps):
            if current == self.goal:
                return SearchResult(
                    "Q-Learning",
                    path,
                    visited_order,
                    compute_path_cost(self.grid, path),
                    True,
                )
            action = int(np.argmax(self.q_table[current[0], current[1]]))
            nxt, valid = self.valid_move(current, action)
            if not valid or nxt in seen:
                ranked_actions = np.argsort(self.q_table[current[0], current[1]])[::-1]
                found = False
                for candidate in ranked_actions:
                    trial, valid = self.valid_move(current, int(candidate))
                    if valid and trial not in seen:
                        nxt = trial
                        found = True
                        break
                if not found:
                    break
            current = nxt
            path.append(current)
            visited_order.append(current)
            seen.add(current)

        return SearchResult("Q-Learning", path, visited_order, compute_path_cost(self.grid, path), current == self.goal)


def potential_value(grid, node, goal, obstacle_positions):
    attractive = 1.2 * manhattan(node, goal)
    repulsive = 0.0
    node_arr = np.array(node, dtype=float)
    for obs in obstacle_positions:
        dist = np.linalg.norm(node_arr - obs)
        if 0.0 < dist < 2.5:
            repulsive += 10.0 * ((1.0 / dist) - (1.0 / 2.5)) ** 2
    return attractive + repulsive


def artificial_potential_field_search(grid, start, goal, max_steps=500):
    current = start
    path = [current]
    visited_order = [current]
    seen = {current: 1}
    obstacle_positions = np.argwhere(grid == 1)

    for _ in range(max_steps):
        if current == goal:
            return SearchResult("Artificial Potential Field", path, visited_order, compute_path_cost(grid, path), True)

        neighbors = get_neighbors(grid, current)
        if not neighbors:
            break

        candidates = []
        for nxt in neighbors:
            value = potential_value(grid, nxt, goal, obstacle_positions)
            if nxt in seen:
                value += 3.0 * seen[nxt]
            candidates.append((value, manhattan(nxt, goal), nxt))
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_value, _, best = candidates[0]

        # 简单逃逸机制: 若陷入局部极小值，则尝试次优可行方向
        current_value = potential_value(grid, current, goal, obstacle_positions)
        if best_value >= current_value and len(candidates) > 1:
            best = candidates[1][2]

        current = best
        path.append(current)
        visited_order.append(current)
        seen[current] = seen.get(current, 0) + 1

        if seen[current] > 4:
            break

    return SearchResult("Artificial Potential Field", path, visited_order, compute_path_cost(grid, path), current == goal)


def draw_grid(ax, grid, start, end, title, path=None, visited_order=None):
    display = np.copy(grid)
    if visited_order:
        for node in visited_order:
            if node not in (start, end) and display[node] == 0:
                display[node] = 2
    if path:
        for node in path:
            if node not in (start, end):
                display[node] = 3

    cmap = mcolors.ListedColormap(["white", "#23395B", "#BFD7EA", "#FFB703"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(display, cmap=cmap, norm=norm, origin="upper")
    ax.set_title(title, fontsize=11, pad=10)
    ax.text(start[1], start[0], "S", color="#D62828", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(end[1], end[0], "E", color="#111111", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="#999999", linestyle="-", linewidth=0.35, alpha=0.35)


def save_search_animation(grid, start, end, result, save_path, interval=120, frame_stride=1):
    fig, ax = plt.subplots(figsize=(7, 7))
    base_grid = np.copy(grid)
    # 0: empty, 1: obstacle, 2: explored, 3: final path, 4: current frontier node
    cmap = mcolors.ListedColormap(["white", "#23395B", "#BFD7EA", "#FFB703", "#219EBC"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    image = ax.imshow(base_grid, cmap=cmap, norm=norm, origin="upper")
    ax.text(start[1], start[0], "S", color="#D62828", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(end[1], end[0], "E", color="#111111", ha="center", va="center", fontsize=12, weight="bold")
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color="#999999", linestyle="-", linewidth=0.35, alpha=0.35)

    visited_frames = result.visited_order[::frame_stride] if result.visited_order else []
    if visited_frames and visited_frames[-1] != result.visited_order[-1]:
        visited_frames.append(result.visited_order[-1])
    path_frames = result.path[1::frame_stride] if result.path else []
    if path_frames and path_frames[-1] != result.path[-1]:
        path_frames.append(result.path[-1])
    total_frames = len(visited_frames) + len(path_frames) + 1

    def update(frame_idx):
        display = np.copy(base_grid)
        if frame_idx > 0:
            visited_limit = min(frame_idx, len(visited_frames))
            for node in visited_frames[:visited_limit]:
                if node not in (start, end) and display[node] == 0:
                    display[node] = 2
            if visited_limit > 0 and frame_idx <= len(visited_frames):
                current_node = visited_frames[visited_limit - 1]
                if current_node not in (start, end):
                    display[current_node] = 4
            path_progress = frame_idx - len(visited_frames)
            if path_progress > 0 and result.success:
                for node in path_frames[:path_progress]:
                    if node not in (start, end):
                        display[node] = 3
                if 0 < path_progress <= len(path_frames):
                    current_path_node = path_frames[path_progress - 1]
                    if current_path_node not in (start, end):
                        display[current_path_node] = 4
        image.set_data(display)
        if frame_idx < len(visited_frames):
            current_title = f"{result.name}: searching ({frame_idx + 1}/{len(visited_frames)})"
        else:
            path_idx = max(frame_idx - len(visited_frames), 0)
            current_title = f"{result.name}: path reconstruction ({min(path_idx, len(path_frames))}/{len(path_frames)})"
        ax.set_title(current_title, fontsize=12, pad=10)
        return [image]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval,
        blit=True,
        repeat=False,
    )
    anim.save(save_path, writer="pillow", dpi=140)
    plt.close(fig)


def plot_rewards(rewards, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, color="#2A9D8F", linewidth=1.2, alpha=0.8, label="Episode reward")
    window = 50
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), moving, color="#E76F51", linewidth=2.0, label="50-episode moving average")
    ax.set_title("Q-Learning Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def compare_on_map(grid, start, end, prefix):
    bfs_result = bfs_search(grid, start, end)
    astar_result = a_star_search(grid, start, end)
    improved_astar = a_star_search(grid, start, end, heuristic_weight=1.15, improved=True)

    q_agent = QLearningPlanner(grid, start, end)
    rewards = q_agent.train()
    q_result = q_agent.extract_policy_path()

    apf_result = artificial_potential_field_search(grid, start, end)
    results = [bfs_result, astar_result, improved_astar, q_result, apf_result]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    draw_grid(axes[0], grid, start, end, f"{prefix}: Map")
    for ax, result in zip(axes[1:], results):
        detail = "Success" if result.success else "Failed"
        title = f"{result.name}\n{detail}, visited={len(result.visited_order)}, steps={max(len(result.path)-1, 0)}"
        draw_grid(ax, grid, start, end, title, path=result.path, visited_order=result.visited_order)

    fig.tight_layout()
    compare_path = f"{prefix.lower().replace(' ', '_')}_comparison.png"
    fig.savefig(compare_path, dpi=180)
    plt.close(fig)

    reward_path = f"{prefix.lower().replace(' ', '_')}_q_learning_curve.png"
    plot_rewards(rewards, reward_path)
    return results, compare_path, reward_path


def export_dynamic_demos(grid, start, end, results, prefix):
    exported = []
    animate_targets = {"BFS", "A*", "Improved A*"}
    frame_stride = 1
    interval = 180 if grid.shape[0] <= 10 else 70
    for result in results:
        if result.name not in animate_targets:
            continue
        save_path = f"{prefix.lower().replace(' ', '_')}_{result.name.lower().replace('*', 'star').replace(' ', '_')}.gif"
        save_search_animation(grid, start, end, result, save_path, interval=interval, frame_stride=frame_stride)
        exported.append(save_path)
    return exported


def print_summary(results):
    print("\n算法对比结果")
    print("-" * 72)
    print(f"{'Algorithm':<28}{'Success':<10}{'Visited':<10}{'Steps':<10}{'Cost':<10}")
    print("-" * 72)
    for result in results:
        steps = max(len(result.path) - 1, 0)
        cost = f"{result.path_cost:.2f}" if result.success else "N/A"
        print(f"{result.name:<28}{str(result.success):<10}{len(result.visited_order):<10}{steps:<10}{cost:<10}")
    print("-" * 72)


def main():
    print("开始在原始 9x8 地图上测试 BFS、A*、改进算法与强化学习方法...")
    basic_grid, basic_start, basic_end = get_basic_map()
    basic_results, basic_fig, basic_reward_fig = compare_on_map(basic_grid, basic_start, basic_end, "Basic Map")
    basic_gifs = export_dynamic_demos(basic_grid, basic_start, basic_end, basic_results, "Basic Map")
    print_summary(basic_results)
    print(f"基础场景对比图已保存: {basic_fig}")
    print(f"基础场景训练曲线已保存: {basic_reward_fig}")
    print("基础场景动态展示已保存:", ", ".join(basic_gifs))

    print("\n开始在自主设计的 30x30 高难度场景上测试...")
    challenge_grid, challenge_start, challenge_end = get_challenge_map()
    challenge_results, challenge_fig, challenge_reward_fig = compare_on_map(
        challenge_grid, challenge_start, challenge_end, "Challenge 30x30"
    )
    challenge_gifs = export_dynamic_demos(
        challenge_grid, challenge_start, challenge_end, challenge_results, "Challenge 30x30"
    )
    print_summary(challenge_results)
    print(f"30x30 场景对比图已保存: {challenge_fig}")
    print(f"30x30 场景训练曲线已保存: {challenge_reward_fig}")
    print("30x30 场景动态展示已保存:", ", ".join(challenge_gifs))


if __name__ == "__main__":
    np.random.seed(7)
    main()
