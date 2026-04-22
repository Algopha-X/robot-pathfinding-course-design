import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

temp_dir = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(temp_dir, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", temp_dir)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise SystemExit(
        "未检测到 PyTorch。请先执行 `pip install torch`，然后再运行本脚本。"
    ) from exc


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


def in_bounds(grid, node):
    r, c = node
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def compute_path_cost(path):
    if len(path) < 2:
        return 0.0
    return float(len(path) - 1)


def bfs_reference(grid, start, goal):
    queue = deque([start])
    visited = {start}
    came_from = {}
    visited_order = []

    while queue:
        current = queue.popleft()
        visited_order.append(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            return SearchResult("BFS-Reference", path, visited_order, compute_path_cost(path), True)

        for dr, dc in DIRECTIONS:
            nxt = (current[0] + dr, current[1] + dc)
            if in_bounds(grid, nxt) and grid[nxt] == 0 and nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                queue.append(nxt)

    return SearchResult("BFS-Reference", [], visited_order, float("inf"), False)


def generate_complex_map(size=30, seed=42):
    """
    生成一个 30x30 默认复杂障碍场景:
    - 外边框封闭
    - 多层竖墙和横墙
    - 保留门洞确保整体可达
    - 增加局部封闭区，提升策略学习难度
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=int)

    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    vertical_cols = list(range(6, size - 6, 6))
    horizontal_rows = list(range(7, size - 7, 7))

    for col in vertical_cols:
        grid[1:-1, col] = 1
        gate_rows = sorted(rng.choice(np.arange(3, size - 3), size=4, replace=False))
        for row in gate_rows:
            grid[max(1, row - 1): min(size - 1, row + 2), col] = 0

    for row in horizontal_rows:
        grid[row, 1:-1] = 1
        gate_cols = sorted(rng.choice(np.arange(3, size - 3), size=4, replace=False))
        for col in gate_cols:
            grid[row, max(1, col - 1): min(size - 1, col + 2)] = 0

    boxes = [
        (6, 6, 5, 6),
        (11, 17, 6, 5),
        (18, 7, 5, 7),
    ]
    for top, left, height, width in boxes:
        bottom = min(size - 2, top + height)
        right = min(size - 2, left + width)
        grid[top:bottom, left] = 1
        grid[top:bottom, right] = 1
        grid[top, left:right + 1] = 1
        grid[bottom, left:right + 1] = 1
        gate_side = rng.integers(0, 4)
        if gate_side == 0:
            gate = rng.integers(left + 1, right)
            grid[top, gate] = 0
        elif gate_side == 1:
            gate = rng.integers(left + 1, right)
            grid[bottom, gate] = 0
        elif gate_side == 2:
            gate = rng.integers(top + 1, bottom)
            grid[gate, left] = 0
        else:
            gate = rng.integers(top + 1, bottom)
            grid[gate, right] = 0

    start = (1, 1)
    goal = (size - 2, size - 2)
    grid[start] = 0
    grid[goal] = 0

    reference = bfs_reference(grid, start, goal)
    if not reference.success:
        return generate_complex_map(size=size, seed=seed + 1)
    return grid, start, goal, reference


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 256),
            nn.ReLU(),
            nn.Linear(256, len(DIRECTIONS)),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class GridWorldEnv:
    def __init__(self, grid, start, goal, max_steps=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.size = grid.shape[0]
        self.max_steps = max_steps or grid.shape[0] * grid.shape[1] * 2
        self.reset()

    def reset(self):
        self.agent = self.start
        self.steps = 0
        self.visited = {self.start}
        self.visited_order = [self.start]
        return self.encode_state()

    def encode_state(self):
        obstacle = self.grid.astype(np.float32)
        agent_map = np.zeros_like(obstacle, dtype=np.float32)
        goal_map = np.zeros_like(obstacle, dtype=np.float32)
        visited_map = np.zeros_like(obstacle, dtype=np.float32)

        agent_map[self.agent] = 1.0
        goal_map[self.goal] = 1.0
        for cell in self.visited:
            visited_map[cell] = 1.0

        return np.stack([obstacle, agent_map, goal_map, visited_map], axis=0)

    def step(self, action_idx):
        self.steps += 1
        dr, dc = DIRECTIONS[action_idx]
        nxt = (self.agent[0] + dr, self.agent[1] + dc)

        reward = -0.2
        done = False
        reached_goal = False
        valid = in_bounds(self.grid, nxt) and self.grid[nxt] == 0
        old_distance = abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])

        if not valid:
            reward -= 3.0
            nxt = self.agent
        else:
            new_distance = abs(nxt[0] - self.goal[0]) + abs(nxt[1] - self.goal[1])
            if new_distance < old_distance:
                reward += 0.8
            else:
                reward -= 0.3

            if nxt in self.visited:
                reward -= 0.6
            else:
                reward += 0.2

            self.agent = nxt
            self.visited.add(nxt)
            self.visited_order.append(nxt)

        if self.agent == self.goal:
            reward += 40.0
            done = True
            reached_goal = True

        truncated = False
        if self.steps >= self.max_steps:
            done = True
            truncated = True

        return self.encode_state(), reward, done, {
            "position": self.agent,
            "reached_goal": reached_goal,
            "truncated": truncated and not reached_goal,
        }


class DeepRLPathPlanner:
    def __init__(
        self,
        grid,
        start,
        goal,
        lr=1e-3,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        target_update=20,
        buffer_capacity=20000,
        device=None,
    ):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.eval_grid = np.copy(grid)
        self.eval_start = start
        self.eval_goal = goal
        self.map_size = grid.shape[0]
        env_max_steps = min(grid.shape[0] * grid.shape[1], 1200)
        self.env = GridWorldEnv(grid, start, goal, max_steps=env_max_steps)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        grid_size = grid.shape[0]
        self.policy_net = DQN(grid_size).to(self.device)
        self.target_net = DQN(grid_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

    def set_problem(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        env_max_steps = min(grid.shape[0] * grid.shape[1], 1200)
        self.env = GridWorldEnv(grid, start, goal, max_steps=env_max_steps)

    def load_weights(self, model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            return False

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()
        return True

    def select_action(self, state, greedy=False):
        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(len(DIRECTIONS))

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def train(
        self,
        episodes=180,
        random_maps=False,
        map_seed_start=1000,
        eval_interval=20,
        eval_mazes=5,
        eval_seed_start=2000,
    ):
        rewards_history = []
        loss_history = []
        best_path = None
        eval_history = []

        for episode in range(1, episodes + 1):
            if random_maps:
                train_grid, train_start, train_goal, _ = generate_complex_map(
                    size=self.map_size,
                    seed=map_seed_start + episode,
                )
                self.set_problem(train_grid, train_start, train_goal)

            state = self.env.reset()
            episode_reward = 0.0
            losses = []

            for _ in range(self.env.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                loss = self.optimize()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                episode_reward += reward
                if done:
                    break

            rewards_history.append(episode_reward)
            loss_history.append(float(np.mean(losses)) if losses else 0.0)

            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            rollout = self.rollout_policy(
                max_steps=min(self.eval_grid.shape[0] * self.eval_grid.shape[1], 1200),
                grid=self.eval_grid,
                start=self.eval_start,
                goal=self.eval_goal,
            )
            if rollout.success:
                if best_path is None or len(rollout.path) < len(best_path.path):
                    best_path = rollout

            eval_summary = None
            if eval_interval and episode % eval_interval == 0:
                eval_summary = evaluate_generalization(
                    self,
                    num_mazes=eval_mazes,
                    start_seed=eval_seed_start,
                )
                eval_history.append(
                    {
                        "episode": episode,
                        "success_rate": eval_summary["success_count"] / eval_summary["num_mazes"],
                    }
                )

            if episode % 10 == 0:
                message = (
                    f"Episode {episode:03d} | reward={episode_reward:7.2f} | "
                    f"reward_ma20={moving_average(rewards_history, 20)[-1]:7.2f} | "
                    f"loss_ma20={moving_average(loss_history, 20)[-1]:.4f} | "
                    f"epsilon={self.epsilon:.3f} | success={rollout.success}"
                )
                if eval_summary is not None:
                    message += (
                        f" | eval={eval_summary['success_count']}/{eval_summary['num_mazes']}"
                    )
                print(message)

        return rewards_history, loss_history, best_path, eval_history

    def rollout_policy(self, max_steps=None, grid=None, start=None, goal=None):
        if grid is None:
            env = self.env
            start = self.start
            goal = self.goal
        else:
            env_max_steps = min(grid.shape[0] * grid.shape[1], 1200)
            env = GridWorldEnv(grid, start, goal, max_steps=env_max_steps)

        state = env.reset()
        max_steps = max_steps or env.max_steps
        path = [start]
        visited_order = [start]
        seen_positions = {start}
        reached_goal = start == goal

        for _ in range(max_steps):
            action = self.select_action(state, greedy=True)
            next_state, _, done, info = env.step(action)
            pos = info["position"]
            if pos != path[-1]:
                path.append(pos)
            if pos not in seen_positions:
                visited_order.append(pos)
                seen_positions.add(pos)
            reached_goal = reached_goal or info.get("reached_goal", False) or pos == goal
            state = next_state
            if done:
                break

        success = reached_goal or env.agent == goal or path[-1] == goal
        return SearchResult(
            "DQN",
            path,
            visited_order,
            compute_path_cost(path) if success else float("inf"),
            success,
            {
                "final_position": path[-1],
                "reached_goal": reached_goal,
                "steps": len(path) - 1,
            },
        )


def evaluate_generalization(planner, num_mazes=10, start_seed=100):
    records = []
    success_count = 0
    total_ratio = 0.0

    for seed in range(start_seed, start_seed + num_mazes):
        grid, start, goal, bfs_result = generate_complex_map(size=planner.map_size, seed=seed)
        rollout = planner.rollout_policy(grid=grid, start=start, goal=goal)
        if rollout.success:
            success_count += 1
            total_ratio += (len(rollout.path) - 1) / max(len(bfs_result.path) - 1, 1)
        records.append((seed, rollout.success, len(rollout.path) - 1, len(bfs_result.path) - 1))

    avg_ratio = total_ratio / success_count if success_count else float("inf")
    return {
        "success_count": success_count,
        "num_mazes": num_mazes,
        "avg_path_ratio": avg_ratio,
        "records": records,
    }


def moving_average(values, window):
    if not values:
        return np.array([])
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    prefix = values[: window - 1]
    smooth = np.convolve(values, kernel, mode="valid")
    return np.concatenate([prefix, smooth])


def render_result(grid, start, goal, dqn_result, bfs_result, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    titles = [
        f"DQN Path ({'Success' if dqn_result.success else 'Failed'})",
        f"BFS Reference (length={len(bfs_result.path) - 1})",
    ]
    results = [dqn_result, bfs_result]

    for ax, title, result in zip(axes, titles, results):
        ax.imshow(grid, cmap="gray_r")
        ax.scatter(start[1], start[0], c="green", s=60, label="Start")
        ax.scatter(goal[1], goal[0], c="red", s=60, label="Goal")
        if result.path:
            ys = [node[0] for node in result.path]
            xs = [node[1] for node in result.path]
            ax.plot(xs, ys, color="#1f77b4", linewidth=2)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_training_curve(rewards, losses, output_path, eval_points=None):
    episodes = np.arange(1, len(rewards) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(episodes, rewards, color="#2c7fb8", linewidth=1.4)
    axes[0].plot(
        episodes,
        moving_average(rewards, window=20),
        color="#e34a33",
        linewidth=2.0,
        label="20-episode moving average",
    )
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("DQN Training Reward Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].plot(episodes, losses, color="#d95f0e", linewidth=1.2)
    axes[1].plot(
        episodes,
        moving_average(losses, window=20),
        color="#756bb1",
        linewidth=2.0,
        label="20-episode moving average",
    )
    axes[1].set_ylabel("Loss")
    axes[1].set_title("DQN Training Loss Curve")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    eval_points = eval_points or []
    if eval_points:
        eval_episodes = [item["episode"] for item in eval_points]
        eval_rates = [item["success_rate"] * 100.0 for item in eval_points]
        axes[2].plot(
            eval_episodes,
            eval_rates,
            color="#238b45",
            linewidth=1.8,
            marker="o",
            markersize=4,
        )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title("Generalization Eval on Fixed Unseen Mazes")
    axes[2].set_ylim(0, 100)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    model_path = "dqn_30x30_model.pt"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    grid, start, goal, bfs_result = generate_complex_map(size=30, seed=42)
    planner = DeepRLPathPlanner(
        grid=grid,
        start=start,
        goal=goal,
        lr=1e-3,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        batch_size=64,
        target_update=20,
    )

    resumed = planner.load_weights(model_path)
    if resumed:
        # 继续训练时适度恢复探索，避免模型被过早收敛到局部策略。
        planner.epsilon = max(planner.epsilon, 0.20)
        print(f"检测到已有模型 {model_path}，将从当前参数继续训练。")

    print("开始训练 DQN 路径规划器（多迷宫训练模式）...")
    rewards, losses, best_result, eval_history = planner.train(
        episodes=800,
        random_maps=True,
        map_seed_start=1000,
        eval_interval=20,
        eval_mazes=5,
        eval_seed_start=2000,
    )

    last_result = planner.rollout_policy()
    final_result = best_result if best_result is not None else last_result
    generalization = evaluate_generalization(planner, num_mazes=10, start_seed=100)

    print(f"BFS 最短路径长度: {len(bfs_result.path) - 1}")
    print(f"最终一次贪心回放是否成功: {last_result.success}")
    if final_result.success:
        print(f"DQN 规划成功，路径长度: {len(final_result.path) - 1}")
    else:
        print("DQN 当前训练轮数下尚未稳定收敛到目标点，可增加训练轮数继续优化。")
    print(
        f"10 张新迷宫测试成功率: "
        f"{generalization['success_count']}/{generalization['num_mazes']}"
    )
    if generalization["success_count"] > 0:
        print(f"成功样本相对 BFS 的平均路径倍率: {generalization['avg_path_ratio']:.2f}")

    render_result(
        grid,
        start,
        goal,
        final_result,
        bfs_result,
        output_path="dqn_30x30_result.png",
    )
    render_training_curve(
        rewards,
        losses,
        output_path="dqn_30x30_training_curve.png",
        eval_points=eval_history,
    )

    torch.save(planner.policy_net.state_dict(), model_path)
    print("结果图已保存为 dqn_30x30_result.png")
    print("训练曲线已保存为 dqn_30x30_training_curve.png")
    print("模型权重已保存为 dqn_30x30_model.pt")


if __name__ == "__main__":
    main()
