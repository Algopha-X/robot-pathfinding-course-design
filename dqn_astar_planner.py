import argparse
import heapq
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass

temp_dir = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(temp_dir, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", temp_dir)

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:
    raise SystemExit("PyTorch is required. Please run: pip install torch") from exc

from pathfinder import (
    DIRECTIONS,
    SearchResult,
    a_star_search,
    compute_path_cost,
    get_challenge_map,
    in_bounds,
    manhattan,
    nearby_obstacle_penalty,
    reconstruct_path,
)
from deep_rl_pathfinder import generate_complex_map


MODEL_PATH = "dqn_astar_cost_model.pt"
CURVE_PATH = "dqn_astar_training_curve.png"
RESULT_PATH = "dqn_astar_comparison.png"
EVAL_METRICS_PATH = "dqn_astar_eval_metrics.png"
IMPROVEMENT_PATH = "dqn_astar_improvement_rate.png"
QUALITY_WEIGHTS = {
    "steps": 1.0,
    "turns": 0.8,
    "safety": 7.0,
    "visited": 0.01,
}


@dataclass
class DQNAStarConfig:
    episodes: int = 500
    pretrain_maps: int = 300
    pretrain_epochs: int = 3
    batch_size: int = 64
    replay_capacity: int = 50000
    gamma: float = 0.98
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.993
    target_update: int = 25
    optimize_interval: int = 2
    expert_replay: int = 2
    max_steps: int | None = None
    dqn_weight: float = 0.8
    safety_weight: float = 3.2
    turn_weight: float = 0.2
    heuristic_weight: float = 1.05
    seed: int = 42
    map_size: int = 30
    train_seed_start: int = 1000
    pretrain_seed_start: int = 100
    eval_seed_start: int = 5000
    random_maps: bool = True


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class CostDQN(nn.Module):
    def __init__(self, height, width, action_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(7, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * height * width, 256),
            nn.ReLU(),
            nn.Linear(256, action_count),
        )

    def forward(self, x):
        return self.head(self.features(x))


class CostLearningEnv:
    def __init__(self, grid, start, goal, max_steps=None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape
        self.max_steps = max_steps or self.height * self.width
        self.reset()

    def reset(self):
        self.agent = self.start
        self.prev_dir = None
        self.steps = 0
        self.seen = {self.start}
        return encode_state(self.grid, self.agent, self.goal, self.prev_dir)

    def step(self, action):
        self.steps += 1
        dr, dc = DIRECTIONS[action]
        nxt = (self.agent[0] + dr, self.agent[1] + dc)
        direction = (dr, dc)
        old_dist = manhattan(self.agent, self.goal)
        reward = -0.35
        done = False
        valid = in_bounds(self.grid, nxt) and self.grid[nxt] == 0

        if not valid:
            reward -= 6.0
            nxt = self.agent
            direction = self.prev_dir
        else:
            new_dist = manhattan(nxt, self.goal)
            reward += 1.1 * (old_dist - new_dist)
            reward -= 1.2 * nearby_obstacle_penalty(self.grid, nxt)
            if self.prev_dir is not None and direction != self.prev_dir:
                reward -= 0.65
            if nxt in self.seen:
                reward -= 1.0

            self.agent = nxt
            self.prev_dir = direction
            self.seen.add(nxt)

        if self.agent == self.goal:
            reward += 80.0
            done = True
        elif self.steps >= self.max_steps:
            reward -= 12.0
            done = True

        return encode_state(self.grid, self.agent, self.goal, self.prev_dir), reward, done


def encode_state(grid, current, goal, prev_dir):
    obstacle = grid.astype(np.float32)
    agent = np.zeros_like(obstacle, dtype=np.float32)
    target = np.zeros_like(obstacle, dtype=np.float32)
    prev_channels = np.zeros((4, grid.shape[0], grid.shape[1]), dtype=np.float32)

    agent[current] = 1.0
    target[goal] = 1.0
    if prev_dir in DIRECTIONS:
        prev_channels[DIRECTIONS.index(prev_dir), :, :] = 1.0

    return np.concatenate(
        [obstacle[None, :, :], agent[None, :, :], target[None, :, :], prev_channels],
        axis=0,
    )


class DQNAStarPlanner:
    def __init__(self, grid, start, goal, config):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = CostDQN(grid.shape[0], grid.shape[1], len(DIRECTIONS)).to(self.device)
        self.target_net = CostDQN(grid.shape[0], grid.shape[1], len(DIRECTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.epsilon = config.epsilon_start

    def set_problem(self, grid, start, goal):
        if grid.shape != self.grid.shape:
            raise ValueError("DQN-A* currently expects all training maps to have the same shape.")
        self.grid = grid
        self.start = start
        self.goal = goal

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(len(DIRECTIONS))
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            return int(torch.argmax(self.policy_net(tensor), dim=1).item())

    def optimize(self):
        if len(self.replay) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)
        states_t = torch.from_numpy(states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def supervised_update_from_path(self, result, repeat=1):
        samples = self.expert_samples_from_path(result)
        if not samples:
            return None, None

        states = np.stack([item[0] for item in samples])
        actions = np.array([item[1] for item in samples], dtype=np.int64)
        losses = []
        accuracies = []

        for _ in range(repeat):
            order = np.random.permutation(len(actions))
            for start_idx in range(0, len(actions), self.config.batch_size):
                batch_idx = order[start_idx:start_idx + self.config.batch_size]
                states_t = torch.from_numpy(states[batch_idx]).float().to(self.device)
                actions_t = torch.from_numpy(actions[batch_idx]).long().to(self.device)
                logits = self.policy_net(states_t)
                loss = nn.functional.cross_entropy(logits, actions_t)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
                self.optimizer.step()

                preds = torch.argmax(logits, dim=1)
                accuracies.append(float((preds == actions_t).float().mean().item()))
                losses.append(float(loss.item()))

        return float(np.mean(losses)), float(np.mean(accuracies))

    def expert_samples_from_path(self, result):
        samples = []
        if not result.success or len(result.path) < 2:
            return samples

        prev_dir = None
        for idx in range(1, len(result.path)):
            current = result.path[idx - 1]
            nxt = result.path[idx]
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            action = DIRECTIONS.index(direction)
            state = encode_state(self.grid, current, self.goal, prev_dir)
            samples.append((state, action))
            prev_dir = direction
        return samples

    def pretrain_with_improved_astar(self):
        if self.config.pretrain_maps <= 0 or self.config.pretrain_epochs <= 0:
            return [], []

        states = []
        actions = []
        for idx in range(self.config.pretrain_maps):
            if self.config.random_maps:
                grid, start, goal, _ = generate_complex_map(
                    size=self.config.map_size,
                    seed=self.config.pretrain_seed_start + idx,
                )
                self.set_problem(grid, start, goal)

            expert = a_star_search(self.grid, self.start, self.goal, heuristic_weight=1.15, improved=True)
            for state, action in self.expert_samples_from_path(expert):
                states.append(state)
                actions.append(action)

            if (idx + 1) % 50 == 0 or idx == 0:
                print(f"Pretrain data {idx + 1:04d}/{self.config.pretrain_maps} maps | samples={len(states)}")

        if not states:
            return [], []

        states = np.stack(states)
        actions = np.array(actions, dtype=np.int64)
        losses = []
        accuracies = []
        dataset_size = len(actions)

        for epoch in range(1, self.config.pretrain_epochs + 1):
            order = np.random.permutation(dataset_size)
            epoch_losses = []
            correct = 0
            seen = 0

            for start_idx in range(0, dataset_size, self.config.batch_size):
                batch_idx = order[start_idx:start_idx + self.config.batch_size]
                states_t = torch.from_numpy(states[batch_idx]).float().to(self.device)
                actions_t = torch.from_numpy(actions[batch_idx]).long().to(self.device)

                logits = self.policy_net(states_t)
                loss = nn.functional.cross_entropy(logits, actions_t)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
                self.optimizer.step()

                preds = torch.argmax(logits, dim=1)
                correct += int((preds == actions_t).sum().item())
                seen += len(batch_idx)
                epoch_losses.append(float(loss.item()))

            self.target_net.load_state_dict(self.policy_net.state_dict())
            avg_loss = float(np.mean(epoch_losses))
            accuracy = correct / max(seen, 1)
            losses.append(avg_loss)
            accuracies.append(accuracy)
            print(f"Pretrain epoch {epoch:02d}/{self.config.pretrain_epochs} | loss={avg_loss:.4f} | acc={accuracy * 100:.1f}%")

        return losses, accuracies

    def path_transition_reward(
        self,
        current,
        nxt,
        prev_dir,
        step_idx,
        path_len,
        baseline_metrics,
        result_metrics,
        is_expert=False,
    ):
        direction = (nxt[0] - current[0], nxt[1] - current[1])
        reward = -0.25
        reward += 0.8 * (manhattan(current, self.goal) - manhattan(nxt, self.goal))
        reward -= self.config.safety_weight * nearby_obstacle_penalty(self.grid, nxt)

        if prev_dir is not None and direction != prev_dir:
            reward -= self.config.turn_weight

        if is_expert:
            reward += 1.2

        if step_idx == path_len - 1 and nxt == self.goal:
            reward += 45.0 if is_expert else 28.0
            if baseline_metrics["steps"] > 0:
                reward += 6.0 * (baseline_metrics["steps"] - result_metrics["steps"]) / baseline_metrics["steps"]
            if baseline_metrics["turns"] > 0:
                reward += 5.0 * (baseline_metrics["turns"] - result_metrics["turns"]) / baseline_metrics["turns"]
            if baseline_metrics["safety"] > 0:
                reward += 8.0 * (baseline_metrics["safety"] - result_metrics["safety"]) / baseline_metrics["safety"]

        return float(reward)

    def learn_from_astar_path(self, result, baseline, is_expert=False):
        if not result.success or len(result.path) < 2:
            return 0.0, 0

        result_metrics = path_metrics(self.grid, result)
        baseline_metrics = path_metrics(self.grid, baseline)
        total_reward = 0.0
        pushed = 0
        prev_dir = None

        for idx in range(1, len(result.path)):
            current = result.path[idx - 1]
            nxt = result.path[idx]
            direction = (nxt[0] - current[0], nxt[1] - current[1])
            action = DIRECTIONS.index(direction)
            done = nxt == self.goal
            state = encode_state(self.grid, current, self.goal, prev_dir)
            next_state = encode_state(self.grid, nxt, self.goal, direction)
            reward = self.path_transition_reward(
                current,
                nxt,
                prev_dir,
                idx,
                len(result.path) - 1,
                baseline_metrics,
                result_metrics,
                is_expert=is_expert,
            )
            self.replay.push(state, action, reward, next_state, done)
            total_reward += reward
            pushed += 1
            prev_dir = direction

        return total_reward, pushed

    def train(self):
        rewards = []
        losses = []
        success_history = []

        for episode in range(1, self.config.episodes + 1):
            if self.config.random_maps:
                train_grid, train_start, train_goal, _ = generate_complex_map(
                    size=self.config.map_size,
                    seed=self.config.train_seed_start + episode,
                )
                self.set_problem(train_grid, train_start, train_goal)

            baseline = a_star_search(self.grid, self.start, self.goal, heuristic_weight=1.15, improved=True)
            result = dqn_astar_search(self.grid, self.start, self.goal, self, self.config)
            episode_reward, pushed = self.learn_from_astar_path(result, baseline)
            for _ in range(self.config.expert_replay):
                expert_reward, expert_pushed = self.learn_from_astar_path(baseline, baseline, is_expert=True)
                episode_reward += expert_reward
                pushed += expert_pushed

            ce_loss, _ = self.supervised_update_from_path(baseline, repeat=max(1, self.config.expert_replay))
            if ce_loss is not None:
                losses.append(ce_loss)

            reached_goal = result.success

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            rewards.append(episode_reward)
            success_history.append(1.0 if reached_goal else 0.0)

            if episode % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if episode % 20 == 0 or episode == 1:
                recent_success = np.mean(success_history[-20:]) * 100.0
                recent_loss = float(np.mean(losses[-50:])) if losses else 0.0
                metrics = path_metrics(self.grid, result)
                print(
                    f"Episode {episode:04d} | reward={episode_reward:8.2f} | "
                    f"success20={recent_success:5.1f}% | loss={recent_loss:.4f} | "
                    f"steps={metrics['steps']:3d} | turns={metrics['turns']:3d} | "
                    f"risk={metrics['safety']:.2f} | map_seed={self.config.train_seed_start + episode}"
                )

        return rewards, losses, success_history

    def q_value(self, grid, current, goal, prev_dir, action):
        state = encode_state(grid, current, goal, prev_dir)
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            return float(self.policy_net(tensor)[0, action].item())

    def action_preference(self, grid, current, goal, prev_dir, action):
        state = encode_state(grid, current, goal, prev_dir)
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            probs = torch.softmax(self.policy_net(tensor), dim=1)
            return float(probs[0, action].item())

    def save(self, path=MODEL_PATH):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path=MODEL_PATH):
        if not os.path.exists(path):
            return False
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        return True


def dqn_astar_search(grid, start, goal, planner, config):
    open_set = []
    heapq.heappush(open_set, (0.0, 0.0, 0.0, start, None))
    came_from = {}
    best_cost = {(start, None): 0.0}
    node_best = {start: 0.0}
    visited_order = []
    expanded = set()

    while open_set:
        _, heuristic_value, g_cost, current, prev_dir = heapq.heappop(open_set)
        state_key = (current, prev_dir)
        if state_key in expanded:
            continue
        expanded.add(state_key)
        if current not in visited_order:
            visited_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return SearchResult(
                "DQN-A*",
                path,
                visited_order,
                compute_path_cost(grid, path),
                True,
                {"expanded_states": len(expanded)},
            )

        for action, (dr, dc) in enumerate(DIRECTIONS):
            nxt = (current[0] + dr, current[1] + dc)
            if not in_bounds(grid, nxt) or grid[nxt] == 1:
                continue

            direction = (dr, dc)
            preference = planner.action_preference(grid, current, goal, prev_dir, action)
            learned_bonus = config.dqn_weight * preference
            step_cost = 1.0
            step_cost += config.safety_weight * nearby_obstacle_penalty(grid, nxt)
            if prev_dir is not None and direction != prev_dir:
                step_cost += config.turn_weight
            step_cost = max(0.05, step_cost - learned_bonus)

            tentative_g = g_cost + step_cost
            nxt_state = (nxt, direction)
            if tentative_g >= best_cost.get(nxt_state, float("inf")):
                continue

            best_cost[nxt_state] = tentative_g
            if tentative_g < node_best.get(nxt, float("inf")):
                node_best[nxt] = tentative_g
                came_from[nxt] = current

            heuristic = manhattan(nxt, goal)
            heuristic += 0.25 * nearby_obstacle_penalty(grid, nxt)
            f_score = tentative_g + config.heuristic_weight * heuristic
            heapq.heappush(open_set, (f_score, heuristic, tentative_g, nxt, direction))

    return SearchResult("DQN-A*", [], visited_order, float("inf"), False)


def quality_score(metrics):
    if np.isinf(metrics["cost"]):
        return float("inf")
    return (
        QUALITY_WEIGHTS["steps"] * metrics["steps"]
        + QUALITY_WEIGHTS["turns"] * metrics["turns"]
        + QUALITY_WEIGHTS["safety"] * metrics["safety"]
        + QUALITY_WEIGHTS["visited"] * metrics.get("visited", 0.0)
    )


def guarded_dqn_astar_search(grid, start, goal, planner, config, baseline=None):
    baseline = baseline or a_star_search(grid, start, goal, heuristic_weight=1.15, improved=True)
    learned = dqn_astar_search(grid, start, goal, planner, config)
    baseline_metrics = path_metrics(grid, baseline)
    learned_metrics = path_metrics(grid, learned)
    baseline_metrics["visited"] = len(baseline.visited_order)
    learned_metrics["visited"] = len(learned.visited_order)

    baseline_score = quality_score(baseline_metrics)
    learned_score = quality_score(learned_metrics)
    if learned.success and learned_score <= baseline_score:
        learned.name = "DQN-A* Guarded"
        learned.extra = learned.extra or {}
        learned.extra["quality_score"] = learned_score
        learned.extra["guarded"] = False
        learned.extra["baseline_score"] = baseline_score
        return learned

    guarded = SearchResult(
        "DQN-A* Guarded",
        baseline.path,
        baseline.visited_order,
        baseline.path_cost,
        baseline.success,
        {
            "quality_score": baseline_score,
            "guarded": True,
            "rejected_dqn_score": learned_score,
        },
    )
    return guarded


def path_metrics(grid, result):
    if not result.success or len(result.path) < 2:
        return {"steps": 0, "turns": 0, "safety": 0.0, "cost": float("inf")}

    turns = 0
    safety = 0.0
    prev_dir = None
    for idx in range(1, len(result.path)):
        prev = result.path[idx - 1]
        curr = result.path[idx]
        direction = (curr[0] - prev[0], curr[1] - prev[1])
        if prev_dir is not None and direction != prev_dir:
            turns += 1
        prev_dir = direction
        safety += nearby_obstacle_penalty(grid, curr)

    return {
        "steps": len(result.path) - 1,
        "turns": turns,
        "safety": safety,
        "cost": result.path_cost,
    }


def moving_average(values, window):
    if not values:
        return np.array([])
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    prefix = values[: window - 1]
    smooth = np.convolve(values, kernel, mode="valid")
    return np.concatenate([prefix, smooth])


def draw_result(ax, grid, start, goal, result, title):
    display = np.copy(grid)
    for node in result.visited_order:
        if node not in (start, goal) and display[node] == 0:
            display[node] = 2
    for node in result.path:
        if node not in (start, goal):
            display[node] = 3

    cmap = mcolors.ListedColormap(["white", "#23395B", "#BFD7EA", "#FFB703"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(display, cmap=cmap, norm=norm, origin="upper")
    ax.scatter(start[1], start[0], c="#D62828", s=50, marker="s", label="Start")
    ax.scatter(goal[1], goal[0], c="#111111", s=50, marker="s", label="Goal")
    if result.path:
        ys = [p[0] for p in result.path]
        xs = [p[1] for p in result.path]
        ax.plot(xs, ys, color="#E76F51", linewidth=2.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def render_training_curve(rewards, losses, success_history, output_path, pretrain_losses=None, pretrain_accuracies=None):
    episodes = np.arange(1, len(rewards) + 1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    axes[0].plot(episodes, rewards, color="#2A9D8F", linewidth=1.1)
    axes[0].plot(episodes, moving_average(rewards, 30), color="#E76F51", linewidth=2.0)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("A*-in-the-loop DQN Cost Learning Reward")
    axes[0].grid(alpha=0.25)

    if losses:
        loss_x = np.arange(1, len(losses) + 1)
        axes[1].plot(loss_x, losses, color="#577590", linewidth=0.9)
        axes[1].plot(loss_x, moving_average(losses, 80), color="#F3722C", linewidth=2.0)
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Optimization Loss")
    axes[1].grid(alpha=0.25)

    axes[2].plot(episodes, moving_average(success_history, 20) * 100.0, color="#43AA8B", linewidth=2.0)
    axes[2].set_ylabel("Success %")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("20-Episode DQN-A* Search Success Rate")
    axes[2].grid(alpha=0.25)

    pretrain_losses = pretrain_losses or []
    pretrain_accuracies = pretrain_accuracies or []
    if pretrain_losses:
        pre_x = np.arange(1, len(pretrain_losses) + 1)
        axes[3].plot(pre_x, pretrain_losses, marker="o", color="#577590", label="CE loss")
        if pretrain_accuracies:
            ax_acc = axes[3].twinx()
            ax_acc.plot(pre_x, np.array(pretrain_accuracies) * 100.0, marker="s", color="#F3722C", label="accuracy")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.set_ylim(0, 100)
        axes[3].set_xlabel("Pretrain Epoch")
        axes[3].set_ylabel("Loss")
        axes[3].set_title("Improved A* Expert Pretraining")
        axes[3].grid(alpha=0.25)
    else:
        axes[3].axis("off")

    fig.tight_layout(pad=1.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def render_comparison(grid, start, goal, results, output_path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        metrics = path_metrics(grid, result)
        guarded = ""
        if result.extra and result.extra.get("guarded"):
            guarded = ", guarded"
        title = (
            f"{result.name}{guarded}\n"
            f"steps={metrics['steps']}, turns={metrics['turns']}, "
            f"risk={metrics['safety']:.2f}, visited={len(result.visited_order)}"
        )
        draw_result(ax, grid, start, goal, result, title)

    fig.tight_layout(pad=1.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def print_comparison(grid, results):
    print("\nDQN-A* comparison")
    print("-" * 82)
    print(f"{'Algorithm':<16}{'Success':<10}{'Steps':<10}{'Turns':<10}{'Safety':<12}{'Visited':<10}{'Cost':<10}")
    print("-" * 82)
    for result in results:
        metrics = path_metrics(grid, result)
        cost_text = f"{metrics['cost']:.2f}" if result.success else "N/A"
        print(
            f"{result.name:<16}{str(result.success):<10}{metrics['steps']:<10}"
            f"{metrics['turns']:<10}{metrics['safety']:<12.2f}"
            f"{len(result.visited_order):<10}{cost_text:<10}"
        )
    print("-" * 82)


def evaluate_generalization(planner, config, num_mazes, seed_start):
    records = []
    totals = {
        "A*": {"success": 0, "steps": 0.0, "turns": 0.0, "safety": 0.0, "visited": 0.0, "guarded": 0},
        "Improved A*": {"success": 0, "steps": 0.0, "turns": 0.0, "safety": 0.0, "visited": 0.0, "guarded": 0},
        "DQN-A* Raw": {"success": 0, "steps": 0.0, "turns": 0.0, "safety": 0.0, "visited": 0.0, "guarded": 0},
        "DQN-A* Guarded": {"success": 0, "steps": 0.0, "turns": 0.0, "safety": 0.0, "visited": 0.0, "guarded": 0},
    }

    first_case = None
    for idx in range(num_mazes):
        seed = seed_start + idx
        grid, start, goal, _ = generate_complex_map(size=config.map_size, seed=seed)
        astar = a_star_search(grid, start, goal)
        improved = a_star_search(grid, start, goal, heuristic_weight=1.15, improved=True)
        dqn_raw = dqn_astar_search(grid, start, goal, planner, config)
        dqn_raw.name = "DQN-A* Raw"
        dqn_guarded = guarded_dqn_astar_search(grid, start, goal, planner, config, baseline=improved)
        results = [astar, improved, dqn_raw, dqn_guarded]
        if first_case is None:
            first_case = (grid, start, goal, results)

        row = {"seed": seed}
        for result in results:
            metrics = path_metrics(grid, result)
            row[result.name] = metrics | {"success": result.success, "visited": len(result.visited_order)}
            stats = totals[result.name]
            if result.success:
                stats["success"] += 1
                stats["steps"] += metrics["steps"]
                stats["turns"] += metrics["turns"]
                stats["safety"] += metrics["safety"]
                stats["visited"] += len(result.visited_order)
            if result.extra and result.extra.get("guarded"):
                stats["guarded"] += 1
        records.append(row)

    return records, totals, first_case


def print_eval_summary(totals, num_mazes):
    print("\nUnseen-map generalization")
    print("-" * 98)
    print(
        f"{'Algorithm':<16}{'Success':<12}{'AvgSteps':<12}{'AvgTurns':<12}"
        f"{'AvgSafety':<12}{'AvgVisited':<12}{'Guarded':<10}"
    )
    print("-" * 98)
    for name, stats in totals.items():
        success = stats["success"]
        denom = max(success, 1)
        print(
            f"{name:<16}{success}/{num_mazes:<9}"
            f"{stats['steps'] / denom:<12.2f}"
            f"{stats['turns'] / denom:<12.2f}"
            f"{stats['safety'] / denom:<12.2f}"
            f"{stats['visited'] / denom:<12.2f}"
            f"{stats['guarded']:<10}"
        )
    print("-" * 98)


def averaged_eval_metrics(totals):
    averaged = {}
    for name, stats in totals.items():
        denom = max(stats["success"], 1)
        averaged[name] = {
            "success_rate": stats["success"],
            "steps": stats["steps"] / denom,
            "turns": stats["turns"] / denom,
            "safety": stats["safety"] / denom,
            "visited": stats["visited"] / denom,
        }
    return averaged


def render_eval_metrics(totals, num_mazes, output_path):
    metrics = averaged_eval_metrics(totals)
    names = list(metrics.keys())
    colors = ["#577590", "#43AA8B", "#F3722C", "#277DA1"]
    panels = [
        ("success_rate", "Success Rate (%)", lambda value: value / num_mazes * 100.0),
        ("steps", "Average Path Steps", lambda value: value),
        ("turns", "Average Turns", lambda value: value),
        ("safety", "Average Obstacle-Risk Cost", lambda value: value),
        ("visited", "Average Expanded Nodes", lambda value: value),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (key, title, transform) in zip(axes, panels):
        values = [transform(metrics[name][key]) for name in names]
        bars = ax.bar(names, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.78,
        "Lower is better for steps, turns,\nrisk cost, and expanded nodes.\nHigher is better for success rate.",
        fontsize=11,
        va="top",
    )

    fig.suptitle(f"DQN-A* Generalization Metrics on {num_mazes} Unseen Maps", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=1.5)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render_improvement_rates(totals, output_path):
    metrics = averaged_eval_metrics(totals)
    target_name = "DQN-A* Raw" if "DQN-A* Raw" in metrics else "DQN-A*"
    target = metrics[target_name]
    baselines = ["A*", "Improved A*"]
    metric_keys = [
        ("steps", "Path Steps"),
        ("turns", "Turns"),
        ("safety", "Risk Cost"),
        ("visited", "Expanded Nodes"),
    ]

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(16, 4.8), constrained_layout=True)
    for ax, (key, title) in zip(axes, metric_keys):
        values = []
        labels = []
        for baseline in baselines:
            base_value = metrics[baseline][key]
            improvement = 0.0
            if base_value > 0:
                improvement = (base_value - target[key]) / base_value * 100.0
            values.append(improvement)
            labels.append(f"vs {baseline}")

        colors = ["#43AA8B" if value >= 0 else "#E76F51" for value in values]
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Improvement (%)")
        ax.grid(axis="y", alpha=0.25)
        max_abs = max(abs(value) for value in values)
        if max_abs < 1.0:
            ax.set_ylim(-1.0, 1.0)
        else:
            margin = max(1.0, max_abs * 0.18)
            ax.set_ylim(min(values) - margin, max(values) + margin)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            offset = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            offset = offset if value >= 0 else -offset
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.1f}%",
                ha="center",
                va=va,
                fontsize=9,
            )

    fig.suptitle(f"{target_name} Relative Improvement on Unseen Maps", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN-guided A* cost planner.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--pretrain-maps", type=int, default=300)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="continue from dqn_astar_cost_model.pt")
    parser.add_argument("--eval-only", action="store_true", help="load model and only run DQN-A* evaluation")
    parser.add_argument("--single-map", action="store_true", help="train only on pathfinder.get_challenge_map()")
    parser.add_argument("--eval-mazes", type=int, default=10, help="number of unseen random maps for evaluation")
    parser.add_argument("--pretrain-seed-start", type=int, default=100)
    parser.add_argument("--train-seed-start", type=int, default=1000)
    parser.add_argument("--eval-seed-start", type=int, default=5000)
    parser.add_argument("--optimize-interval", type=int, default=2)
    parser.add_argument("--expert-replay", type=int, default=2)
    parser.add_argument("--dqn-weight", type=float, default=0.8)
    parser.add_argument("--safety-weight", type=float, default=3.2)
    parser.add_argument("--turn-weight", type=float, default=0.2)
    parser.add_argument("--heuristic-weight", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DQNAStarConfig(
        episodes=args.episodes,
        pretrain_maps=args.pretrain_maps,
        pretrain_epochs=args.pretrain_epochs,
        dqn_weight=args.dqn_weight,
        safety_weight=args.safety_weight,
        turn_weight=args.turn_weight,
        heuristic_weight=args.heuristic_weight,
        seed=args.seed,
        pretrain_seed_start=args.pretrain_seed_start,
        train_seed_start=args.train_seed_start,
        eval_seed_start=args.eval_seed_start,
        optimize_interval=max(args.optimize_interval, 1),
        expert_replay=max(args.expert_replay, 0),
        random_maps=not args.single_map,
    )
    set_seed(config.seed)

    if config.random_maps:
        grid, start, goal, _ = generate_complex_map(size=config.map_size, seed=config.train_seed_start)
    else:
        grid, start, goal = get_challenge_map()
    planner = DQNAStarPlanner(grid, start, goal, config)

    if args.resume or args.eval_only:
        loaded = planner.load(MODEL_PATH)
        print(f"Loaded existing model: {loaded}")
        if args.eval_only and not loaded:
            raise SystemExit(f"No model found at {MODEL_PATH}. Train first or remove --eval-only.")

    if not args.eval_only:
        mode = "A*-in-the-loop random-map generalization" if config.random_maps else "A*-in-the-loop single fixed map"
        pretrain_losses, pretrain_accuracies = [], []
        if not args.resume:
            print("Pretraining DQN cost model from Improved A* expert paths...")
            pretrain_losses, pretrain_accuracies = planner.pretrain_with_improved_astar()
        print(f"Training DQN cost model for DQN-A* ({mode})...")
        rewards, losses, success_history = planner.train()
        planner.save(MODEL_PATH)
        render_training_curve(
            rewards,
            losses,
            success_history,
            CURVE_PATH,
            pretrain_losses=pretrain_losses,
            pretrain_accuracies=pretrain_accuracies,
        )
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved training curve: {CURVE_PATH}")

    records, totals, first_case = evaluate_generalization(
        planner,
        config,
        num_mazes=args.eval_mazes,
        seed_start=config.eval_seed_start,
    )
    print_eval_summary(totals, args.eval_mazes)
    render_eval_metrics(totals, args.eval_mazes, EVAL_METRICS_PATH)
    render_improvement_rates(totals, IMPROVEMENT_PATH)

    grid, start, goal, results = first_case
    print_comparison(grid, results)
    render_comparison(grid, start, goal, results, RESULT_PATH)
    print(f"Saved evaluation metrics figure: {EVAL_METRICS_PATH}")
    print(f"Saved relative improvement figure: {IMPROVEMENT_PATH}")
    print(f"Saved comparison figure: {RESULT_PATH}")


if __name__ == "__main__":
    main()
