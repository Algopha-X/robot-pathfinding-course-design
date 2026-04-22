import random
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("未检测到 PyTorch，请先在 robot_dqn 环境中安装 torch。") from exc

from deep_rl_pathfinder import (
    DeepRLPathPlanner,
    generate_complex_map,
    render_result,
)


MODEL_PATH = Path("dqn_30x30_model.pt")
TEST_MAZE_COUNT = 10
TEST_SEED_START = 100
MAP_SIZE = 30
SAVE_CASE_FIGURES = 3


def load_model_weights(model_path, planner):
    try:
        state_dict = torch.load(model_path, map_location=planner.device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=planner.device)

    planner.policy_net.load_state_dict(state_dict)
    planner.target_net.load_state_dict(state_dict)
    planner.policy_net.eval()
    planner.target_net.eval()
    planner.epsilon = 0.0


def evaluate_on_unseen_mazes(
    model_path,
    map_size=MAP_SIZE,
    maze_count=TEST_MAZE_COUNT,
    seed_start=TEST_SEED_START,
    save_case_figures=SAVE_CASE_FIGURES,
):
    base_grid, base_start, base_goal, _ = generate_complex_map(size=map_size, seed=42)
    planner = DeepRLPathPlanner(
        grid=base_grid,
        start=base_start,
        goal=base_goal,
        lr=1e-3,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.992,
        batch_size=64,
        target_update=20,
    )
    load_model_weights(model_path, planner)

    records = []
    success_count = 0
    dqn_steps_success = []
    bfs_steps_success = []

    for idx, seed in enumerate(range(seed_start, seed_start + maze_count), start=1):
        grid, start, goal, bfs_result = generate_complex_map(size=map_size, seed=seed)
        dqn_result = planner.rollout_policy(grid=grid, start=start, goal=goal)

        if dqn_result.success:
            success_count += 1
            dqn_steps_success.append(len(dqn_result.path) - 1)
            bfs_steps_success.append(len(bfs_result.path) - 1)

        records.append(
            {
                "index": idx,
                "seed": seed,
                "success": dqn_result.success,
                "dqn_steps": len(dqn_result.path) - 1 if dqn_result.path else -1,
                "bfs_steps": len(bfs_result.path) - 1,
                "final_position": dqn_result.extra.get("final_position") if dqn_result.extra else None,
            }
        )

        if idx <= save_case_figures:
            output_path = f"dqn_generalization_case_{idx:02d}.png"
            render_result(grid, start, goal, dqn_result, bfs_result, output_path)

    return {
        "maze_count": maze_count,
        "success_count": success_count,
        "success_rate": success_count / maze_count if maze_count else 0.0,
        "records": records,
        "avg_dqn_steps_success": np.mean(dqn_steps_success) if dqn_steps_success else None,
        "avg_bfs_steps_success": np.mean(bfs_steps_success) if bfs_steps_success else None,
    }


def print_report(summary):
    print("DQN 未见迷宫泛化测试结果")
    print("-" * 72)
    print(f"测试迷宫数量: {summary['maze_count']}")
    print(f"成功数量: {summary['success_count']}")
    print(f"成功率: {summary['success_rate']:.2%}")
    if summary["avg_dqn_steps_success"] is not None:
        print(f"成功样本平均 DQN 路径长度: {summary['avg_dqn_steps_success']:.2f}")
        print(f"成功样本平均 BFS 路径长度: {summary['avg_bfs_steps_success']:.2f}")
    else:
        print("当前没有成功样本，暂时无法统计平均路径长度。")

    print("-" * 72)
    for record in summary["records"]:
        print(
            f"case={record['index']:02d} | seed={record['seed']} | "
            f"success={record['success']} | dqn_steps={record['dqn_steps']} | "
            f"bfs_steps={record['bfs_steps']} | final={record['final_position']}"
        )
    print("-" * 72)
    print("前 3 个测试样例对比图已保存为 dqn_generalization_case_01.png 等文件。")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if not MODEL_PATH.exists():
        raise SystemExit(
            "未找到 dqn_30x30_model.pt，请先运行 deep_rl_pathfinder.py 完成训练并保存模型。"
        )

    summary = evaluate_on_unseen_mazes(MODEL_PATH)
    print_report(summary)


if __name__ == "__main__":
    main()
