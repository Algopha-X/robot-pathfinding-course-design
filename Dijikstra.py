import os
import heapq

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import matplotlib.pyplot as plt

from pathfinder import (
    SearchResult,
    a_star_search,
    bfs_search,
    compute_path_cost,
    draw_grid,
    get_basic_map,
    get_challenge_map,
    get_neighbors,
    reconstruct_path,
)

matplotlib.use("Agg")


def dijkstra_search(grid, start, end):
    """
    Dijkstra 算法:
    适用于非负权图。当前栅格每一步代价相同，因此它会退化成“按累计距离扩展”的一致代价搜索。
    相比 A*，Dijkstra 不使用终点方向启发信息，因此更稳健，但通常会扩展更多节点。
    """
    priority_queue = [(0.0, start)]
    distances = {start: 0.0}
    came_from = {}
    visited_order = []
    settled = set()

    while priority_queue:
        current_cost, current = heapq.heappop(priority_queue)
        if current in settled:
            continue

        settled.add(current)
        visited_order.append(current)

        if current == end:
            path = reconstruct_path(came_from, current)
            return SearchResult("Dijkstra", path, visited_order, compute_path_cost(grid, path), True)

        for neighbor in get_neighbors(grid, current):
            new_cost = current_cost + 1.0
            if new_cost < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(priority_queue, (new_cost, neighbor))

    return SearchResult("Dijkstra", [], visited_order, float("inf"), False)


def plot_comparison(grid, start, end, results, save_path, title_prefix):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        title = f"{result.name}\nSuccess={result.success}, visited={len(result.visited_order)}"
        draw_grid(ax, grid, start, end, title, path=result.path, visited_order=result.visited_order)
    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def print_result_table(results, scene_name):
    print(f"\n{scene_name} 结果对比")
    print("-" * 72)
    print(f"{'Algorithm':<16}{'Success':<10}{'Visited':<10}{'Steps':<10}{'Cost':<10}")
    print("-" * 72)
    for result in results:
        steps = max(len(result.path) - 1, 0)
        cost = f"{result.path_cost:.2f}" if result.success else "N/A"
        print(f"{result.name:<16}{str(result.success):<10}{len(result.visited_order):<10}{steps:<10}{cost:<10}")
    print("-" * 72)


def print_analysis():
    print(
        """
局限性分析
1. Dijkstra 不使用启发式信息，虽然能保证最短路，但在大地图或障碍复杂时会扩展大量无关节点，搜索效率偏低。
2. 当前实现默认每一步代价相同，无法直接体现转弯风险、贴障危险度、地形代价等更细粒度约束。
3. 当环境动态变化时，Dijkstra 需要重新规划，实时性不如增量式搜索或学习型方法。

改进点
1. 引入 A* 或加权 A*，利用终点方向信息减少盲目扩展。
2. 将代价函数设计为“步长 + 转向代价 + 贴障代价”，使规划结果更平滑、更安全。
3. 对动态场景可进一步考虑 D* Lite、LPA* 等增量式算法。
4. 对超大规模地图，可结合双向搜索、分层规划或强化学习策略网络提升效率。
        """.strip()
    )


def main():
    basic_grid, basic_start, basic_end = get_basic_map()
    challenge_grid, challenge_start, challenge_end = get_challenge_map()

    basic_results = [
        bfs_search(basic_grid, basic_start, basic_end),
        dijkstra_search(basic_grid, basic_start, basic_end),
        a_star_search(basic_grid, basic_start, basic_end),
    ]
    challenge_results = [
        bfs_search(challenge_grid, challenge_start, challenge_end),
        dijkstra_search(challenge_grid, challenge_start, challenge_end),
        a_star_search(challenge_grid, challenge_start, challenge_end),
    ]

    print_result_table(basic_results, "基础 9x8 场景")
    print_result_table(challenge_results, "复杂 30x30 场景")
    print_analysis()

    plot_comparison(
        basic_grid,
        basic_start,
        basic_end,
        basic_results,
        "dijkstra_basic_comparison.png",
        "Basic Map: BFS vs Dijkstra vs A*",
    )
    plot_comparison(
        challenge_grid,
        challenge_start,
        challenge_end,
        challenge_results,
        "dijkstra_challenge_comparison.png",
        "Challenge 30x30: BFS vs Dijkstra vs A*",
    )
    print("\n对比图已保存: dijkstra_basic_comparison.png, dijkstra_challenge_comparison.png")


if __name__ == "__main__":
    main()
