#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from asl_tb3_lib.navigation import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool


class FrontierExplorer(Node):

    def __init__(self) -> None:
        super().__init__("frontier_explorer")
        self.declare_parameters(
            "",
            [
                ("window_size", 7),
                ("unknown_ratio_threshold", 0.2),
                ("free_ratio_threshold", 0.3),
                ("occupancy_threshold", 50),
                ("min_goal_distance", 0.35),
                ("post_nav_delay_sec", 1.0),
                ("state_topic", "/state"),
                ("map_topic", "/map"),
                ("nav_success_topic", "/nav_success"),
                ("nav_cmd_topic", "/cmd_nav"),
            ],
        )

        self._param = lambda key: self.get_parameter(key).value

        self._current_state: Optional[TurtleBotState] = None
        self._occupancy: Optional[StochOccupancyGrid2D] = None
        self._frontier_mask: Optional[np.ndarray] = None
        self._awaiting_nav: bool = False
        self._last_goal_time: Optional[Time] = None
        self._last_nav_complete: Optional[Time] = None
        self._active_goal_grid: Optional[Tuple[int, int]] = None
        self._reported_completion: bool = False
        self._frontier_blacklist: set[Tuple[int, int]] = set()

        self._nav_pub = self.create_publisher(
            TurtleBotState,
            self._param("nav_cmd_topic"),
            10,
        )
        self._state_sub = self.create_subscription(
            TurtleBotState,
            self._param("state_topic"),
            self._state_callback,
            10,
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid,
            self._param("map_topic"),
            self._map_callback,
            1,
        )
        self._nav_success_sub = self.create_subscription(
            Bool,
            self._param("nav_success_topic"),
            self._nav_status_callback,
            10,
        )
        self.create_timer(0.5, self._timer_callback)

    def _state_callback(self, msg: TurtleBotState) -> None:
        self._current_state = msg
        self.try_dispatch_goal()

    def _map_callback(self, msg: OccupancyGrid) -> None:
        map_data = np.asarray(msg.data, dtype=float)
        size_xy = np.array([msg.info.width, msg.info.height], dtype=int)
        origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y], dtype=float)
        window = self._sanitized_window_size()

        try:
            self._occupancy = StochOccupancyGrid2D(
                msg.info.resolution,
                size_xy,
                origin_xy,
                window,
                map_data,
            )
        except ValueError as exc:
            self.get_logger().warn(f"Failed to build occupancy grid: {exc}")
            self._frontier_mask = None
            return

        self._frontier_mask = self._compute_frontiers(self._occupancy)
        self._apply_blacklist()
        self._reported_completion = False
        self.try_dispatch_goal()

    def _nav_status_callback(self, msg: Bool) -> None:
        if not self._awaiting_nav:
            return
        self.get_logger().info(
            f"Navigation {'succeeded' if msg.data else 'failed'}; computing next frontier target"
        )
        if self._active_goal_grid is not None:
            (self._frontier_blacklist.add if not msg.data else self._frontier_blacklist.discard)(
                self._active_goal_grid
            )
            if self._frontier_mask is not None:
                self._frontier_mask[self._active_goal_grid] = False
        self._awaiting_nav = False
        self._active_goal_grid = None
        self._last_nav_complete = self.get_clock().now()
        self.try_dispatch_goal()

    def _timer_callback(self) -> None:
        self.try_dispatch_goal()

    def _sanitized_window_size(self) -> int:
        window = int(self._param("window_size"))
        window = max(3, window)
        if window % 2 == 0:
            window += 1
        return window

    def _compute_frontiers(self, occupancy: StochOccupancyGrid2D) -> np.ndarray:
        grid = occupancy.probs
        if grid.size == 0:
            return np.zeros_like(grid, dtype=bool)
        window = self._sanitized_window_size()
        if min(grid.shape) < window:
            self.get_logger().warn(
                f"Occupancy grid ({grid.shape}) smaller than window size {window}; skipping frontier detection",
                throttle_duration_sec=5.0,
            )
            return np.zeros_like(grid, dtype=bool)

        occ_thresh = float(self._param("occupancy_threshold"))
        unknown_mask = grid < 0
        occupied_mask = grid >= occ_thresh
        free_mask = (~unknown_mask) & (~occupied_mask)

        unknown_ratio = self._window_sum(unknown_mask, window) / float(window * window)
        occupied_count = self._window_sum(occupied_mask, window).astype(int)
        free_ratio = self._window_sum(free_mask, window) / float(window * window)

        candidates = free_mask
        candidates &= unknown_ratio >= float(self._param("unknown_ratio_threshold"))
        candidates &= occupied_count == 0
        candidates &= free_ratio >= float(self._param("free_ratio_threshold"))

        return candidates

    @staticmethod
    def _window_sum(mask: np.ndarray, window: int) -> np.ndarray:
        if mask.shape[0] < window or mask.shape[1] < window:
            return np.zeros_like(mask, dtype=float)
        pad = window // 2
        padded = np.pad(mask.astype(float), pad_width=pad, mode="constant", constant_values=0)
        windows = sliding_window_view(padded, (window, window))
        return windows.sum(axis=(-2, -1))

    def try_dispatch_goal(self) -> None:
        if (
            self._awaiting_nav
            or self._current_state is None
            or self._occupancy is None
            or self._frontier_mask is None
        ):
            return
        if not np.any(self._frontier_mask):
            if not self._reported_completion:
                self.get_logger().info("No remaining frontier cells; exploration complete")
                self._reported_completion = True
            return

        now = self.get_clock().now()
        delay = float(self._param("post_nav_delay_sec"))
        if self._last_nav_complete is not None:
            elapsed = (now - self._last_nav_complete).nanoseconds / 1e9
            if elapsed < delay:
                return

        selection = self._select_next_goal()
        if selection is None:
            return
        goal, grid_idx = selection
        self._nav_pub.publish(goal)
        self._awaiting_nav = True
        self._active_goal_grid = grid_idx
        self._last_goal_time = now
        self.get_logger().info(
            f"Commanding frontier goal at ({goal.x:.2f}, {goal.y:.2f}) heading {goal.theta:.2f} rad",
        )

    def _select_next_goal(self) -> Optional[Tuple[TurtleBotState, Tuple[int, int]]]:
        assert self._frontier_mask is not None
        frontier_indices = np.argwhere(self._frontier_mask)
        if frontier_indices.size == 0 or self._current_state is None or self._occupancy is None:
            return None

        state_xy = np.array([self._current_state.x, self._current_state.y], dtype=float)
        grid_xy = frontier_indices[:, [1, 0]].astype(float)
        world_xy = grid_xy * self._occupancy.resolution + self._occupancy.origin_xy

        deltas = world_xy - state_xy
        distances = np.linalg.norm(deltas, axis=1)
        if distances.size == 0:
            return None

        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])
        min_goal_distance = float(self._param("min_goal_distance"))
        if min_distance < min_goal_distance:
            cell = tuple(frontier_indices[min_idx])
            self._frontier_mask[cell] = False
            self._frontier_blacklist.add(cell)
            return None

        heading = math.atan2(deltas[min_idx, 1], deltas[min_idx, 0])
        goal = TurtleBotState()
        goal.x = float(world_xy[min_idx, 0])
        goal.y = float(world_xy[min_idx, 1])
        goal.theta = float(heading)
        return goal, tuple(frontier_indices[min_idx])

    def _apply_blacklist(self) -> None:
        if self._frontier_mask is None or not self._frontier_blacklist:
            return
        h, w = self._frontier_mask.shape
        for cell in list(self._frontier_blacklist):
            r, c = cell
            if 0 <= r < h and 0 <= c < w:
                self._frontier_mask[r, c] = False
            else:
                self._frontier_blacklist.discard(cell)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
