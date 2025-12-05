# 137AH

## Frontier Exploration Section Prep

Launch the full navigation + frontier exploration stack (RViz, navigator, and explorer) with:

```bash
ros2 launch autonomy_repo frontier_exploration.launch.py
```

The launch file accepts the following arguments:

- `use_sim_time` (default `true`): switch between simulation and hardware clocks.
- `explorer_start_delay` (default `2.0` seconds): wait time before spawning the explorer node so that the navigator and map streams are ready.

The frontier explorer node publishes autonomous navigation goals on `/cmd_nav` using the heuristics from HW4 Section Prep (unknown ≥ 20%, occupied = 0%, free ≥ 30% within a configurable window). Parameters such as window size and thresholds can be overridden via ROS parameters on the `frontier_explorer` node.
