#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _build_explorer(context, *args, **kwargs):
    explorer_node = Node(
        package="autonomy_repo",
        executable="frontier_explorer.py",
        name="frontier_explorer",
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="screen",
    )
    delay_value = float(LaunchConfiguration("explorer_start_delay").perform(context))
    if delay_value <= 0.0:
        return [explorer_node]
    return [TimerAction(period=delay_value, actions=[explorer_node])]


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")

    rviz_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
        ),
        launch_arguments={
            "config": PathJoinSubstitution(
                [FindPackageShare("autonomy_repo"), "rviz", "default.rviz"]
            ),
            "use_sim_time": use_sim_time,
        }.items(),
    )

    rviz_goal_relay = Node(
        package="asl_tb3_lib",
        executable="rviz_goal_relay.py",
        parameters=[{"output_channel": "/cmd_nav"}],
        output="screen",
    )

    state_publisher = Node(
        package="asl_tb3_lib",
        executable="state_publisher.py",
        output="screen",
    )

    navigator = Node(
        package="autonomy_repo",
        executable="navigator.py",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("explorer_start_delay", default_value="2.0"),
            rviz_launch,
            rviz_goal_relay,
            state_publisher,
            navigator,
            OpaqueFunction(function=_build_explorer),
        ]
    )
