# Nav2 for the Unity CAR, no AMCL. Needs tools/unity_nav2_bridge.py running (TF + /odom).
#   ros2 launch config/nav2/nav2_unity.launch.py map:=$HOME/nico_ws/scenes/flat_mix/nav2/map.yaml
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
PARAMS = os.path.join(HERE, "nav2_params.yaml")
NODES = ["map_server", "planner_server", "controller_server", "behavior_server", "bt_navigator"]


def generate_launch_description():
    m = LaunchConfiguration("map")
    nodes = [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output="screen",
             parameters=[PARAMS, {"yaml_filename": m}]),
        Node(package="nav2_planner", executable="planner_server", name="planner_server", output="screen",
             parameters=[PARAMS]),
        Node(package="nav2_controller", executable="controller_server", name="controller_server", output="screen",
             parameters=[PARAMS]),
        Node(package="nav2_behaviors", executable="behavior_server", name="behavior_server", output="screen",
             parameters=[PARAMS]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", name="bt_navigator", output="screen",
             parameters=[PARAMS]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
             output="screen", parameters=[{"use_sim_time": False, "autostart": True, "node_names": NODES}]),
    ]
    return LaunchDescription([DeclareLaunchArgument("map", description="map.yaml from tools/nav2_map.py")] + nodes)
