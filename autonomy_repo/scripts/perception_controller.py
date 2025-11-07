#!/usr/bin/env python3

import rclpy
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl

class PerceptionController(BaseController):
    def __init__(self, node_name: str = "perception_controller") -> None:
        super().__init__(node_name)
        self.declare_parameter("active", True)

    @property
    def active(self) -> bool:
        return bool(self.get_parameter("active").value)
    
    def compute_control(self) -> TurtleBotControl:
        ctrl = TurtleBotControl()
        if self.active:
            if hasattr(self, "_stop_start_time"):
                delattr(self, "_stop_start_time")
            ctrl.omega = 0.5
            return ctrl
        current_time = self.get_clock().now().nanoseconds / 1e9
        if not hasattr(self, "_stop_start_time"):
            self._stop_start_time = current_time
        if current_time - self._stop_start_time < 5.0:
            ctrl.omega = 0.0
            return ctrl
        self.set_parameters([rclpy.Parameter("active", value=True)])
        delattr(self, "_stop_start_time")
        ctrl.omega = 0.5
        return ctrl

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = PerceptionController()  # instantiate the perception controller node
    rclpy.spin(node)    # Use ROS2 built-in scheduler for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context