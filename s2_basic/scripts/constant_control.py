#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


class ConstantControl(Node):
    def __init__(self) -> None:
        super().__init__("constant_control")

        # Create publisher
        self.cc_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Create timer
        self.cc_timer = self.create_timer(0.2, self.cc_callback)

        self.kill_sub = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

    def cc_callback(self) -> None:
        #construct the cc message
        msg = Twist()
        msg.linear.x = 10.0
        msg.angular.z = 10.0

        self.cc_pub.publish(msg)

    def kill_callback(self) -> None:
        self.cc_timer.cancel()
        stop_msg = Twist()
        
        self.cc_pub.publish(stop_msg)


if __name__ == "__main__":
    rclpy.init()
    node = ConstantControl()
    rclpy.spin(node)
    rclpy.shutdown()