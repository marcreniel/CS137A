#!/usr/bin/env python3
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from nav_msgs.msg import OccupancyGrid
from asl_tb3_msgs.msg import TurtleBotState
from std_msgs.msg import Bool
from asl_tb3_lib.grids import StochOccupancyGrid2D

import numpy as np
import typing as T
"""Frontier exploration node for TurtleBot3."""

class Frontier_Explorer(Node):
    def __init__(self):
        super().__init__("Explorer")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.nav_success = None
        self.state = None
        self.state_flag = 0 
        self.map_flag = 0
        self.stop_flag = 0 # 0 means no stop sign detected
        self.start_explore_flag = 0
        self.prev_time = 0
        self.current_time = 0
        self.declare_parameter("active", True)
        self._delayed_timer = None

        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_cb, 10)
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.detector_sub = self.create_subscription(Bool, "/detector_bool", self.detector_cb, 10)
    
        
    def explore(self):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.
        """
        print("Running Explore\n")
        if self.state is None or self.occupancy is None:
            self.get_logger().warn("Explore called before state/map ready; skipping")
            return
        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        current_state = np.array([self.state.x, self.state.y])
        
        occupied_mask = np.where(self.occupancy.probs >= 0.5, 1, 0)
        unknown_mask = np.where(self.occupancy.probs == -1, 1, 0)
        unoccupied_mask = np.where((self.occupancy.probs >= 0) & (self.occupancy.probs < 0.5), 1, 0)

        def box_sum(mask: np.ndarray) -> np.ndarray:
            # Fast box filter via summed-area table to avoid SciPy dependency
            half = window_size // 2
            padded = np.pad(mask, ((half, half), (half, half)), mode="constant")
            integral = padded.cumsum(axis=0).cumsum(axis=1)
            # add leading zero row/col so we can use exclusive/inclusive indexing safely
            integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
            h, w = mask.shape
            k = window_size
            y1 = np.arange(h)
            y2 = y1 + k
            x1 = np.arange(w)
            x2 = x1 + k
            A = integral[np.ix_(y1, x1)]
            B = integral[np.ix_(y1, x2)]
            C = integral[np.ix_(y2, x1)]
            D = integral[np.ix_(y2, x2)]
            return D - B - C + A

        occupied = box_sum(occupied_mask)
        unoccupied = box_sum(unoccupied_mask)
        unknown = box_sum(unknown_mask)

        area = window_size * window_size
        frontier_mask = np.where(
            (unknown >= 0.2 * area) & (occupied == 0) & (unoccupied >= 0.3 * area), 1, 0
        )
        frontier_states = np.transpose(np.nonzero(np.transpose(frontier_mask)))
        frontier_states = self.occupancy.grid2state(frontier_states)
        
        if len(frontier_states) == 0:
            print("Finished exploring")
            return

        frontier_state  = np.argmin(np.linalg.norm(frontier_states-current_state,axis=1))
        frontier_state = frontier_states[frontier_state,:]

       
        msg = TurtleBotState()
        msg.x, msg.y = frontier_state
        msg.theta = 0.0
        self.get_logger().info(f"Publishing frontier goal to /cmd_nav: x={msg.x:.2f}, y={msg.y:.2f}")
        self.cmd_nav_pub.publish(msg)
		

    
    def nav_success_cb(self, msg: Bool) -> None:
        """ Callback triggered when nav_success is updated

        Args:
            msg (Bool): updated nav_success message
        """
        # current_time = self.get_clock().now().nanoseconds/1e9
        if self.active and self.state is not None and self.occupancy is not None: 
            self.explore()
        elif self.active:
            self.get_logger().warn("nav_success received but state/map not ready yet")
        # else:           
        #     if (current_time-self.prev_time)<=5:
        #         pass
        #     else:
        #         self.explore()
        #         self.prev_time = 0
        #         self.set_parameters([rclpy.Parameter("active",value = True)])

        # self.explore()       
        print("i got a message")
        self.nav_success = msg

    def state_callback(self, msg: TurtleBotState) -> None:
        """ Callback triggered when nav_success is updated

        Args:
            msg (Bool): updated nav_success message
        """
        self.state_flag = 1
        self.state=msg

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        print("Obtained Map")
        if self.occupancy is None:
            self.map_flag=1
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )
        if self.map_flag==1 and self.state_flag==1:
            print("Exploring (with delay)")
            if self._delayed_timer is None:
                self._delayed_timer = self.create_timer(3.0, self._delayed_explore_once)
            self.map_flag=0

    def _delayed_explore_once(self):
        # run explore once after initial delay, then cancel timer
        if self._delayed_timer is not None:
            self._delayed_timer.cancel()
            self._delayed_timer = None
        self.explore()

    @property  
    def active(self)-> bool:
        return self.get_parameter("active").value
    
         
    def detector_cb(self, msg:Bool):
        if msg.data and (self.stop_flag == 0):
            self.set_parameters([rclpy.Parameter("active",value = False)])
            if self.prev_time == 0: # guards in case another stop sign is detected
                self.prev_time = self.get_clock().now().nanoseconds/1e9
            self.stop_sign_wait()# call function to deal with stop sign detection
        elif (self.current_time - self.prev_time) <= 13:
            self.stop_sign_wait()

    def stop_sign_wait(self):
        # this if check may be redundant
        # if self.active: # if stop sign has been detected and robot has not been stopped
        #     self.active = self.set_parameters([rclpy.Parameter("active",value = False)])
        #     msg = self.state
        #     self.cmd_nav_pub.publish(self.state)
            
        # else: # if it's not detected
        self.current_time = self.get_clock().now().nanoseconds/1e9
        if (self.current_time - self.prev_time) <= 5:
            # msg = self.state
            self.cmd_nav_pub.publish(self.state)
            print('stop sign seen within 5 seconds')
        elif (self.current_time - self.prev_time) <= 10: # some buffer time in between
            if (self.start_explore_flag == 0):
                self.explore()
                self.start_explore_flag = 1
            self.set_parameters([rclpy.Parameter("active",value = True)])
            self.stop_flag = 1 # stop sign has been detected and being dealt with (end process)
            print('stop sign seen within 10 seconds')

        else:
            # self.explore()
            self.stop_flag = 0
            self.prev_time = 0
        
        self.start_explore_flag = 0
        
            
        # return msg

    
            

def main():
    rclpy.init(args=None)
    print("main")
    node = Frontier_Explorer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()