#!/usr/bin/env python3
"""
Planner Benchmarking Script
- Runs DWB, TEB, and RPP planners in Warehouse A map
- Each trial = round trip (Start -> Table -> Start)
- Uses current AMCL pose as the start/end location
- Logs round-trip time and success/failure into timestamped CSV

Usage:
    python3 benchmark_planners.py --trials 5 --output results/planner_results.csv
    python3 benchmark_planners.py --trials 5 --output results/dwb_results.csv
    python3 benchmark_planners.py --trials 5 --output results/teb_local_planner_results.csv
    python3 benchmark_planners.py --trials 5 --output results/regulated_pure_pursuit_controller_results.csv
    dwb_results
    teb_local_planner
    RegulatedPurePursuitController
"""

import os
import csv
import time
import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

# === Waypoints (Warehouse A) ===
TABLE = (-12.7, 6.5, 0.0)

# TABLE = (-3.0, 1.0, 0.0)
# TABLE = (1.5, 13.1, 0.0)  #do this for tests

class PlannerBenchmark(Node):
    def __init__(self, trials, output_path):
        super().__init__('planner_benchmark')
        self.navigator = BasicNavigator()
        self.current_pose = None

        # # Subscribe to AMCL pose
        # self.create_subscription(
        #     PoseWithCovarianceStamped,
        #     '/amcl_pose',
        #     self.amcl_callback,
        #     10
        # )
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_callback,
            qos_profile
        )

        # Ensure results dir exists
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = os.path.join(os.path.dirname(output_path),
                                  f"planner_results_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        self.output_file = os.path.join(result_dir, os.path.basename(output_path))

        self.trials = trials

        # Open CSV file
        self.csv_file = open(self.output_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "planner", "trial",
            "start_x", "start_y", "mid_x", "mid_y", "end_x", "end_y",
            "success", "round_trip_time_sec"
        ])
        self.get_logger().info(f"Saving results to {self.output_file}")

    # === AMCL Pose Callback ===
    def amcl_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_current_start_pose(self):
        """Wait until AMCL pose is available, return as PoseStamped"""
        while self.current_pose is None:
            self.get_logger().warn("Waiting for AMCL pose...")
            rclpy.spin_once(self, timeout_sec=0.5)
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose = self.current_pose
        return pose

    def make_pose(self, x, y, yaw=0.0):
        """Helper to create a PoseStamped"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        return pose

    def switch_planner(self, planner):
        """Switch local planner dynamically"""
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

        self.get_logger().info(f"Switching to {planner}")
        client = self.create_client(SetParameters, '/controller_server/set_parameters')

        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for /controller_server/set_parameters service...")

        # Build request
        param = Parameter()
        param.name = "controller_server.local_planner_plugin"
        param.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=planner)

        req = SetParameters.Request()
        req.parameters = [param]

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f"Planner successfully switched to {planner}")
        else:
            self.get_logger().error("Failed to switch planner!")

    def navigate_and_wait(self, goal_pose):
        """Send goal and block until navigation finishes"""
        self.navigator.goToPose(goal_pose)
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.5)
        return self.navigator.getResult()

    def run_round_trip(self, planner, trial, start, mid, end):
        """Run one full round trip and log results"""
        # Set initial pose
        self.navigator.setInitialPose(start)
        self.get_logger().info("Initial pose set, waiting for Nav2...")
        self.navigator.waitUntilNav2Active()

        # Switch planner
        self.switch_planner(planner)

        # === Start timing ===
        start_time = time.time()
        success = 1

        # Leg 1: Start -> Table
        result1 = self.navigate_and_wait(mid)
        if result1 != TaskResult.SUCCEEDED:
            success = 0

        # Leg 2: Table -> Start
        result2 = self.navigate_and_wait(end)
        if result2 != TaskResult.SUCCEEDED:
            success = 0

        elapsed = time.time() - start_time

        # Log round trip
        self.csv_writer.writerow([
            planner, trial,
            start.pose.position.x, start.pose.position.y,
            mid.pose.position.x, mid.pose.position.y,
            end.pose.position.x, end.pose.position.y,
            success, round(elapsed, 2)
        ])
        self.get_logger().info(f"Round trip {trial} with {planner}: "
                               f"{'SUCCESS' if success else 'FAIL'} in {elapsed:.2f}s")

    def benchmark(self):
        # planners = ["dwb_controller", "teb_local_planner", "RegulatedPurePursuitController"]
        planners = ["RegulatedPurePursuitController"] # it will run only dwb_controller
        table_pose = self.make_pose(*TABLE)

        for planner in planners:
            for t in range(1, self.trials + 1):
                start_pose = self.get_current_start_pose()
                self.run_round_trip(planner, t, start_pose, table_pose, start_pose)

        self.csv_file.close()
        self.get_logger().info("Benchmarking complete ✅")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of round-trip trials per planner")
    parser.add_argument("--output", type=str, default="results/planner_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    rclpy.init()
    node = PlannerBenchmark(args.trials, args.output)
    node.benchmark()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# # #!/usr/bin/env python3
# # """
# # Planner Benchmarking Script
# # - Runs DWB, TEB, and RPP planners in Warehouse A map
# # - Sends robot between charging station and table
# # - Logs time-to-goal, success/failure, and trial info into timestamped CSV

# # Usage:
# #     python3 benchmark_planners.py --trials 5 --output results/planner_results.csv
# # """

# # import os
# # import csv
# # import time
# # import argparse
# # from datetime import datetime

# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import PoseStamped
# # from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


# # # === Waypoints (Warehouse A) ===
# # CHARGING_STATION = (0.0, 0.0, 0.0)     # x, y, yaw
# # TABLE = (-12.7, 6.5, 0.0)


# # class PlannerBenchmark(Node):
# #     def __init__(self, trials, output_path):
# #         super().__init__('planner_benchmark')
# #         self.navigator = BasicNavigator()

# #         # Ensure results dir exists
# #         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #         result_dir = os.path.join(os.path.dirname(output_path),
# #                                   f"planner_results_{timestamp}")
# #         os.makedirs(result_dir, exist_ok=True)
# #         self.output_file = os.path.join(result_dir, os.path.basename(output_path))

# #         self.trials = trials

# #         # Open CSV file
# #         self.csv_file = open(self.output_file, "w", newline="")
# #         self.csv_writer = csv.writer(self.csv_file)
# #         self.csv_writer.writerow([
# #             "planner", "trial",
# #             "start_x", "start_y", "goal_x", "goal_y",
# #             "success", "time_to_goal_sec"
# #         ])
# #         self.get_logger().info(f"Saving results to {self.output_file}")

# #     def make_pose(self, x, y, yaw=0.0):
# #         """Helper to create a PoseStamped at given coordinates"""
# #         pose = PoseStamped()
# #         pose.header.frame_id = 'map'
# #         pose.header.stamp = self.navigator.get_clock().now().to_msg()
# #         pose.pose.position.x = x
# #         pose.pose.position.y = y
# #         # Ignore orientation (yaw=0) for simplicity
# #         pose.pose.orientation.w = 1.0
# #         return pose

# #     def run_trial(self, planner, trial, start, goal):
# #         """Run one trial with given planner and log results"""
# #         # Set initial pose
# #         self.navigator.setInitialPose(start)
# #         self.navigator.waitUntilNav2Active()
# #         # Switch local planner parameter on controller_server
# #         from rcl_interfaces.srv import SetParameters
# #         from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

# #         self.get_logger().info(f"Switching to {planner}")
# #         client = self.create_client(SetParameters, '/controller_server/set_parameters')

# #         while not client.wait_for_service(timeout_sec=1.0):
# #             self.get_logger().warn("Waiting for /controller_server/set_parameters service...")

# #         # Build request
# #         param = Parameter()
# #         param.name = "controller_server.local_planner_plugin"
# #         param.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=planner)

# #         req = SetParameters.Request()
# #         req.parameters = [param]

# #         future = client.call_async(req)
# #         rclpy.spin_until_future_complete(self, future)

# #         if future.result() is not None:
# #             self.get_logger().info(f"Planner successfully switched to {planner}")
# #         else:
# #             self.get_logger().error("Failed to switch planner!")


# #         # Send goal
# #         start_time = time.time()
# #         self.navigator.goToPose(goal)

# #         # Wait for task completion (polling loop)
# #         while not self.navigator.isTaskComplete():
# #             rclpy.spin_once(self, timeout_sec=0.5)

# #         elapsed = time.time() - start_time

# #         # Check result
# #         result = self.navigator.getResult()
# #         success = 1 if result == TaskResult.SUCCEEDED else 0

# #         self.csv_writer.writerow([
# #             planner, trial,
# #             start.pose.position.x, start.pose.position.y,
# #             goal.pose.position.x, goal.pose.position.y,
# #             success, round(elapsed, 2)
# #         ])
# #         self.get_logger().info(f"Trial {trial} with {planner}: "
# #                                f"{'SUCCESS' if success else 'FAIL'} in {elapsed:.2f}s")

# #     def benchmark(self):
# #         # Define planners
# #         planners = ["dwb_controller", "teb_local_planner", "RegulatedPurePursuitController"]

# #         # Define poses
# #         start_pose = self.make_pose(*CHARGING_STATION)
# #         table_pose = self.make_pose(*TABLE)

# #         # Run trials
# #         for planner in planners:
# #             for t in range(1, self.trials + 1):
# #                 # Charging → Table
# #                 self.run_trial(planner, t, start_pose, table_pose)
# #                 # Table → Charging
# #                 self.run_trial(planner, t, table_pose, start_pose)

# #         self.csv_file.close()
# #         self.get_logger().info("Benchmarking complete ✅")


# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--trials", type=int, default=3,
# #                         help="Number of trials per planner")
# #     parser.add_argument("--output", type=str, default="results/planner_results.csv",
# #                         help="Output CSV file path")
# #     args = parser.parse_args()

# #     rclpy.init()
# #     node = PlannerBenchmark(args.trials, args.output)
# #     node.benchmark()
# #     rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()


# #!/usr/bin/env python3


# """
# Planner Benchmarking Script
# - Runs DWB, TEB, and RPP planners in Warehouse A map
# - Executes full round trips (Start -> Table -> Start)
# - Logs round-trip time and success/failure into timestamped CSV

# Usage:
#     python3 benchmark_planners.py --trials 5 --output results/planner_results.csv
# """

# import os
# import csv
# import time
# import argparse
# from datetime import datetime

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


# # === Waypoints (Warehouse A) ===
# START_POINT = (-0.3, 0.0, 0.0)   # x, y, yaw (shifted from charging station)
# TABLE = (-12.7, 6.5, 0.0)


# class PlannerBenchmark(Node):
#     def __init__(self, trials, output_path):
#         super().__init__('planner_benchmark')
#         self.navigator = BasicNavigator()

#         # Ensure results dir exists
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         result_dir = os.path.join(os.path.dirname(output_path),
#                                   f"planner_results_{timestamp}")
#         os.makedirs(result_dir, exist_ok=True)
#         self.output_file = os.path.join(result_dir, os.path.basename(output_path))

#         self.trials = trials

#         # Open CSV file
#         self.csv_file = open(self.output_file, "w", newline="")
#         self.csv_writer = csv.writer(self.csv_file)
#         self.csv_writer.writerow([
#             "planner", "trial",
#             "start_x", "start_y", "mid_x", "mid_y", "end_x", "end_y",
#             "success", "round_trip_time_sec"
#         ])
#         self.get_logger().info(f"Saving results to {self.output_file}")

#     def make_pose(self, x, y, yaw=0.0):
#         """Helper to create a PoseStamped at given coordinates"""
#         pose = PoseStamped()
#         pose.header.frame_id = 'map'
#         pose.header.stamp = self.navigator.get_clock().now().to_msg()
#         pose.pose.position.x = x
#         pose.pose.position.y = y
#         pose.pose.orientation.w = 1.0  # facing forward
#         return pose

#     def switch_planner(self, planner):
#         """Switch local planner dynamically"""
#         from rcl_interfaces.srv import SetParameters
#         from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

#         self.get_logger().info(f"Switching to {planner}")
#         client = self.create_client(SetParameters, '/controller_server/set_parameters')

#         while not client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().warn("Waiting for /controller_server/set_parameters service...")

#         # Build request
#         param = Parameter()
#         param.name = "controller_server.local_planner_plugin"
#         param.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=planner)

#         req = SetParameters.Request()
#         req.parameters = [param]

#         future = client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)

#         if future.result() is not None:
#             self.get_logger().info(f"Planner successfully switched to {planner}")
#         else:
#             self.get_logger().error("Failed to switch planner!")

#     def navigate_and_wait(self, goal_pose):
#         """Send goal and block until navigation finishes"""
#         self.navigator.goToPose(goal_pose)
#         while not self.navigator.isTaskComplete():
#             rclpy.spin_once(self, timeout_sec=0.5)
#         return self.navigator.getResult()

#     def run_round_trip(self, planner, trial, start, mid, end):
#         """Run one full round trip and log results"""
#         # Set initial pose
#         self.navigator.setInitialPose(start)
#         self.get_logger().info("Initial pose set, waiting for Nav2...")
#         self.navigator.waitUntilNav2Active()

#         # Switch planner
#         self.switch_planner(planner)

#         # === Start timing ===
#         start_time = time.time()
#         success = 1

#         # Leg 1: Start -> Table
#         result1 = self.navigate_and_wait(mid)
#         if result1 != TaskResult.SUCCEEDED:
#             success = 0

#         # Leg 2: Table -> Start
#         result2 = self.navigate_and_wait(end)
#         if result2 != TaskResult.SUCCEEDED:
#             success = 0

#         elapsed = time.time() - start_time

#         # Log round trip
#         self.csv_writer.writerow([
#             planner, trial,
#             start.pose.position.x, start.pose.position.y,
#             mid.pose.position.x, mid.pose.position.y,
#             end.pose.position.x, end.pose.position.y,
#             success, round(elapsed, 2)
#         ])
#         self.get_logger().info(f"Round trip {trial} with {planner}: "
#                                f"{'SUCCESS' if success else 'FAIL'} in {elapsed:.2f}s")

#     def benchmark(self):
#         # Define planners
#         planners = ["dwb_controller", "teb_local_planner", "RegulatedPurePursuitController"]

#         # Define poses
#         start_pose = self.make_pose(*START_POINT)
#         table_pose = self.make_pose(*TABLE)

#         # Run trials
#         for planner in planners:
#             for t in range(1, self.trials + 1):
#                 self.run_round_trip(planner, t, start_pose, table_pose, start_pose)

#         self.csv_file.close()
#         self.get_logger().info("Benchmarking complete ✅")


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--trials", type=int, default=3,
#                         help="Number of round-trip trials per planner")
#     parser.add_argument("--output", type=str, default="results/planner_results.csv",
#                         help="Output CSV file path")
#     args = parser.parse_args()

#     rclpy.init()
#     node = PlannerBenchmark(args.trials, args.output)
#     node.benchmark()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
