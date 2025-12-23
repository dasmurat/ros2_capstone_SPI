#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
import json
import os
import re
import requests
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import euler2quat
from nav2_simple_commander.robot_navigator import TaskResult


class LLMNavCommander(Node):
    def __init__(self):
        super().__init__('llm_nav_commander')

        self.navigator = BasicNavigator()

        # API config
        self.api_key = 
        self.model = "openai/gpt-4o"

        # Load warehouse locations
        self.location_file = os.path.join(
            get_package_share_directory('llm_nav_interface'),
            'warehouse_locations.json'
        )
        self.locations = self.load_locations()

        # Hardcoded environment hints
        self.hints = {
            "water": "table",
            "bottle of water": "table",
            "panadol": "foldable chair",
            "pain relief": "foldable chair",
            "paracetamol": "foldable chair"
        }

        # Build system prompt with both locations and hints
        self.system_prompt = self.build_system_prompt()

        # Run once
        self.run_pipeline()

    def load_locations(self):
        try:
            with open(self.location_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"Error loading locations file: {e}")
            return {}

    def build_system_prompt(self):
        prompt = "You are a robot navigation assistant in a warehouse environment.\n\n"

        prompt += "Here are the known locations:\n"
        for name, coords in self.locations.items():
            prompt += f"- {name}: ({coords[0]}, {coords[1]})\n"

        prompt += "\nHere are item-to-location hints:\n"
        for item, location in self.hints.items():
            prompt += f"- {item} → {location}\n"

        prompt += '\nRespond only in JSON format like: {"goal": [x, y]}'
        return prompt

    def create_pose_stamped(self, x, y, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = euler2quat(0, 0, yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def call_llm(self, command):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command}
            ],
            "temperature": 0.2,
            # ✅ Ask for strict JSON
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code != 200:
                self.get_logger().error(f"LLM API error {response.status_code}: {response.text}")
                return None

            data = response.json()
            message = data["choices"][0]["message"]["content"]

            # ✅ First try direct JSON parse
            try:
                parsed = json.loads(message)
                if "goal" in parsed:
                    return parsed["goal"]
            except json.JSONDecodeError:
                pass

            # ✅ Fallback: regex extract { ... }
            match = re.search(r"\{.*\}", message, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if "goal" in parsed:
                        return parsed["goal"]
                except Exception as e:
                    self.get_logger().error(f"Regex JSON parse error: {e}")

            # ✅ Extra fallback: if model returns just item name
            clean_text = message.strip().lower()
            if clean_text in self.hints:
                mapped_location = self.hints[clean_text]
                if mapped_location in self.locations:
                    return self.locations[mapped_location]

            self.get_logger().error(f"LLM did not return valid JSON or known item: {message}")
            return None

        except Exception as e:
            self.get_logger().error(f"Request to LLM failed: {e}")
            return None

    def send_goal(self, goal):
        self.navigator.waitUntilNav2Active()

        pose = self.create_pose_stamped(goal[0], goal[1], 0.0)
        self.navigator.goToPose(pose)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            if feedback:
                print(f"Distance remaining: {feedback.distance_remaining:.2f} m")

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print("✅ Goal reached!")
        elif result == TaskResult.CANCELED:
            print("⚠️ Goal was canceled.")
        elif result == TaskResult.FAILED:
            print("❌ Goal failed.")
        else:
            print(f"❓ Unknown result: {result}")
        self.get_logger().info("🟢 Goal sent to the robot.")

    def run_pipeline(self):
        command = input("Enter a command for the robot: ")
        goal = self.call_llm(command)
        if goal:
            print(f"Sending robot to: {goal}")
            self.send_goal(goal)
        else:
            print("No valid goal received from LLM.")


def main(args=None):
    rclpy.init(args=args)
    node = LLMNavCommander()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
