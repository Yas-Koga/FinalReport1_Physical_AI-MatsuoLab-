
#!/usr/bin/env python3

import argparse
import json

import numpy as np
import torch
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from lerobot.common.policies.act.modeling_act import ACTPolicy


# ==================================================
# CRANE+ V2 Joint Configuration
# ==================================================

# Must match crane_features.py:
# joint1, joint2, joint4, joint3, gripper

ARM_JOINT_NAMES = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint4",
    "crane_plus_joint3",
]

GRIPPER_JOINT_NAME = "crane_plus_joint_hand"

ALL_JOINT_NAMES = (
    ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
)

CONTROL_HZ = 20.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ

TRAJECTORY_DURATION = 0.1


# ==================================================
# Inference Node
# ==================================================

class InferenceNode(Node):

    def __init__(
        self,
        model_path,
        enable_publish=False,
        joint_limits=None,
    ):
        super().__init__("crane_plus_act_infer")

        self.enable_publish = enable_publish
        self.joint_limits = joint_limits

        self.device = torch.device("cpu")

        self.latest_joint_state = None
        self.last_joint_state_time = None

        # ------------------------------------------
        # Load ACT model
        # ------------------------------------------

        self.get_logger().info(
            f"Loading ACT model: {model_path}"
        )

        self.policy = ACTPolicy.from_pretrained(
            model_path
        )

        self.policy.to(self.device)
        self.policy.eval()

        self.get_logger().info(
            "ACT model loaded successfully."
        )

        # ------------------------------------------
        # ROS2 Publishers
        # ------------------------------------------

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            "/crane_plus_arm_controller/joint_trajectory",
            10,
        )

        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            "/crane_plus_gripper_controller/joint_trajectory",
            10,
        )

        # ------------------------------------------
        # ROS2 Subscriber
        # ------------------------------------------

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        # ------------------------------------------
        # Timer: 20Hz
        # ------------------------------------------

        self.timer = self.create_timer(
            CONTROL_PERIOD,
            self.control_loop,
        )

        if self.enable_publish:
            self.get_logger().warning(
                "PUBLISH ENABLED. Robot commands may be sent."
            )
        else:
            self.get_logger().info(
                "DRY RUN MODE. No robot commands will be sent."
            )

    # ==================================================
    # Receive JointState
    # ==================================================

    def joint_state_callback(self, msg):

        if len(msg.name) != len(msg.position):
            self.get_logger().warning(
                "Invalid JointState message."
            )
            return

        name_to_position = dict(
            zip(msg.name, msg.position)
        )

        missing_joints = [
            name
            for name in ALL_JOINT_NAMES
            if name not in name_to_position
        ]

        if missing_joints:
            self.get_logger().warning(
                f"Missing joints: {missing_joints}"
            )
            return

        positions = np.array(
            [
                name_to_position[name]
                for name in ALL_JOINT_NAMES
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(positions)):
            self.get_logger().warning(
                "JointState contains NaN or Inf."
            )
            return

        self.latest_joint_state = positions

        self.last_joint_state_time = self.get_clock().now()

    # ==================================================
    # Prepare ACT observation
    # ==================================================

    def prepare_observation(self):

        state = torch.tensor(
            self.latest_joint_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        # Must match crane_features.py

        observation = {
            "observation.state": state,
            "observation.environment_state": state,
        }

        return observation

    # ==================================================
    # ACT Inference
    # ==================================================

    def infer_action(self):

        observation = self.prepare_observation()

        with torch.inference_mode():

            action = self.policy.select_action(
                observation
            )

        action = (
            action.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if action.shape != (5,):
            raise ValueError(
                f"Expected 5 actions, got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                "ACT returned NaN or Inf."
            )

        return action.astype(np.float32)

    # ==================================================
    # Validate Action
    # ==================================================

    def validate_action(self, action):

        if self.joint_limits is None:
            return False

        for i, (lower, upper) in enumerate(
            self.joint_limits
        ):

            value = float(action[i])

            if not lower <= value <= upper:

                self.get_logger().error(
                    f"Joint limit violation: "
                    f"{ALL_JOINT_NAMES[i]}, "
                    f"value={value:.5f}, "
                    f"allowed=[{lower}, {upper}]"
                )

                return False

        return True

    # ==================================================
    # Check JointState freshness
    # ==================================================

    def is_joint_state_fresh(self):

        if self.last_joint_state_time is None:
            return False

        elapsed = (
            self.get_clock().now()
            - self.last_joint_state_time
        ).nanoseconds / 1e9

        return elapsed <= 0.5

    # ==================================================
    # Create JointTrajectory
    # ==================================================

    def create_trajectory(
        self,
        joint_names,
        positions,
    ):

        trajectory = JointTrajectory()

        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()

        point.positions = [
            float(value)
            for value in positions
        ]

        point.time_from_start.sec = 0

        point.time_from_start.nanosec = int(
            TRAJECTORY_DURATION * 1e9
        )

        trajectory.points.append(point)

        return trajectory

    # ==================================================
    # Main Control Loop
    # ==================================================

    def control_loop(self):

        # Wait for joint states

        if self.latest_joint_state is None:
            self.get_logger().info(
                "Waiting for /joint_states...",
                throttle_duration_sec=5.0,
            )
            return

        # Check communication freshness

        if not self.is_joint_state_fresh():
            self.get_logger().error(
                "JointState timeout. Skipping inference."
            )
            return

        try:

            # ACT inference
            action = self.infer_action()

            arm_action = action[:4]
            gripper_action = action[4]

            self.get_logger().info(
                f"Action: {np.round(action, 5)}"
            )

            # Dry Run
            if not self.enable_publish:
                return

            # Validate joint limits
            if not self.validate_action(action):
                self.get_logger().error(
                    "Action rejected."
                )
                return

            # Create trajectories
            arm_trajectory = self.create_trajectory(
                ARM_JOINT_NAMES,
                arm_action,
            )

            gripper_trajectory = self.create_trajectory(
                [GRIPPER_JOINT_NAME],
                [gripper_action],
            )

            # Publish
            self.arm_pub.publish(
                arm_trajectory
            )

            self.gripper_pub.publish(
                gripper_trajectory
            )

        except Exception as e:

            self.get_logger().error(
                f"Inference error: {e}"
            )


# ==================================================
# Main
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="CRANE+ V2 ACT Inference"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Trained ACT model directory",
    )

    parser.add_argument(
        "--enable-publish",
        action="store_true",
        help="Enable ROS2 command publishing",
    )

    parser.add_argument(
        "--joint-limits",
        type=str,
        default=None,
        help="JSON file containing 5 joint limits",
    )

    parser.add_argument(
        "--confirm-dataset-validated",
        action="store_true",
        help="Confirm that dataset ordering and action values have been validated",
    )

    args = parser.parse_args()

    joint_limits = None

    if args.joint_limits:

        with open(
            args.joint_limits,
            "r",
            encoding="utf-8",
        ) as f:

            joint_limits = json.load(f)

        if (
            not isinstance(joint_limits, list)
            or len(joint_limits) != 5
        ):
            parser.error(
                "joint-limits must contain 5 pairs."
            )

        for pair in joint_limits:

            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(
                    isinstance(x, (int, float))
                    and np.isfinite(x)
                    for x in pair
                )
                or pair[0] >= pair[1]
            ):
                parser.error(
                    "Invalid joint limit pair."
                )

    if args.enable_publish:

        if joint_limits is None:
            parser.error(
                "--enable-publish requires --joint-limits"
            )

        if not args.confirm_dataset_validated:
            parser.error(
                "Publishing requires "
                "--confirm-dataset-validated"
            )

    rclpy.init()

    node = None

    try:

        node = InferenceNode(
            model_path=args.model_path,
            enable_publish=args.enable_publish,
            joint_limits=joint_limits,
        )

        rclpy.spin(node)

    except KeyboardInterrupt:

        print("\nInference stopped.")

    finally:

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
