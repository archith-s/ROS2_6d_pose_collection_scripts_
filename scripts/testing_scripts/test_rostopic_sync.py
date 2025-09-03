# test_rostopic_sync_ros2.py
#export PYTHONPATH=$HOME/surgical_robotics_challenge/scripts:$PYTHONPATH

import time
import click

import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ambf6dpose.DataCollection.Rostopics import RosTopics
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

reliable_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

selected_topics = [
    RosTopics.CAMERA_L_STATE,
    RosTopics.CAMERA_FRAME,
    RosTopics.NEEDLE,
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_L_SEG_IMAGE,
    RosTopics.CAMERA_L_DEPTH,
    RosTopics.PSM1_TOOL_PITCH_LINK,
    RosTopics.PSM2_TOOL_PITCH_LINK,
    RosTopics.PSM1_TOOL_YAW_LINK,
    RosTopics.PSM2_TOOL_YAW_LINK,
]


class TestRosSyncClient(Node):
    def __init__(self, slop):
        super().__init__("test_ros_client")

        self.subscribers = []
        for topic in selected_topics:
            topic_name, msg_type = topic.value         
            # ROS 2 Subscriber signature: (node, msg_type, topic, …)
            self.subscribers.append(
                Subscriber(
                    self,
                    msg_type,
                    topic_name,
                    qos_profile=reliable_qos,
                )
            )

        # ApproximateTimeSynchronizer works the same in ROS 2
        self.time_sync = ApproximateTimeSynchronizer(
            self.subscribers, queue_size=6, slop=slop
        )
        self.time_sync.registerCallback(self.common_cb)
        self.last_time = time.time()
        time.sleep(0.25)   # allow connections to settle

    # ───────────────────────── callback (unchanged name) ───────────────────
    def common_cb(self, *inputs):
        print(
            f"Time from last message {(time.time() - self.last_time) * 1000:0.3f} ms",
            end="\r",
        )
        self.last_time = time.time()


# ─────────────────────────── CLI entry-point (unchanged) ───────────────────
@click.command()
@click.option("--slop", default=0.2, type=float)
def test_sync_client(slop: float):
    """Test approximate-time synchronisation under ROS 2."""
    rclpy.init()
    node = TestRosSyncClient(slop)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    test_sync_client()


'''
import time

import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ambf6dpose.DataCollection.Rostopics import RosTopics  # Ensure this uses ROS 2 types


selected_topics = [
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_FRAME,
    #RosTopics.NEEDLE,
    #RosTopics.CAMERA_L_IMAGE,
    #RosTopics.CAMERA_L_SEG_IMAGE,
    #RosTopics.CAMERA_L_DEPTH,
    #RosTopics.PSM1_TOOL_PITCH_LINK,
    #RosTopics.PSM2_TOOL_PITCH_LINK,
    #RosTopics.PSM1_TOOL_YAW_LINK,
    #RosTopics.PSM2_TOOL_YAW_LINK,
]


class TestRosSyncClient(Node):
    def __init__(self, slop):
        super().__init__("test_ros_client")
        self.subscribers = []
        for topic in selected_topics:
            topic_name, msg_type = topic.value
            self.subscribers.append(Subscriber(self, msg_type, topic_name))
        print("Stop1")
        self.time_sync = ApproximateTimeSynchronizer(self.subscribers, queue_size=10, slop=slop)
        print("Stop2")
        self.time_sync.registerCallback(self.common_cb)
        print("Stop3")
        self.last_time = time.time()

    def common_cb(self, *inputs):
        print("Stop12")
        now = time.time()
        dt = (now - self.last_time) * 1000
        self.get_logger().info(f"Time from last message: {dt:.3f} ms")
        self.last_time = now
        print("Stop22")


def main():
    rclpy.init()
    slop = 0.05 
    node = TestRosSyncClient(slop)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    

import time
import rospy, rostopic
import message_filters
import rostopic
from ambf6dpose.DataCollection.Rostopics import RosTopics
import click


selected_topics = [
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_FRAME,
    RosTopics.NEEDLE,
    RosTopics.CAMERA_L_IMAGE,
    RosTopics.CAMERA_L_SEG_IMAGE,
    RosTopics.CAMERA_L_DEPTH,
    RosTopics.PSM1_TOOL_PITCH_LINK,
    RosTopics.PSM2_TOOL_PITCH_LINK,
    RosTopics.PSM1_TOOL_YAW_LINK,
    RosTopics.PSM2_TOOL_YAW_LINK,
]


class TestRosSyncClient:
    def __init__(self, slop):
        self.subscribers = []
        for topic in selected_topics:
            self.subscribers.append(
                message_filters.Subscriber(topic.value[0], topic.value[1])
            )

        # WARNING: TimeSynchronizer did not work. Use ApproximateTimeSynchronizer instead.
        # self.time_sync = message_filters.TimeSynchronizer(self.subscribers, 10)
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=slop
        )

        self.last_time = time.time()
        self.time_sync.registerCallback(self.common_cb)
        time.sleep(0.25)

    def common_cb(self, *inputs):
        print(
            f"Time from last message {(time.time() - self.last_time)*1000:0.3f}ms",
            end="\r",
        )
        self.last_time = time.time()


@click.command()
@click.option("--slop", default=0.05, type=float)
def test_sync_client(slop: float):
   # """Test approximate syncronize message filter"""

    rospy.init_node("test_ros_client")
    client = TestRosSyncClient(slop)
    rospy.spin()


if __name__ == "__main__":
    test_sync_client()
    '''
