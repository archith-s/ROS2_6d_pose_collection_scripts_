# ros2_sync_client.py  –  API-compatible ROS-2 version
from __future__ import annotations

import time
from dataclasses import dataclass, field
from abc import ABC
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node
import message_filters                                   # ROS-2 port
from ambf6dpose.DataCollection.Rostopics import (
    RosTopics,
    get_topics_processing_cb,
    topic_to_attr_dict,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RawSimulationData (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RawSimulationData:
    camera_l_pose: np.ndarray
    camera_frame_pose: np.ndarray
    needle_pose: np.ndarray
    camera_l_img: np.ndarray
    camera_l_seg_img: np.ndarray
    camera_0_pose: np.ndarray
    camera_0_img: np.ndarray
    camera_1_pose: np.ndarray 
    camera_1_img: np.ndarray
    camera_l_depth: np.ndarray
    psm1_toolpitchlink_pose: np.ndarray
    psm2_toolpitchlink_pose: np.ndarray
    psm1_toolyawlink_pose: np.ndarray
    psm2_toolyawlink_pose: np.ndarray

    def __post_init__(self):
        if self.has_none_members():
            raise ValueError("SimulationData cannot have None members")

    def has_none_members(self) -> bool:
        return any(v is None for v in vars(self).values())

    @classmethod
    def from_dict(cls: "RawSimulationData",
                  data: Dict[RosTopics, np.ndarray]) -> "RawSimulationData":
        mapped = {topic_to_attr_dict[k]: v for k, v in data.items()}
        return cls(**mapped)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AbstractSimulationClient  (keeps the same public API)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AbstractSimulationClient(ABC):
    """
    Abstract ROS-2 client for collecting synchronous AMBF data.
    """
    raw_data: RawSimulationData = field(default=None, init=False)
    client_name = "ambf_collection_client"
    # internal helper: every derived instance shares one rclpy init
    _rclpy_inited: bool = field(default=False, init=False, repr=False)
    _node: Node | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not AbstractSimulationClient._rclpy_inited:
            rclpy.init()
            AbstractSimulationClient._rclpy_inited = True
        # create a private node so we don’t impose one on user code
        self._node = rclpy.create_node(self.client_name)
    # ------------- public helpers (unchanged signatures) -------------------
    def get_data(self) -> RawSimulationData:
        if self.raw_data is None:
            raise ValueError("No data has been received")
        data, self.raw_data = self.raw_data, None
        return data

    def has_data(self) -> bool:
        return self.raw_data is not None

    def wait_for_data(self, timeout: float = 100.0) -> None:
        start = time.time()
        while not self.has_data() and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError(f"No data received for {timeout}s")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SyncRosInterface  (same external behaviour)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SyncRosInterface(AbstractSimulationClient):
    def __post_init__(self):
        super().__post_init__()
        self.subscribers = []
        self.callback_dict = get_topics_processing_cb()

        # ROS-2 message_filters wants the node first, then msg-type, then topic
        for topic in RosTopics:
            print(f"[DEBUG] Setting up subscriber for: {topic.name} - {topic.value[0]}")
            self.subscribers.append(
                message_filters.Subscriber(
                    self._node,               # Node
                    topic.value[1],           # Msg type
                    topic.value[0],           # Topic name
                    qos_profile=rclpy.qos.qos_profile_sensor_data,
                )
            )
        # All topics publish at a steady 50Hz (20ms period, std dev ~0.00005s
        # -- measured via `ros2 topic hz` on /cameras/camera_0/State and
        # /ambf/env/psm1/toolpitchlink/State, 2026-07-08). The old slop=0.1
        # (100ms = 5 full simulation ticks) let the synchronizer pair up
        # messages from up to 5 different instants as if they were
        # simultaneous -- when the arm was moving quickly, that produced a
        # visible mismatch between the saved image and the saved pose (the
        # GT silhouette landing near, but not exactly on, the tool).
        # slop=0.01 (10ms, half the tick period) makes it impossible for the
        # synchronizer to span two different simulation ticks, while still
        # leaving ~100x margin over the measured jitter for normal
        # publishing/serialization delays (e.g. image messages take longer
        # to serialize than plain pose messages).
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=40, slop=0.01
        )
        self.time_sync.registerCallback(self.cb)
        time.sleep(0.25)      # give subscriptions a moment to connect

    # ------------------- internal callback (unchanged name) -----------------
    def cb(self, *inputs):
        print('Inside cb registering callback')
        raw_dict = {
            topic: self.callback_dict[topic](msg)
            for msg, topic in zip(inputs, RosTopics, strict=True)
        }
        self.raw_data = RawSimulationData.from_dict(raw_dict)
