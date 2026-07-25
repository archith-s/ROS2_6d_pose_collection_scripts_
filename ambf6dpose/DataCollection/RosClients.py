# ros2_sync_client.py  –  API-compatible ROS-2 version
from __future__ import annotations

import time
from dataclasses import dataclass, field
from abc import ABC
from typing import Dict

import numpy as np
import rclpy
import message_filters
from rclpy.node import Node
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
    camera_frame_pose: np.ndarray
    needle_pose: np.ndarray
    camera_l_pose: np.ndarray
    camera_l_img: np.ndarray
    camera_l_seg_img: np.ndarray
    camera_l_depth: np.ndarray
    camera_0_pose: np.ndarray
    camera_0_img: np.ndarray
    camera_1_pose: np.ndarray 
    camera_1_img: np.ndarray
    camera_2_pose: np.ndarray
    camera_2_img: np.ndarray
    camera_3_pose: np.ndarray
    camera_3_img: np.ndarray
    camera_4_pose: np.ndarray
    camera_4_img: np.ndarray
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
#
# Synchronizes on wall-clock `header.stamp` proximity via
# message_filters.ApproximateTimeSynchronizer, with a `slop` tolerance.
# (Previously replaced with a custom AMBF `sim_step`-matching synchronizer,
# reverted 2026-07-25: `sim_step` turned out to be a per-object publish
# counter -- ambf_server/include/ambf_server/RosComBase.h `increment_sim_step()`
# -- that starts at 0 independently for every afObject, not a shared global
# simulation tick. Objects loaded from a separate ADF file at a different
# point during simulator startup (e.g. camera_0-4, loaded from
# camera_generator.yaml as multibody config #14) end up with a constant
# integer offset from objects loaded earlier, so exact sim_step equality
# across topics from different config files never succeeds.)
@dataclass
class SyncRosInterface(AbstractSimulationClient):
    slop: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        self.callback_dict = get_topics_processing_cb()

        self.subscribers = [
            message_filters.Subscriber(
                self._node,
                topic.value[1],   # msg type
                topic.value[0],   # topic name
                qos_profile=rclpy.qos.qos_profile_sensor_data,
            )
            for topic in RosTopics
        ]
        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=self.slop
        )
        self.time_sync.registerCallback(self.cb)
        time.sleep(0.25)      # give subscriptions a moment to connect

    # ------------------- internal callback (unchanged name) -----------------
    def cb(self, *inputs):
        raw_dict = {
            topic: self.callback_dict[topic](msg)
            for msg, topic in zip(inputs, RosTopics, strict=True)
        }
        self.raw_data = RawSimulationData.from_dict(raw_dict)
