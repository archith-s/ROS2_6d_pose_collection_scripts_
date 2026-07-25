# ros2_sync_client.py  –  API-compatible ROS-2 version
from __future__ import annotations

import time
from dataclasses import dataclass, field
from abc import ABC
from typing import Dict

import numpy as np
import rclpy
from rclpy.node import Node
from ambf_msgs.msg import CameraState, RigidBodyState        # type: ignore
from ambf6dpose.DataCollection.Rostopics import (
    RosTopics,
    get_topics_processing_cb,
    topic_to_attr_dict,
)

# Only these two ambf_msgs types carry a `sim_step` field (confirmed via raw
# `ros2 topic echo`, 2026-07-09). sensor_msgs/Image and sensor_msgs/PointCloud2
# (the ImageData/DepthData topics) are standard ROS messages with no such
# field -- they only have the usual header.stamp.
_SIM_STEP_MSG_TYPES = (CameraState, RigidBodyState)


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
# Synchronizes on AMBF's own `sim_step` field (present on every CameraState
# and RigidBodyState message -- confirmed via raw `ros2 topic echo`,
# 2026-07-09) instead of message_filters.ApproximateTimeSynchronizer's
# wall-clock `header.stamp` + slop tolerance.
#
# Why: header.stamp is set at publish() time, so two messages describing the
# *same* simulation tick can still end up with different stamps purely from
# per-topic publish/serialization latency (e.g. an image takes longer to
# encode than a small pose message) -- slop exists to paper over exactly
# that gap, but it can't distinguish "same tick, published a few ms apart"
# from "genuinely adjacent ticks", and while the arm is moving, either one
# produces a visible GT/image mismatch. Confirmed empirically 2026-07-09:
# holding the arm stationary gave a perfect overlay even at slop=0.01, but
# t6/t7 (arm moving) still showed a small, consistent silhouette offset even
# at slop=0.001 -- the remaining source was the arm actually moving during
# whatever real (sub-slop) publish-time gap remained, not synchronization
# looseness.
#
# sim_step is assigned once per simulation tick, before any publishing
# happens, so two messages from the same tick carry the identical value
# regardless of how differently long they each took to reach ROS. Matching
# on sim_step equality makes "these describe the same instant" an exact fact
# instead of a wall-clock-proximity guess.
#
# Not every topic can be matched this way: sensor_msgs/Image and
# sensor_msgs/PointCloud2 (every ImageData/DepthData topic) have no sim_step
# field, only the standard header.stamp. So this uses a hybrid: the
# CameraState/RigidBodyState ("State") topics are matched by exact sim_step
# equality; once that set is complete, its (shared) stamp is used as the
# reference instant, and each image/depth topic contributes whichever of its
# recently buffered messages has the closest header.stamp to that reference.
# This still eliminates the original failure mode (a State topic silently
# describing a different simulation tick than another State topic), and
# narrows the remaining approximate part to "which image is closest to an
# exactly-known instant" rather than "are these poses even from the same
# tick" -- a strictly smaller source of error than before.
def _stamp_to_seconds(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class _SimStepSynchronizer:
    def __init__(self, node: Node, topics, callback,
                 max_buffered_steps: int = 20, max_buffered_timed: int = 40,
                 max_timed_gap_sec: float = 0.015):
        self._callback = callback
        self._topics = list(topics)
        self._step_topics = [t for t in self._topics if t.value[1] in _SIM_STEP_MSG_TYPES]
        self._timed_topics = [t for t in self._topics if t.value[1] not in _SIM_STEP_MSG_TYPES]

        self._max_buffered_steps = max_buffered_steps
        self._max_buffered_timed = max_buffered_timed
        # Bound on how far (in wall-clock seconds) an image/depth message may
        # sit from the sim_step-resolved reference instant and still count as
        # a match. Without this, "closest available" can silently accept a
        # stale message left over from a different tick if nothing closer has
        # arrived yet -- default is under one 20ms tick period, so a
        # same-tick image (however late its publish latency) still qualifies
        # but a genuinely different tick's leftover image does not.
        self._max_timed_gap_sec = max_timed_gap_sec
        self._step_buffers: Dict[RosTopics, Dict[int, object]] = {t: {} for t in self._step_topics}
        # each timed topic keeps a small list of (stamp_seconds, msg), newest last
        self._timed_buffers: Dict[RosTopics, list] = {t: [] for t in self._timed_topics}

        self._subscriptions = []
        for topic in self._topics:
            print(f"[DEBUG] Setting up subscriber for: {topic.name} - {topic.value[0]}")
            sub = node.create_subscription(
                topic.value[1],   # msg type
                topic.value[0],   # topic name
                self._make_topic_callback(topic),
                rclpy.qos.qos_profile_sensor_data,
            )
            self._subscriptions.append(sub)

    def _make_topic_callback(self, topic: RosTopics):
        def _on_msg(msg):
            self._handle(topic, msg)
        return _on_msg

    def _handle(self, topic: RosTopics, msg) -> None:
        if topic in self._step_topics:
            step = msg.sim_step
            buf = self._step_buffers[topic]
            buf[step] = msg
            if len(buf) > self._max_buffered_steps:
                del buf[min(buf.keys())]
            self._try_complete(step)
        else:
            buf = self._timed_buffers[topic]
            buf.append((_stamp_to_seconds(msg), msg))
            if len(buf) > self._max_buffered_timed:
                buf.pop(0)
            # This new image might be the piece that was missing for a
            # sim_step whose State topics already all arrived -- a match
            # that was blocked earlier (no close-enough image yet) is never
            # retried by the step-topic side alone, since sim_step is
            # monotonic and that exact value won't be seen again. Retry
            # every currently fully-buffered-but-unmatched step so a later,
            # closer image can still complete it.
            if self._step_topics:
                candidate_steps = set.intersection(
                    *(set(self._step_buffers[t].keys()) for t in self._step_topics)
                )
                for step in sorted(candidate_steps):
                    if self._try_complete(step):
                        break  # buffers mutated; remaining steps handled on next call

    def _try_complete(self, step: int) -> bool:
        """Attempt to fire a synchronized sample for this sim_step. Returns
        True if it fired (and cleared the relevant buffers), False if still
        incomplete (missing a topic, or no close-enough timed message yet)."""
        if not all(step in self._step_buffers[t] for t in self._step_topics):
            return False

        step_matched = {t: self._step_buffers[t][step] for t in self._step_topics}
        reference_time = _stamp_to_seconds(next(iter(step_matched.values())))

        timed_matched: Dict[RosTopics, object] = {}
        for t in self._timed_topics:
            candidates = self._timed_buffers[t]
            if not candidates:
                return False  # haven't received a single message on this topic yet
            best_gap, best_msg = min(
                ((abs(stamp - reference_time), m) for stamp, m in candidates),
                key=lambda pair: pair[0],
            )
            if best_gap > self._max_timed_gap_sec:
                # Closest available is still too far (e.g. a stale leftover
                # from a different tick) -- wait for a closer one instead of
                # matching against something we can't trust yet.
                return False
            timed_matched[t] = best_msg

        # Drop this and older buffered sim_steps everywhere -- consumed or
        # were never going to complete a set.
        for t in self._step_topics:
            for stale_step in [s for s in self._step_buffers[t] if s <= step]:
                del self._step_buffers[t][stale_step]

        matched = {**step_matched, **timed_matched}
        self._callback(*[matched[t] for t in self._topics])
        return True


@dataclass
class SyncRosInterface(AbstractSimulationClient):
    def __post_init__(self):
        super().__post_init__()
        self.callback_dict = get_topics_processing_cb()
        self.sim_step_sync = _SimStepSynchronizer(self._node, RosTopics, self.cb)
        time.sleep(0.25)      # give subscriptions a moment to connect

    # ------------------- internal callback (unchanged name) -----------------
    def cb(self, *inputs):
        print('Inside cb registering callback')
        raw_dict = {
            topic: self.callback_dict[topic](msg)
            for msg, topic in zip(inputs, RosTopics, strict=True)
        }
        self.raw_data = RawSimulationData.from_dict(raw_dict)
