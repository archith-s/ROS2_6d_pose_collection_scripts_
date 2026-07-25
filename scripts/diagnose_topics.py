# diagnose_topics.py
#
# One-shot diagnostic: subscribes to all 20 RosTopics entries directly
# (bypassing SyncRosInterface/ApproximateTimeSynchronizer entirely) and
# reports, per topic, how many messages arrived and the age of the most
# recent one relative to the freshest topic seen.
#
# This exists to answer one question when collect_data.py hangs waiting for
# a synchronized sample: is a specific topic not publishing at all (count=0,
# flagged SILENT), or publishing too infrequently/late to ever land within
# `slop` seconds of the others (count>0 but consistently large `age`)? Either
# one explains why ApproximateTimeSynchronizer never fires.
#
# Usage:
#   python scripts/diagnose_topics.py
#
# Run this from wherever collect_data.py normally runs (same PYTHONPATH /
# ROS_DOMAIN_ID), with the simulator already up.

import time
from collections import defaultdict

import rclpy

from ambf6dpose.DataCollection.Rostopics import RosTopics

RUN_SECONDS = 15.0


def stamp_to_seconds(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main():
    rclpy.init()
    node = rclpy.create_node("topic_diagnostic")

    counts = defaultdict(int)
    last_stamp = {}
    subs = []

    def make_cb(topic):
        def _cb(msg):
            counts[topic] += 1
            last_stamp[topic] = stamp_to_seconds(msg)
        return _cb

    for topic in RosTopics:
        sub = node.create_subscription(
            topic.value[1],
            topic.value[0],
            make_cb(topic),
            rclpy.qos.qos_profile_sensor_data,
        )
        subs.append(sub)

    print(f"Listening on {len(subs)} topics for {RUN_SECONDS:.0f}s...\n")
    start = time.time()
    while time.time() - start < RUN_SECONDS and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f"{'topic':45s} {'name':45s} {'count':>7s}  last_stamp")
    print("-" * 110)
    stamps = list(last_stamp.values())
    newest = max(stamps) if stamps else None
    for topic in RosTopics:
        c = counts[topic]
        flag = "  <-- SILENT" if c == 0 else ""
        stamp_info = ""
        if topic in last_stamp and newest is not None:
            stamp_info = f"{last_stamp[topic]:.3f} (age {newest - last_stamp[topic]:.3f}s)"
        print(f"{topic.name:45s} {topic.value[0]:45s} {c:7d}  {stamp_info}{flag}")

    rclpy.shutdown()


if __name__ == "__main__":
    main()
