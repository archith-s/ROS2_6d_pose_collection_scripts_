# debug_full_transform.py
#
# Calls the REAL, deployed SimulatorDataProcessor.transform_from_world_to_cam
# directly (no reimplementation) on one live synchronized snapshot, and
# prints every intermediate value: the raw world-frame object pose, the raw
# camera pose, and the final composed result for camera_0. This exists to
# rule out any gap between "what the code is supposed to do" and "what it
# actually computes" on a real snapshot, since the axis-flip fix in
# SimulatorDataProcessor.py checked out in isolation but the resulting
# dataset still showed camera_0 misaligned.
#
# Usage:
#   python scripts/debug_full_transform.py [--samples 10] [--interval 1.0]
#
# Move the psm1 arm around (teleop/replay) WHILE this runs so at least one
# snapshot catches it actually in the cameras' working volume -- a snapshot
# where camera_l's own Z comes out negative means the arm simply wasn't in
# view of ANY camera at that instant, and isn't a meaningful test case.

import argparse
import time

import numpy as np

from ambf6dpose import SyncRosInterface
from ambf6dpose.DataCollection.SimulatorDataProcessor import SimulatorDataProcessor

np.set_printoptions(precision=6, suppress=True)


def dump_snapshot(processor, raw, sample_idx):
    print(f"\n########## snapshot {sample_idx} ##########")
    print("=== raw.camera_frame_pose (world) ===")
    print(raw.camera_frame_pose)
    print()

    for cam_name, pose_attr, is_world_frame in [
        ("camera_l", "camera_l_pose", False),
        ("camera_0", "camera_0_pose", True),
        ("camera_1", "camera_1_pose", True),
    ]:
        cam_pose = getattr(raw, pose_attr)
        print(f"=== {cam_name} ===")
        print(f"raw {pose_attr} (world if is_world_frame else relative-to-CameraFrame):")
        print(cam_pose)

        print("raw.psm1_toolpitchlink_pose (world):")
        print(raw.psm1_toolpitchlink_pose)

        # Call the REAL deployed method directly -- not a reimplementation.
        result = processor.transform_from_world_to_cam(
            raw.psm1_toolpitchlink_pose, cam_pose, raw.camera_frame_pose, is_world_frame
        )
        result_mm = processor.convert_pose_to_mm(result.copy())
        z_mm = result_mm[2, 3]
        print(f"transform_from_world_to_cam() result (obj-in-cam, mm translation):")
        print(result_mm)
        print(f"Z={z_mm:.2f}mm  ({'in front -- usable' if z_mm > 0 else 'BEHIND camera -- not a useful sample for this camera'})")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    client = SyncRosInterface()
    processor = SimulatorDataProcessor(client)

    for i in range(args.samples):
        print(f"Waiting for snapshot {i}...")
        client.wait_for_data(timeout=30.0)
        raw = client.get_data()
        dump_snapshot(processor, raw, i)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
