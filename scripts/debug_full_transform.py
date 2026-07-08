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
#   python scripts/debug_full_transform.py

import numpy as np

from ambf6dpose import SyncRosInterface
from ambf6dpose.DataCollection.SimulatorDataProcessor import SimulatorDataProcessor

np.set_printoptions(precision=6, suppress=True)


def main():
    client = SyncRosInterface()
    processor = SimulatorDataProcessor(client)

    print("Waiting for one synchronized snapshot...")
    client.wait_for_data(timeout=30.0)
    raw = client.get_data()
    print("Got snapshot.\n")

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
        print(f"transform_from_world_to_cam() result (obj-in-cam, mm translation):")
        print(result_mm)
        print()


if __name__ == "__main__":
    main()
