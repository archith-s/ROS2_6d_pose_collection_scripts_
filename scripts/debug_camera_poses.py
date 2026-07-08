# debug_camera_poses.py
#
# One-shot diagnostic: connects to the running simulator, grabs a single
# synchronized snapshot, and prints the RAW published pose of camera_0 and
# camera_1 (position + rotation, in world coordinates) next to the pose the
# ADF (camera_generator.yaml) *intends* them to have.
#
# This exists to answer one question: is AMBF actually placing camera_0
# where the ADF says it should be? If the "actual" and "expected" numbers
# below don't match, the bug is on the simulator/ADF-loading side for that
# specific camera. If they DO match, the bug must be in how the collection
# code (SimulatorDataProcessor) uses that pose, and we look there instead.
#
# Usage:
#   python scripts/debug_camera_poses.py
#
# Run this with the simulator already up and camera_generator.yaml loaded
# (i.e. launched with --cameras).

import numpy as np

from ambf6dpose import SyncRosInterface

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------------------
# Expected placement, taken directly from camera_generator.yaml / the
# LOOK_AT constant in camera_generator.py — recomputed here independently
# so this script doesn't depend on either file being correct.
# ---------------------------------------------------------------------------
LOOK_AT = np.array([0.0307, 0.1695, 0.6978])

EXPECTED = {
    "camera_0": {
        "location": np.array([-0.040011, 0.1695, 0.768511]),
        "up": np.array([0.707107, 0.0, 0.707107]),
    },
    "camera_1": {
        "location": np.array([0.101411, 0.1695, 0.768511]),
        "up": np.array([-0.707107, 0.0, 0.707107]),
    },
}


def pointing_error_deg(actual_R: np.ndarray, actual_pos: np.ndarray) -> tuple:
    """
    Convention-independent check: does EITHER local axis of actual_R
    (+Z or -Z — whichever AMBF actually uses as "forward") point from
    actual_pos toward LOOK_AT? Returns (best_error_deg, which_axis).

    This deliberately avoids assuming AMBF's exact right/up handedness
    (Z-forward vs -Z-forward, right = up x forward vs forward x up, ...) —
    guessing that wrong would produce a false MISMATCH verdict. Checking
    "does some camera axis point at the target" is convention-free and
    still answers the question that actually matters: is camera_0 oriented
    toward the phantom at all, or off in some unrelated direction?
    """
    required_dir = LOOK_AT - actual_pos
    required_dir = required_dir / np.linalg.norm(required_dir)

    z_axis = actual_R[:, 2]
    best_err = None
    best_which = None
    for which, cand in [("+Z", z_axis), ("-Z", -z_axis)]:
        cos_a = np.clip(np.dot(cand, required_dir), -1.0, 1.0)
        err = np.degrees(np.arccos(cos_a))
        if best_err is None or err < best_err:
            best_err = err
            best_which = which
    return best_err, best_which


def main():
    client = SyncRosInterface()
    print("Waiting for one synchronized snapshot...")
    client.wait_for_data(timeout=30.0)
    raw = client.get_data()
    print("Got snapshot.\n")

    for name, pose_attr in [("camera_0", "camera_0_pose"), ("camera_1", "camera_1_pose")]:
        cam_pose = getattr(raw, pose_attr)
        actual_pos = cam_pose[:3, 3]
        actual_R = cam_pose[:3, :3]

        expected_pos = EXPECTED[name]["location"]
        pos_err = np.linalg.norm(actual_pos - expected_pos)
        point_err_deg, which_axis = pointing_error_deg(actual_R, actual_pos)

        print(f"=== {name} ===")
        print(f"  actual   position (world): {actual_pos}")
        print(f"  expected position (world): {expected_pos}")
        print(f"  position error: {pos_err*1000:.2f} mm")
        print(f"  actual rotation matrix:\n{actual_R}")
        print(f"  pointing error (best of +Z/-Z axis vs true LOOK_AT direction): "
              f"{point_err_deg:.2f} deg  (matched axis: {which_axis})")

        pos_ok = pos_err < 0.005
        point_ok = point_err_deg < 10.0
        if pos_ok and point_ok:
            verdict = "OK — AMBF placed and oriented this camera correctly"
        elif not pos_ok and point_ok:
            verdict = "POSITION WRONG — camera is oriented fine but sitting in the wrong spot"
        elif pos_ok and not point_ok:
            verdict = "ORIENTATION WRONG — camera is in the right spot but not looking at the phantom"
        else:
            verdict = "BOTH WRONG — position and orientation both off"
        print(f"  VERDICT: {verdict}\n")

    # Also dump camera_frame_pose since camera_l's math depends on it, as a
    # sanity check that it's stable/non-degenerate.
    print("=== camera_frame_pose (world, for reference) ===")
    print(raw.camera_frame_pose)


if __name__ == "__main__":
    main()
