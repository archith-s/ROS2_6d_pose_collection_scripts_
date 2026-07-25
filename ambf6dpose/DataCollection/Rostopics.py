# rostopics_ros2.py  –  identical API to the old ROS 1 helper
# rostopics_ros2.py
from __future__ import annotations

from typing import Any, Callable, Dict
from enum import Enum

# ————————————————————————  ROS 2 message types  ————————————————————————
from sensor_msgs.msg import Image, PointCloud2                     # type: ignore
from ambf_msgs.msg import CameraState, RigidBodyState              # type: ignore

# ————————————————————————  Remaining deps  ———————————————————————————
import numpy as np
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge                                     # type: ignore
from surgical_robotics_challenge.units_conversion import SimToSI


# ============================================================================
# Pose helpers
# Works for both data collection and rendering — no manual switching needed.
# ============================================================================

def _from_msg_to_matrix(msg_pose) -> np.ndarray:
    """Convert a ROS geometry_msgs/Pose to a 4x4 NumPy matrix."""
    p = msg_pose.position
    o = msg_pose.orientation
    m = np.eye(4)
    m[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
    m[:3, 3]  = [p.x, p.y, p.z]
    return m


def _process_pose(msg: RigidBodyState) -> np.ndarray:
    """Convert a RigidBodyState message to a 4x4 pose matrix in SI units."""
    mat = _from_msg_to_matrix(msg.pose)
    mat[:3, 3] = mat[:3, 3] / SimToSI.linear_factor
    return mat


# ============================================================================
# 1.  Topic enumeration
# ============================================================================
class RosTopics(Enum):
    CAMERA_FRAME         = ("/ambf/env/phantom/CameraFrame/State",      RigidBodyState)
    NEEDLE               = ("/ambf/env/phantom/Needle/State",           RigidBodyState)
    CAMERA_L_STATE       = ("/ambf/env/cameras/cameraL/State",          CameraState)
    CAMERA_L_IMAGE       = ("/ambf/env/cameras/cameraL/ImageData",      Image)
    CAMERA_L_SEG_IMAGE   = ("/ambf/env/cameras/cameraL2/ImageData",     Image)
    CAMERA_L_DEPTH       = ("/ambf/env/cameras/cameraL/DepthData",      PointCloud2)
    CAMERA_0_IMAGE       = ("/cameras/camera_0/ImageData",              Image)
    CAMERA_0_STATE       = ("/cameras/camera_0/State",                  CameraState)
    PSM1_TOOL_PITCH_LINK = ("/ambf/env/psm1/toolpitchlink/State",       RigidBodyState)
    PSM2_TOOL_PITCH_LINK = ("/ambf/env/psm2/toolpitchlink/State",       RigidBodyState)
    PSM1_TOOL_YAW_LINK   = ("/ambf/env/psm1/toolyawlink/State",         RigidBodyState)
    PSM2_TOOL_YAW_LINK   = ("/ambf/env/psm2/toolyawlink/State",         RigidBodyState)


# ============================================================================
# 2.  topic → RawSimulationData attribute map
# ============================================================================
topic_to_attr_dict: Dict[RosTopics, str] = {
    RosTopics.CAMERA_FRAME:         "camera_frame_pose",
    RosTopics.NEEDLE:               "needle_pose",
    RosTopics.CAMERA_L_STATE:       "camera_l_pose",
    RosTopics.CAMERA_L_IMAGE:       "camera_l_img",
    RosTopics.CAMERA_L_SEG_IMAGE:   "camera_l_seg_img",
    RosTopics.CAMERA_L_DEPTH:       "camera_l_depth",
    RosTopics.CAMERA_0_IMAGE:       "camera_0_img",
    RosTopics.CAMERA_0_STATE:       "camera_0_pose",
    RosTopics.PSM1_TOOL_PITCH_LINK: "psm1_toolpitchlink_pose",
    RosTopics.PSM2_TOOL_PITCH_LINK: "psm2_toolpitchlink_pose",
    RosTopics.PSM1_TOOL_YAW_LINK:   "psm1_toolyawlink_pose",
    RosTopics.PSM2_TOOL_YAW_LINK:   "psm2_toolyawlink_pose",
}


# ============================================================================
# 3.  Public callback factory
# ============================================================================
def get_topics_processing_cb() -> Dict[RosTopics, Callable[[Any], np.ndarray]]:
    """
    Return a dict mapping each RosTopics member to a callable that converts
    its incoming ROS 2 message into a NumPy array.
    """
    img_cb = _make_image_processor()
    pcd_cb = _make_point_cloud_processor()

    return {
        RosTopics.CAMERA_FRAME:          _process_pose,
        RosTopics.NEEDLE:                _process_pose,
        RosTopics.CAMERA_L_STATE:        _process_pose,
        RosTopics.CAMERA_L_IMAGE:        img_cb,
        RosTopics.CAMERA_L_SEG_IMAGE:    img_cb,
        RosTopics.CAMERA_L_DEPTH:        pcd_cb,
        RosTopics.CAMERA_0_IMAGE:        img_cb,
        RosTopics.CAMERA_0_STATE:        _process_pose,
        RosTopics.PSM1_TOOL_PITCH_LINK:  _process_pose,
        RosTopics.PSM2_TOOL_PITCH_LINK:  _process_pose,
        RosTopics.PSM1_TOOL_YAW_LINK:    _process_pose,
        RosTopics.PSM2_TOOL_YAW_LINK:    _process_pose,
    }


# ============================================================================
# 4.  Internal helpers
# ============================================================================

def _make_image_processor() -> Callable[[Image], np.ndarray]:
    bridge = CvBridge()

    def _proc(msg: Image) -> np.ndarray:
        return bridge.imgmsg_to_cv2(msg, "bgr8")

    return _proc


def _make_point_cloud_processor() -> Callable[[PointCloud2], np.ndarray]:
    w, h  = 640, 480
    scale = (1.0 / SimToSI.linear_factor) * 1000.0
    xform = np.array([[0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

    def _proc(msg: PointCloud2) -> np.ndarray:
        fmt       = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        data      = np.frombuffer(msg.data, dtype=np.uint8)
        reshaped  = data.reshape(-1, msg.point_step)
        xyz_bytes = reshaped[:, 0:12].view(dtype=fmt)
        depth = np.concatenate(
            (xyz_bytes["x"], xyz_bytes["y"], xyz_bytes["z"]), axis=-1
        ).astype(np.float32) * scale
        depth = depth.reshape(h, w, 3)[::-1].copy()
        depth = np.einsum("ab,hwb->hwa", xform[:3, :3], depth)[..., -1]
        return depth

    return _proc