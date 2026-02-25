# rostopics_ros2.py  –  identical API to the old ROS 1 helper
from __future__ import annotations

from typing import Any, Callable, Dict
from enum import Enum

# ————————————————————————  ROS 2 message types  ————————————————————————
from sensor_msgs.msg import Image, PointCloud2                     # type: ignore
from ambf_msgs.msg import CameraState, RigidBodyState              # type: ignore

# ————————————————————————  Pose helpers (ROS 1 → ROS 2 shim)  —————————————————
try:                            # If you manually installed tf_conversions for ROS 2
    import tf_conversions.posemath as pm                           # type: ignore
except ModuleNotFoundError:      # Minimal drop-in replacement
    import types
    import numpy as np
    from geometry_msgs.msg import Pose

    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import types

    # ————————————————————————  Pose helpers (No PyKDL version) FOR RENDERING —————————————————
    '''def _from_msg_to_matrix(msg_pose):
        # msg_pose is expected to have .position and .orientation
        p = msg_pose.position
        o = msg_pose.orientation
        
        # 1. Create 4x4 Identity Matrix
        m = np.eye(4)
        
        # 2. Set Rotation from Quaternion (x, y, z, w)
        quat = [o.x, o.y, o.z, o.w]
        m[:3, :3] = R.from_quat(quat).as_matrix()
        
        # 3. Set Position
        m[:3, 3] = [p.x, p.y, p.z]
        
        return m

    def _process_pose(msg: RigidBodyState) -> np.ndarray:
        # Get the raw matrix
        mat = _from_msg_to_matrix(msg.pose)
        
        # Apply units conversion (scaling the translation/position column)
        # Equivalent to PyKDL: f.p / SimToSI.linear_factor
        mat[:3, 3] = mat[:3, 3] / SimToSI.linear_factor
        
        return mat

    # Create a "shim" so the rest of your code works without changing logic
    pm = types.SimpleNamespace(toMatrix=lambda x: x, fromMsg=lambda x: x)'''

    # ————————————————————————  (PyKDL version) FOR DATA COLLECTION —————————————————
    import PyKDL

    def _from_msg(p: Pose) -> PyKDL.Frame:
        rot = PyKDL.Rotation.Quaternion(p.orientation.x, p.orientation.y,
                                        p.orientation.z, p.orientation.w)
        pos = PyKDL.Vector(p.position.x, p.position.y, p.position.z)
        return PyKDL.Frame(rot, pos)

    def _to_matrix(f: PyKDL.Frame) -> np.ndarray:
        m = np.eye(4, dtype=float)
        for i in range(3):
            for j in range(3):
                m[i, j] = f.M[i, j]
        m[:3, 3] = (f.p.x(), f.p.y(), f.p.z())
        return m

    pm = types.SimpleNamespace(fromMsg=_from_msg, toMatrix=_to_matrix) # type: ignore

# ————————————————————————  Remaining unchanged deps  ————————————————————
import numpy as np
#import PyKDL
from cv_bridge import CvBridge                                     # type: ignore
'''try:
    import ros2_numpy as ros_numpy              # ROS 2 fork
except ModuleNotFoundError:
    import ros_numpy'''                                            # type: ignore
from surgical_robotics_challenge.units_conversion import SimToSI


# ============================================================================
# 1.  Topic enumeration (unchanged)
# ============================================================================
class RosTopics(Enum):
    CAMERA_L_STATE        = ("/ambf/env/cameras/cameraL/State",           CameraState)
    CAMERA_FRAME          = ("/ambf/env/phantom/CameraFrame/State",       RigidBodyState)
    NEEDLE                = ("/ambf/env/phantom/Needle/State",            RigidBodyState)
    CAMERA_L_IMAGE        = ("/ambf/env/cameras/cameraL/ImageData",       Image)
    CAMERA_L_SEG_IMAGE    = ("/ambf/env/cameras/cameraL2/ImageData",      Image)
    #CAMERA_L_DEPTH        = ("/ambf/env/cameras/cameraL/DepthData",       PointCloud2)
    PSM1_TOOL_PITCH_LINK  = ("/ambf/env/psm1/toolpitchlink/State",        RigidBodyState)
    PSM2_TOOL_PITCH_LINK  = ("/ambf/env/psm2/toolpitchlink/State",        RigidBodyState)
    PSM1_TOOL_YAW_LINK    = ("/ambf/env/psm1/toolyawlinksimple/State",          RigidBodyState)
    PSM2_TOOL_YAW_LINK    = ("/ambf/env/psm2/toolyawlinksimple/State",          RigidBodyState)


# ============================================================================
# 2.  topic → RawSimulationData attribute map  (unchanged)
# ============================================================================
topic_to_attr_dict: Dict[RosTopics, str] = {
    RosTopics.CAMERA_L_STATE:       "camera_l_pose",
    RosTopics.CAMERA_FRAME:         "camera_frame_pose",
    RosTopics.NEEDLE:               "needle_pose",
    RosTopics.CAMERA_L_IMAGE:       "camera_l_img",
    RosTopics.CAMERA_L_SEG_IMAGE:   "camera_l_seg_img",
    #RosTopics.CAMERA_L_DEPTH:       "camera_l_depth",
    RosTopics.PSM1_TOOL_PITCH_LINK: "psm1_toolpitchlink_pose",
    RosTopics.PSM2_TOOL_PITCH_LINK: "psm2_toolpitchlink_pose",
    RosTopics.PSM1_TOOL_YAW_LINK:   "psm1_toolyawlink_pose",
    RosTopics.PSM2_TOOL_YAW_LINK:   "psm2_toolyawlink_pose",
}


# ============================================================================
# 3.  Public factory – exactly the same signature as before
# ============================================================================
def get_topics_processing_cb() -> Dict[RosTopics, Callable[[Any], np.ndarray]]:
    """
    Return a dict that maps each :class:`RosTopics` to a callable that converts
    its incoming ROS 2 message into a NumPy array.
    """
    img_cb   = _make_image_processor()
    pcd_cb   = _make_point_cloud_processor()

    return {
        RosTopics.CAMERA_L_STATE:        _process_pose,
        RosTopics.CAMERA_FRAME:          _process_pose,
        RosTopics.NEEDLE:                _process_pose,
        RosTopics.CAMERA_L_IMAGE:        img_cb,
        RosTopics.CAMERA_L_SEG_IMAGE:    img_cb,
        #RosTopics.CAMERA_L_DEPTH:        pcd_cb,
        RosTopics.PSM1_TOOL_PITCH_LINK:  _process_pose,
        RosTopics.PSM2_TOOL_PITCH_LINK:  _process_pose,
        RosTopics.PSM1_TOOL_YAW_LINK:    _process_pose,
        RosTopics.PSM2_TOOL_YAW_LINK:    _process_pose,
    }


# ============================================================================
# 4.  Internal helpers (prefixed “_” but otherwise identical behaviour)
# ============================================================================
def _convert_units(f: PyKDL.Frame) -> PyKDL.Frame:
    return PyKDL.Frame(f.M, f.p / SimToSI.linear_factor)


def _process_pose(msg: RigidBodyState) -> np.ndarray:
    return pm.toMatrix(_convert_units(pm.fromMsg(msg.pose)))


# ———————————————————————  Images  ————————————————————————
def _make_image_processor() -> Callable[[Image], np.ndarray]:
    bridge = CvBridge()

    def _proc(msg: Image) -> np.ndarray:
        return bridge.imgmsg_to_cv2(msg, "bgr8")

    return _proc


# ————————————————————  Point clouds  ————————————————————
# ————————————————————  Point clouds  ————————————————————
def _make_point_cloud_processor() -> Callable[[PointCloud2], np.ndarray]:
    w, h   = 640, 480
    scale  = (1.0 / SimToSI.linear_factor) * 1000.0          #  → mm
    xform  = np.array([[0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [-1, 0, 0, 0],
                       [0, 0, 0, 1]], dtype=np.float32)       # T_cv_ambf

    def _proc(msg: PointCloud2) -> np.ndarray:
        # MANUAL REPLACEMENT FOR ros_numpy
        # PointCloud2 data is a raw byte buffer. We cast it to a structured numpy array.
        # Most ROS2 depth clouds use 32-bit floats for x, y, z.
        
        # 1. Convert raw bytes to a numpy record array
        fmt = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        # We skip the extra fields (like intensity/rgb) by calculating the point_step
        data = np.frombuffer(msg.data, dtype=np.uint8)
        
        # 2. Reshape and extract just the first 12 bytes (x,y,z) of every point_step
        reshaped = data.reshape(-1, msg.point_step)
        xyz_bytes = reshaped[:, 0:12].view(dtype=fmt)
        
        # 3. Reconstruct the depth map
        depth = np.concatenate(
            (xyz_bytes["x"], xyz_bytes["y"], xyz_bytes["z"]), axis=-1
        ).astype(np.float32) * scale

        # 4. Final reshaping and transform (matching your original logic)
        depth = depth.reshape(h, w, 3)[::-1].copy()          # flip height (AMBF)
        depth = np.einsum("ab,hwb->hwa", xform[:3, :3], depth)[..., -1]
        return depth                                         # float32 (mm)

    return _proc

'''def _make_point_cloud_processor() -> Callable[[PointCloud2], np.ndarray]:
    w, h   = 640, 480
    scale  = (1.0 / SimToSI.linear_factor) * 1000.0          #  → mm
    xform  = np.array([[0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [-1, 0, 0, 0],
                       [0, 0, 0, 1]], dtype=np.float32)       # T_cv_ambf

    def _proc(msg: PointCloud2) -> np.ndarray:
        xyz = ros_numpy.point_cloud2.point_cloud2_to_array(msg)
        depth = np.concatenate(
            (xyz["x"][:, None], xyz["y"][:, None], xyz["z"][:, None]), axis=-1
        ) * scale

        depth = depth.reshape(h, w, 3)[::-1].copy()          # flip height (AMBF)
        depth = np.einsum("ab,hwb->hwa", xform[:3, :3], depth)[..., -1]
        return depth                                         # float32 (mm)

    return _proc'''
