# simulator_data_processor_ros2.py
from __future__ import annotations

import time
from dataclasses import dataclass
import types

import cv2
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as SciPyRot

# --------------------------------------------------------------------------- #
# NumPy/SciPy Shim — Replacing PyKDL
# --------------------------------------------------------------------------- #
def _from_msg(p):
    """
    Converts a geometry_msgs/Pose to a 4x4 NumPy matrix.
    Works with both ROS2 message objects and simple objects with .position/.orientation
    """
    m = np.eye(4)
    # Rotation from Quaternion (x, y, z, w)
    quat = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    m[:3, :3] = SciPyRot.from_quat(quat).as_matrix()
    # Translation
    m[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return m

def _to_matrix(m: np.ndarray) -> np.ndarray:
    """Since we are using NumPy now, this just returns the matrix back."""
    return m

# Create the 'pm' namespace to satisfy existing code logic
pm = types.SimpleNamespace(fromMsg=_from_msg, toMatrix=_to_matrix)

# --------------------------------------------------------------------------- #
# Application-specific imports
# --------------------------------------------------------------------------- #
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose import AbstractSimulationClient, RawSimulationData

np.set_printoptions(precision=3, suppress=True)


@dataclass
class SimulatorDataProcessor:
    """
    Express poses in terms of the camera-left frame and convert units to mm.
    """
    simulation_client: AbstractSimulationClient

    def __post_init__(self):
        self.intrinsic_params = self.calculate_intrinsics()

    def calculate_intrinsics(self) -> np.ndarray:
        fov = 1.2                        # rad
        w, h = 640, 480
        f = h / (2 * np.tan(fov / 2))

        K = np.zeros((3, 3))
        K[0, 0] = K[1, 1] = f
        K[0, 2] = w / 2
        K[1, 2] = h / 2
        K[2, 2] = 1.0
        return K

    def get_intrinsics(self) -> np.ndarray:
        return self.intrinsic_params

    def get_needle_extrinsics(self, raw: RawSimulationData) -> np.ndarray:
        pose = self.transform_from_world_to_cam_l(
            raw.needle_pose, raw.camera_l_pose, raw.camera_frame_pose
        )
        return self.convert_pose_to_mm(pose)

    def get_tool_pose(self, raw: RawSimulationData, raw_tool_pose: np.ndarray) -> np.ndarray:
        pose = self.transform_from_world_to_cam_l(
            raw_tool_pose, raw.camera_l_pose, raw.camera_frame_pose
        )
        return self.convert_pose_to_mm(pose)

    def transform_from_world_to_cam_l(
        self,
        obj2world_pose: np.ndarray,
        caml2camframe_pose: np.ndarray,
        camframe2world_pose: np.ndarray,
    ) -> np.ndarray:
        # T_A_B means "Pose of B relative to A"
        T_W_OBJ = obj2world_pose
        T_FL = caml2camframe_pose
        T_WF = camframe2world_pose

        T_WL = T_WF @ T_FL            # CamL -> World
        T_LW = inv(T_WL)              # World -> CamL
        T_L_OBJ = T_LW @ T_W_OBJ      # Obj  -> CamL

        # 
        # Convert AMBF camera axis (Z forward) to OpenCV camera axis (Z forward, Y down)
        F = np.array([[0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
        return F @ T_L_OBJ

    def convert_pose_to_mm(self, pose: np.ndarray) -> np.ndarray:
        # BOP format requires translation in mm
        pose[:3, 3] *= 1000.0
        return pose

    def generate_dataset_sample(self) -> DatasetSample:
        raw = self.simulation_client.get_data()

        # These methods now use the NumPy-based transform logic
        needle_pose        = self.get_needle_extrinsics(raw)
        psm1_pitch_pose    = self.get_tool_pose(raw, raw.psm1_toolpitchlink_pose)
        psm2_pitch_pose    = self.get_tool_pose(raw, raw.psm2_toolpitchlink_pose)
        psm1_yaw_pose      = self.get_tool_pose(raw, raw.psm1_toolyawlink_pose)
        psm2_yaw_pose      = self.get_tool_pose(raw, raw.psm2_toolyawlink_pose)

        return DatasetSample(
            raw.camera_l_img,
            raw.camera_l_seg_img,
            needle_pose,
            psm1_pitch_pose,
            psm2_pitch_pose,
            psm1_yaw_pose,
            psm2_yaw_pose,
            self.intrinsic_params,
        )

if __name__ == "__main__":
    from ambf6dpose import SyncRosInterface
    client = SyncRosInterface()
    # Ensure ROS node is active
    client.wait_for_data(timeout=10)

    sim_interface = SimulatorDataProcessor(client)
    sample = sim_interface.generate_dataset_sample()
    sample.generate_gt_vis()

    cv2.imshow("blended", sample.gt_vis_img)
    cv2.waitKey(0)
    
# simulator_data_processor_ros2.py
'''from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.linalg import inv

# --------------------------------------------------------------------------- #
# Pose-math import — keep the name “pm”, but fall back to a small ROS 2 shim
# --------------------------------------------------------------------------- #
try:
    import tf_conversions.posemath as pm                 # ROS 1 style (if installed)
except ModuleNotFoundError:                              # ROS 2 fallback
    import types, PyKDL
    from geometry_msgs.msg import Pose

    def _from_msg(p: Pose) -> PyKDL.Frame:
        rot = PyKDL.Rotation.Quaternion(p.orientation.x, p.orientation.y,
                                        p.orientation.z, p.orientation.w)
        pos = PyKDL.Vector(p.position.x, p.position.y, p.position.z)
        return PyKDL.Frame(rot, pos)

    def _to_matrix(f: PyKDL.Frame) -> np.ndarray:
        m = np.eye(4)
        for i in range(3):
            for j in range(3):
                m[i, j] = f.M[i, j]
        m[:3, 3] = (f.p.x(), f.p.y(), f.p.z())
        return m

    pm = types.SimpleNamespace(fromMsg=_from_msg, toMatrix=_to_matrix)

# --------------------------------------------------------------------------- #
# Application-specific imports (unchanged public names)
# --------------------------------------------------------------------------- #
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose import AbstractSimulationClient, RawSimulationData

np.set_printoptions(precision=3, suppress=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core class – unchanged public API
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SimulatorDataProcessor:
    """
    Express poses in terms of the camera-left frame and convert units to mm
    to comply with BOP datasets.
    """

    simulation_client: AbstractSimulationClient

    def __post_init__(self):
        self.intrinsic_params = self.calculate_intrinsics()

    # ----------------------------- helpers ----------------------------------
    def calculate_intrinsics(self) -> np.ndarray:
        fov = 1.2                        # rad
        w, h = 640, 480
        f = h / (2 * np.tan(fov / 2))

        K = np.zeros((3, 3))
        K[0, 0] = K[1, 1] = f
        K[0, 2] = w / 2
        K[1, 2] = h / 2
        K[2, 2] = 1.0
        return K

    def get_intrinsics(self) -> np.ndarray:
        return self.intrinsic_params

    # -----------------------------------------------------------------------
    # Pose conversions (unchanged signatures & behaviour)
    # -----------------------------------------------------------------------
    def get_needle_extrinsics(self, raw: RawSimulationData) -> np.ndarray:
        pose = self.transform_from_world_to_cam_l(
            raw.needle_pose, raw.camera_l_pose, raw.camera_frame_pose
        )
        return self.convert_pose_to_mm(pose)

    def get_tool_pose(
        self, raw: RawSimulationData, raw_tool_pose: np.ndarray
    ) -> np.ndarray:
        pose = self.transform_from_world_to_cam_l(
            raw_tool_pose, raw.camera_l_pose, raw.camera_frame_pose
        )
        return self.convert_pose_to_mm(pose)

    def transform_from_world_to_cam_l(
        self,
        obj2world_pose: np.ndarray,
        caml2camframe_pose: np.ndarray,
        camframe2world_pose: np.ndarray,
    ) -> np.ndarray:
        T_W_OBJ = obj2world_pose
        T_FL = caml2camframe_pose
        T_WF = camframe2world_pose

        T_WL = T_WF @ T_FL            # CamL → World
        T_LW = inv(T_WL)              # World → CamL
        T_L_OBJ = T_LW @ T_W_OBJ      # Obj  → CamL

        # Convert AMBF camera axis to OpenCV camera axis
        F = np.array([[0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
        return F @ T_L_OBJ

    def convert_pose_to_mm(self, pose: np.ndarray) -> np.ndarray:
        pose[:3, 3] *= 1000.0
        return pose

    # -----------------------------------------------------------------------
    # Dataset generation (unchanged public interface)
    # -----------------------------------------------------------------------
    def generate_dataset_sample(self) -> DatasetSample:
        print('simhere4')
        raw = self.simulation_client.get_data()

        needle_pose        = self.get_needle_extrinsics(raw)
        psm1_pitch_pose    = self.get_tool_pose(raw, raw.psm1_toolpitchlink_pose)
        psm2_pitch_pose    = self.get_tool_pose(raw, raw.psm2_toolpitchlink_pose)
        psm1_yaw_pose      = self.get_tool_pose(raw, raw.psm1_toolyawlink_pose)
        psm2_yaw_pose      = self.get_tool_pose(raw, raw.psm2_toolyawlink_pose)
        print('simhere1')
        return DatasetSample(
            raw.camera_l_img,
            raw.camera_l_seg_img,
            #raw.camera_l_depth,
            needle_pose,
            psm1_pitch_pose,
            psm2_pitch_pose,
            psm1_yaw_pose,
            psm2_yaw_pose,
            self.intrinsic_params,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone demo (unchanged behaviour, SyncRosInterface is now ROS 2-ready)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ambf6dpose import SyncRosInterface
    client = SyncRosInterface()
    client.wait_for_data(timeout=10)

    sim_interface = SimulatorDataProcessor(client)
    sample = sim_interface.generate_dataset_sample()
    sample.generate_gt_vis()

    cv2.imshow("blended", sample.gt_vis_img)
    cv2.waitKey(0)'''
