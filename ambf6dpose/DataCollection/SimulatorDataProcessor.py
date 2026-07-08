# simulator_data_processor_ros2.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional
import types

import cv2
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as SciPyRot

# --------------------------------------------------------------------------- #
# NumPy/SciPy Shim — Replacing PyKDL
# --------------------------------------------------------------------------- #
def _from_msg(p):
    m = np.eye(4)
    quat = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    m[:3, :3] = SciPyRot.from_quat(quat).as_matrix()
    m[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return m

def _to_matrix(m: np.ndarray) -> np.ndarray:
    return m

pm = types.SimpleNamespace(fromMsg=_from_msg, toMatrix=_to_matrix)

# --------------------------------------------------------------------------- #
# Application-specific imports
# --------------------------------------------------------------------------- #
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose import AbstractSimulationClient, RawSimulationData

np.set_printoptions(precision=3, suppress=True)


# --------------------------------------------------------------------------- #
# CameraConfig
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    """
    Describes a single camera for data collection.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "camera_l", "camera_0").
    img_attr : str
        Attribute on RawSimulationData holding this camera's image.
    pose_attr : str
        Attribute on RawSimulationData holding this camera's pose.
    seg_img_attr : str, optional
        Attribute on RawSimulationData holding this camera's segmentation image.
        If None the raw image is reused as a placeholder.
    is_world_frame : bool
        If True, the camera's pose is expressed directly in world coordinates
        (i.e. independent multibody cameras like camera_0 and camera_1).
        If False, the pose is expressed relative to camera_frame_pose and
        must be composed with it to reach world coordinates (i.e. camera_l).
    """
    name: str
    img_attr: str
    pose_attr: str
    seg_img_attr: Optional[str] = None
    is_world_frame: bool = False


# --------------------------------------------------------------------------- #
# SimulatorDataProcessor
# --------------------------------------------------------------------------- #
@dataclass
class SimulatorDataProcessor:
    """
    Express poses in terms of each registered camera frame and convert units
    to mm.  One DatasetSample is produced per camera; all samples originate
    from the same RawSimulationData snapshot, guaranteeing time synchronisation.
    """
    simulation_client: AbstractSimulationClient
    camera_registry: List[CameraConfig] = field(default_factory=list)

    def __post_init__(self):
        if not self.camera_registry:
            self.camera_registry = [
                CameraConfig(
                    name="camera_l",
                    img_attr="camera_l_img",
                    pose_attr="camera_l_pose",
                    seg_img_attr="camera_l_seg_img",
                    is_world_frame=False,   # mounted on CameraFrame phantom
                ),
                CameraConfig(
                    name="camera_0",
                    img_attr="camera_0_img",
                    pose_attr="camera_0_pose",
                    seg_img_attr=None,
                    is_world_frame=True,    # independent multibody, pose in world coords
                ),
                CameraConfig(
                    name="camera_1",
                    img_attr="camera_1_img",
                    pose_attr="camera_1_pose",
                    seg_img_attr=None,
                    is_world_frame=True,    # independent multibody, pose in world coords
                ),
            ]
        self.intrinsic_params = self.calculate_intrinsics()

    # ---------------------------------------------------------------------- #
    # Intrinsics
    # ---------------------------------------------------------------------- #
    def calculate_intrinsics(self) -> np.ndarray:
        fov = 1.2
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

    # ---------------------------------------------------------------------- #
    # Pose helpers
    # ---------------------------------------------------------------------- #
    def get_needle_extrinsics(
        self,
        raw: RawSimulationData,
        cam_pose: np.ndarray,
        is_world_frame: bool,
    ) -> np.ndarray:
        pose = self.transform_from_world_to_cam(
            raw.needle_pose, cam_pose, raw.camera_frame_pose, is_world_frame
        )
        return self.convert_pose_to_mm(pose)

    def get_tool_pose(
        self,
        raw_tool_pose: np.ndarray,
        cam_pose: np.ndarray,
        raw: RawSimulationData,
        is_world_frame: bool,
    ) -> np.ndarray:
        pose = self.transform_from_world_to_cam(
            raw_tool_pose, cam_pose, raw.camera_frame_pose, is_world_frame
        )
        return self.convert_pose_to_mm(pose)

    def transform_from_world_to_cam(
        self,
        obj2world_pose: np.ndarray,
        cam_pose: np.ndarray,
        camframe2world_pose: np.ndarray,
        is_world_frame: bool,
    ) -> np.ndarray:
        """
        Transform an object pose into the camera frame.

        For camera_l (is_world_frame=False):
            The camera pose is relative to the CameraFrame phantom, so we
            first compose cam_pose with camframe2world_pose to get the camera
            in world coordinates, then invert to get world-to-cam.

            T_WL = T_WF @ T_FL
            T_LW = inv(T_WL)

        For camera_0/camera_1 (is_world_frame=True):
            The camera pose is already in world coordinates (independent
            multibody), so we invert it directly.

            T_LW = inv(cam_pose)
        """
        T_W_OBJ = obj2world_pose

        if is_world_frame:
            # cam_pose is already T_WL (camera in world coords)
            T_WL = cam_pose
        else:
            # cam_pose is T_FL (camera relative to CameraFrame)
            T_WF = camframe2world_pose
            T_FL = cam_pose
            T_WL = T_WF @ T_FL

        T_LW    = inv(T_WL)
        T_L_OBJ = T_LW @ T_W_OBJ

        # Convert AMBF camera axis to OpenCV camera axis (Z forward, Y down)
        F = np.array([[0, 1, 0, 0],
                      [0, 0, -1, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1]])
        return F @ T_L_OBJ

    def convert_pose_to_mm(self, pose: np.ndarray) -> np.ndarray:
        pose[:3, 3] *= 1000.0
        return pose

    # ---------------------------------------------------------------------- #
    # Per-camera sample builder
    # ---------------------------------------------------------------------- #
    def _build_sample_for_camera(
        self,
        raw: RawSimulationData,
        cfg: CameraConfig,
    ) -> DatasetSample:
        cam_pose = getattr(raw, cfg.pose_attr)
        img      = getattr(raw, cfg.img_attr)
        seg_img  = (
            getattr(raw, cfg.seg_img_attr)
            if cfg.seg_img_attr is not None
            else img
        )
        depth_img = raw.camera_l_depth

        needle_pose     = self.get_needle_extrinsics(raw, cam_pose, cfg.is_world_frame)
        psm1_pitch_pose = self.get_tool_pose(raw.psm1_toolpitchlink_pose, cam_pose, raw, cfg.is_world_frame)
        psm2_pitch_pose = self.get_tool_pose(raw.psm2_toolpitchlink_pose, cam_pose, raw, cfg.is_world_frame)
        psm1_yaw_pose   = self.get_tool_pose(raw.psm1_toolyawlink_pose,   cam_pose, raw, cfg.is_world_frame)
        psm2_yaw_pose   = self.get_tool_pose(raw.psm2_toolyawlink_pose,   cam_pose, raw, cfg.is_world_frame)

        return DatasetSample(
            img,
            seg_img,
            depth_img,
            needle_pose,
            psm1_pitch_pose,
            psm2_pitch_pose,
            psm1_yaw_pose,
            psm2_yaw_pose,
            self.intrinsic_params,
        )

    # ---------------------------------------------------------------------- #
    # Public entry point
    # ---------------------------------------------------------------------- #
    def generate_dataset_sample(self) -> List[DatasetSample]:
        """
        Returns one DatasetSample per camera in self.camera_registry, all
        built from a single synchronised RawSimulationData snapshot.

        Zip with camera_registry to get the camera name alongside each sample:

            for cfg, sample in zip(processor.camera_registry, samples):
                saver.save_sample(cfg.name, sample)
        """
        raw = self.simulation_client.get_data()

        return [
            self._build_sample_for_camera(raw, cfg)
            for cfg in self.camera_registry
        ]


if __name__ == "__main__":
    from ambf6dpose import SyncRosInterface
    client = SyncRosInterface()
    client.wait_for_data(timeout=10)

    sim_interface = SimulatorDataProcessor(client)
    samples = sim_interface.generate_dataset_sample()

    for cfg, sample in zip(sim_interface.camera_registry, samples):
        sample.generate_gt_vis()
        cv2.imshow(f"blended [{cfg.name}]", sample.gt_vis_img)

    cv2.waitKey(0)