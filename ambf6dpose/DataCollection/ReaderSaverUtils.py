from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
import numpy as np
from enum import Enum, auto
import cv2


class ImgDirs(Enum):
    RAW = auto()
    SEGMENTED = auto()
    DEPTH = auto()
    GT_VISUALIZATION = auto()


def trnorm(rot: np.ndarray):
    """Convert to proper rotation matrix
    https://petercorke.github.io/spatialmath-python/func_3d.html?highlight=trnorm#spatialmath.base.transforms3d.trnorm

    Parameters
    ----------
    rot : np.ndarray
        3x3 numpy array

    Returns
    -------
    proper_rot
        proper rotation matrix
    """
    unitvec = lambda x: x / np.linalg.norm(x)
    o = rot[:3, 1]
    a = rot[:3, 2]

    n = np.cross(o, a)        # N = O x A
    o = np.cross(a, n)
    new_rot = np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)
    return new_rot


def is_rotation(rot: np.ndarray, tol=100):
    """Test if matrix is a proper rotation matrix

    Taken from
    https://petercorke.github.io/spatialmath-python/func_nd.html#spatialmath.base.transformsNd.isR

    Parameters
    ----------
    rot : np.ndarray
        3x3 np.array
    """
    _eps = np.finfo(np.float64).eps
    return (
        np.linalg.norm(rot @ rot.T - np.eye(rot.shape[0])) < tol * _eps
        and np.linalg.det(rot @ rot.T) > 0
    )


# --------------------------------------------------------------------------- #
# Abstract base classes (unchanged public API)
# --------------------------------------------------------------------------- #

@dataclass
class AbstractSaver(ABC):
    root: Path

    @abstractmethod
    def save_sample(self, sample: DatasetSample) -> None:
        """Save a single DatasetSample. Implemented by each concrete saver."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@dataclass
class AbstractReader(ABC):
    root: Path


# --------------------------------------------------------------------------- #
# MultiCameraSaver
# --------------------------------------------------------------------------- #

@dataclass
class MultiCameraSaver:
    """
    Creates one child saver per camera under ``root/<camera_name>/`` and
    delegates saving to each child in lock-step with the list of samples
    returned by ``SimulatorDataProcessor.generate_dataset_sample()``.

    Parameters
    ----------
    root : Path
        Top-level dataset directory (e.g. ``./dataset/scene_01``).
    camera_names : list[str]
        Ordered list of camera names — must match the order of
        ``SimulatorDataProcessor.camera_registry``.
    saver_factory : Callable[[Path], AbstractSaver]
        A zero-argument-aside-from-root factory that constructs the
        concrete saver (BopSampleSaver, YamlSampleSaver, …) for a
        given per-camera root directory.

    Directory layout produced
    -------------------------
    root/
        camera_l/
            rgb/  masks/  scene_gt.json  …   (BOP layout)
        camera_0/
            rgb/  masks/  scene_gt.json  …
        camera_1/
            rgb/  masks/  scene_gt.json  …
    """
    root: Path
    camera_names: List[str]
    saver_factory: Callable[[Path], AbstractSaver]
    _savers: Dict[str, AbstractSaver] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for name in self.camera_names:
            camera_root = self.root / name
            camera_root.mkdir(parents=True, exist_ok=True)
            self._savers[name] = self.saver_factory(camera_root)

    # ------------------------------------------------------------------ #
    # context-manager support so 'with saver:' still flushes JSON metadata
    # ------------------------------------------------------------------ #
    def __enter__(self):
        for saver in self._savers.values():
            saver.__enter__()
        return self

    def __exit__(self, *args):
        for saver in self._savers.values():
            saver.__exit__(*args)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def save_samples(self, samples: List[DatasetSample]) -> None:
        """
        Save one DatasetSample per camera.

        Parameters
        ----------
        samples : list[DatasetSample]
            Must be the list returned by
            ``SimulatorDataProcessor.generate_dataset_sample()``.
            Length and order must match ``self.camera_names``.
        """
        if len(samples) != len(self.camera_names):
            raise ValueError(
                f"Expected {len(self.camera_names)} samples "
                f"(one per camera), got {len(samples)}."
            )
        for name, sample in zip(self.camera_names, samples):
            self._savers[name].save_sample(sample)


# --------------------------------------------------------------------------- #
# ImageSaver  (unchanged)
# --------------------------------------------------------------------------- #

@dataclass
class ImageSaver:
    """Utility to save images into a folder structure defined by
    ``folder_names``.  The dict ``folder_names`` needs to include values for
    all the constants defined in ``ImgDirs`` enum.
    """
    root_path: Path
    folder_names: Dict[ImgDirs, str]

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.get_img_dirs()
        self.create_dir()

    def get_img_dirs(self) -> Dict[ImgDirs, Path]:
        img_dirs = {}
        for img_dir in ImgDirs:
            img_dirs[img_dir] = self.root_path / self.folder_names[img_dir]
        return img_dirs

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True)

    def save_sample(self, str_step: str, data: DatasetSample):
        raw_path       = str(self.dir_dict[ImgDirs.RAW]       / f"{str_step}.png")
        segmented_path = str(self.dir_dict[ImgDirs.SEGMENTED]  / f"{str_step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

        save_depth(self.dir_dict[ImgDirs.DEPTH] / f"{str_step}.png", data.depth_img)  # ← uncomment this

        # Save depth
        # save_depth(self.dir_dict[ImgDirs.DEPTH] / f"{str_step}.png", data.depth_img)

        # Generating gt_vis reduces the script performance
        # data.generate_gt_vis()
        # blended_path = str(self.dir_dict[ImgDirs.GT_VISUALIZATION] / f"{str_step}.png")
        # cv2.imwrite(blended_path, data.blended_img)


# --------------------------------------------------------------------------- #
# Depth helper  (unchanged)
# --------------------------------------------------------------------------- #

def save_depth(path: Path, im: np.ndarray):
    """
    Saves a depth image (16-bit) to a PNG file using OpenCV.
    Maintains 100% compatibility with the original depth data.
    """
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError('Only PNG format is currently supported.')

    im_uint16 = np.round(im).astype(np.uint16)
    success = cv2.imwrite(path_str, im_uint16)

    if not success:
        raise IOError(f"Failed to save depth image to {path_str}")
        
'''from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
import numpy as np
from enum import Enum, auto
import cv2
#import png

class ImgDirs(Enum):
    RAW = auto()
    SEGMENTED = auto()
    DEPTH = auto()
    GT_VISUALIZATION = auto()

def trnorm(rot: np.ndarray):
    """Convert to proper rotation matrix
    https://petercorke.github.io/spatialmath-python/func_3d.html?highlight=trnorm#spatialmath.base.transforms3d.trnorm

    Parameters
    ----------
    rot : np.ndarray
        3x3 numpy array

    Returns
    -------
    proper_rot
        proper rotation matrix
    """

    unitvec = lambda x: x / np.linalg.norm(x)
    o = rot[:3, 1]
    a = rot[:3, 2]

    n = np.cross(o, a)  # N = O x A
    o = np.cross(a, n)  # (a)];
    new_rot = np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)

    return new_rot


def is_rotation(rot: np.ndarray, tol=100):
    """Test if matrix is a proper rotation matrix

    Taken from
    https://petercorke.github.io/spatialmath-python/func_nd.html#spatialmath.base.transformsNd.isR

    Parameters
    ----------
    rot : np.ndarray
        3x3 np.array
    """
    _eps = np.finfo(np.float64).eps
    return (
        np.linalg.norm(rot @ rot.T - np.eye(rot.shape[0])) < tol * _eps
        and np.linalg.det(rot @ rot.T) > 0
    )

@dataclass
class AbstractSaver(ABC):
    root:Path
        
    @abstractmethod
    def save_sample(self, sample: DatasetSample) -> None:
        pass



@dataclass
class AbstractReader(ABC):
    root: Path
    

@dataclass
class ImageSaver:
    """ Utility to save images into a folder structure defined by
    `folder_names`.  The dict `folder_names` needs to include values for all the
    constants defined in `ImgDirs` enum.
    """

    root_path: Path
    folder_names: Dict[ImgDirs, str] 

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.get_img_dirs()
        self.create_dir()

    def get_img_dirs(self) -> Dict[ImgDirs, Path]:
        img_dirs = {}
        for img_dir in ImgDirs:
            img_dirs[img_dir] = self.root_path / self.folder_names[img_dir]
        return img_dirs

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True)

    def save_sample(self, str_step: str, data: DatasetSample):
        raw_path = str(self.dir_dict[ImgDirs.RAW] / f"{str_step}.png")
        segmented_path = str(self.dir_dict[ImgDirs.SEGMENTED] / f"{str_step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

        # Save depth
        #save_depth(self.dir_dict[ImgDirs.DEPTH] / f"{str_step}.png", data.depth_img)

        # Generating gt_vis reduces the script performance 

        # data.generate_gt_vis()
        # blended_path = str(self.dir_dict[ImgDirs.GT_VISUALIZATION] / f"{str_step}.png")
        # cv2.imwrite(blended_path, data.blended_img)

# 1. Remove 'import png' from the top
# 2. Keep 'import cv2' and 'import numpy as np'

def save_depth(path: Path, im: np.ndarray):
    """
    Saves a depth image (16-bit) to a PNG file using OpenCV.
    Maintains 100% compatibility with the original depth data.
    """
    path_str = str(path)
    if not path_str.lower().endswith('.png'):
        raise ValueError('Only PNG format is currently supported.')

    # Round and cast to 16-bit unsigned integer (0-65535)
    # This ensures your depth values (e.g., 1000mm) stay accurate.
    im_uint16 = np.round(im).astype(np.uint16)

    # cv2.imwrite automatically detects uint16 and saves a 16-bit PNG
    success = cv2.imwrite(path_str, im_uint16)
    
    if not success:
        raise IOError(f"Failed to save depth image to {path_str}")'''

'''def save_depth(path:Path, im:np.ndarray):
  """Saves a depth image (16-bit) to a PNG file.

  :param path: Path to the output depth image file.
  :param im: ndarray with the depth image to save.
  """
  path = str(path)
  if path.split('.')[-1].lower() != 'png':
    raise ValueError('Only PNG format is currently supported.')

  im_uint16 = np.round(im).astype(np.uint16)

  # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
  w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
  with open(path, 'wb') as f:
    w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))'''