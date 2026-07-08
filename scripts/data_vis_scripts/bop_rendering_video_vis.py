from pathlib import Path
from typing import Dict, List
import math
import numpy as np
import cv2

from ambf6dpose.DataCollection.BOPSaver.BopReader import BopDatasetReader
from ambf6dpose.DataCollection.DatasetSample import DatasetSample, RigidObjectsIds
from ambf6dpose.DataVisualization.bop_vis_utils import (
    BOPRendererWrapper,
    ImageAnnotations,
)


# --------------------------------------------------------------------------- #
# Renderer setup
# --------------------------------------------------------------------------- #

def setup_rendering() -> BOPRendererWrapper:
    file_path = Path(__file__).resolve().parent
    toolpitchlink_model_path = (file_path / "../../SampleData1/Models/ToolPitchLink.ply").resolve()
    assert toolpitchlink_model_path.exists(), f"Model not found: {toolpitchlink_model_path}"

    my_renderer = BOPRendererWrapper()
    my_renderer.add_object(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        toolpitchlink_model_path,
        [0.0, 0.0, 0.8],
    )
    return my_renderer


# --------------------------------------------------------------------------- #
# Per-camera panel: pitch link overlay only
# --------------------------------------------------------------------------- #

def build_pitchlink_panel(my_renderer: BOPRendererWrapper, sample: DatasetSample) -> np.ndarray:
    """
    Renders the psm1 and psm2 pitch link overlays onto the sample's raw image
    and returns the annotated panel.
    """
    text_size   = 20
    text_offset = (2, -25)

    ren_psm1 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        sample.psm1_toolpitchlink_pose,
        sample,
    )
    ren_psm2 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        sample.psm2_toolpitchlink_pose,
        sample,
    )

    annotator = ImageAnnotations(sample.raw_img, text_size, text_offset)
    annotator.add_annotations("psm1_toolpitch", ren_psm1)
    annotator.add_annotations("psm2_toolpitch", ren_psm2)
    return annotator.combine_annotations()


# --------------------------------------------------------------------------- #
# Camera label overlay
# --------------------------------------------------------------------------- #

def add_camera_label(panel: np.ndarray, label: str) -> np.ndarray:
    out       = panel.copy()
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 1.2
    thickness = 2
    origin    = (12, 40)

    # Drop-shadow for readability over any background
    cv2.putText(out, label, (origin[0] + 2, origin[1] + 2), font, scale, (0, 0, 0),       thickness + 2, cv2.LINE_AA)
    cv2.putText(out, label, origin,                          font, scale, (255, 255, 255), thickness,     cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Grid assembly
# --------------------------------------------------------------------------- #

def build_grid(panels: List[np.ndarray]) -> np.ndarray:
    """
    Arrange panels into the tightest square-ish grid.
    Blank cells (if N does not fill the grid perfectly) are black.

        1 camera  → 1x1
        2 cameras → 1x2
        3 cameras → 2x2  (one black cell)
        4 cameras → 2x2
    """
    n      = len(panels)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    h, w   = panels[0].shape[:2]
    blank  = np.zeros((h, w, 3), dtype=np.uint8)
    padded = panels + [blank] * (n_rows * n_cols - n)

    rows = [
        np.hstack(padded[r * n_cols : (r + 1) * n_cols])
        for r in range(n_rows)
    ]
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    file_path    = Path(__file__).resolve().parent
    dataset_path = (file_path / "../../dataset2").resolve()
    assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"

    # Discover cameras by iterating subdirectories — no hardcoding needed.
    # Each subdirectory is one camera's dataset (e.g. camera_l/, camera_0/, camera_1/).
    cameras = sorted(p.name for p in dataset_path.iterdir() if p.is_dir())
    if not cameras:
        raise RuntimeError(f"No camera subdirectories found in {dataset_path}")
    print(f"Discovered cameras: {cameras}")

    # One reader per camera pointed directly at its subfolder
    readers: Dict[str, BopDatasetReader] = {}
    for cam in cameras:
        readers[cam] = BopDatasetReader(
            root=dataset_path / cam,
            scene_id_list=[],
            dataset_split="test",
            dataset_split_type="",
        )
        print(f"  {cam}: {len(readers[cam])} samples")

    # Iterate to the shortest camera dataset in case of any mismatch
    n_samples = min(len(readers[c]) for c in cameras)
    print(f"Rendering {n_samples} samples across {len(cameras)} camera(s).")

    # One shared renderer — the pitch link model is camera-agnostic
    my_renderer  = setup_rendering()
    video_path   = "pitchlink_multicam.mp4"
    fps          = 4
    video_writer = None

    for idx in range(n_samples):
        panels = []
        skip   = False

        for cam in cameras:
            try:
                sample = readers[cam][idx]
            except Exception as e:
                print(f"Skipping index {idx} ({cam}): {e}")
                skip = True
                break

            panel = build_pitchlink_panel(my_renderer, sample)
            panel = add_camera_label(panel, cam)
            panels.append(panel)

        if skip:
            continue

        grid = build_grid(panels)

        if video_writer is None:
            h, w   = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        video_writer.write(grid)

    if video_writer is not None:
        video_writer.release()
        print(f"Video written to {video_path}")

        import subprocess
        try:
            subprocess.run(["ffplay", "-autoexit", "-loop", "0", video_path], check=True)
        except FileNotFoundError:
            print(f"ffplay not found. Play the video manually at: {video_path}")
        except (subprocess.CalledProcessError, KeyboardInterrupt):
            pass
    else:
        print("No frames were rendered.")


if __name__ == "__main__":
    main()

'''from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2

from ambf6dpose.DataCollection.BOPSaver.BopReader import BopDatasetReader
from ambf6dpose.DataCollection.DatasetSample import DatasetSample, RigidObjectsIds
from ambf6dpose.DataVisualization.bop_vis_utils import (
    BOPRendererWrapper,
    ImageAnnotations,
)


def assert_paths_exist(paths: List[Path]):
    for p in paths:
        assert p.exists(), f"Path {p} does not exist"


def setup_rendering() -> BOPRendererWrapper:
    file_path = Path(__file__).resolve().parent
    needle_model_path = file_path / "../../SampleData1/Models/Needle.ply"
    needle_model_path = needle_model_path.resolve()
    toolpitchlink_model_path = file_path / "../../SampleData1/Models/ToolPitchLink.ply"
    toolpitchlink_model_path = toolpitchlink_model_path.resolve()
    toolyawlink_model_path = file_path / "../../SampleData1/Models/ToolYawLink.ply"
    toolyawlink_model_path = toolyawlink_model_path.resolve()

    assert_paths_exist(
        [needle_model_path, toolpitchlink_model_path, toolyawlink_model_path]
    )

    my_renderer = BOPRendererWrapper()
    # NEEDLE
    my_renderer.add_object(
        RigidObjectsIds.needle_pose, needle_model_path, [0.0, 0.8, 0.0]
    )
    # TOOLPITCH
    my_renderer.add_object(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        toolpitchlink_model_path,
        [0.0, 0.0, 0.8],
    )
    # TOOLYAW
    my_renderer.add_object(
        RigidObjectsIds.psm1_toolyawlink_pose,
        toolyawlink_model_path,
        [0.0, 0.8, 0.8],
    )

    return my_renderer


def annotate_img(my_renderer, sample: DatasetSample):
    text_size = 20
    text_offset = (2, -25)
    ## ANNOTATE NEEDLE
    ren_out1 = my_renderer.render_obj(
        RigidObjectsIds.needle_pose, sample.needle_pose, sample
    )
    annotator1 = ImageAnnotations(sample.raw_img, text_size, text_offset)
    annotator1.add_annotations("needle", ren_out1)
    annotated_img1 = annotator1.combine_annotations()

    # ANNOTATE TOOLPITCHLINK
    ren_out2_psm1 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        sample.psm1_toolpitchlink_pose,
        sample,
    )
    ren_out2_psm2 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolpitchlink_pose,
        sample.psm2_toolpitchlink_pose,
        sample,
    )
    annotator2 = ImageAnnotations(sample.raw_img, text_size, text_offset)
    annotator2.add_annotations("psm1_toolpitch", ren_out2_psm1)
    annotator2.add_annotations("psm2_toolpitch", ren_out2_psm2)
    annotated_img2 = annotator2.combine_annotations()

    # ANNOTATE TOOLYAWLINK
    ren_out3_psm1 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolyawlink_pose,
        sample.psm1_toolyawlink_pose,
        sample,
    )
    ren_out3_psm2 = my_renderer.render_obj(
        RigidObjectsIds.psm1_toolyawlink_pose,
        sample.psm2_toolyawlink_pose,
        sample,
    )
    annotator3 = ImageAnnotations(sample.raw_img, text_size, text_offset)
    annotator3.add_annotations("psm1_toolyaw", ren_out3_psm1)
    annotator3.add_annotations("psm2_toolyaw", ren_out3_psm2)
    annotated_img3 = annotator3.combine_annotations()

    return annotated_img1, annotated_img2, annotated_img3

def main():
    file_path = Path(__file__).resolve().parent
    dataset_path = file_path / "../../dataset1"
    dataset_path = dataset_path.resolve()
    assert dataset_path.exists(), f"Path {dataset_path} does not exist"

    reader = BopDatasetReader(
        root=Path(dataset_path),
        scene_id_list=[],
        dataset_split="test",
        dataset_split_type="",
    )

    print(f"Dataset size: {len(reader)}")

    # >>> VIDEO ADDITION (ONLY NEW LOGIC)
    video_path = "overlay_visualization.mp4"
    fps = 4
    video_writer = None
    # <<< VIDEO ADDITION

    # Use index-based iteration to handle errors gracefully
    for idx in range(len(reader)):
        try:
            sample = reader[idx]
        except KeyError as e:
            print(f"Skipping sample {idx} due to KeyError: {e}")
            continue
        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")
            continue
            
        my_renderer = setup_rendering()
        annotated_img1, annotated_img2, annotated_img3 = annotate_img(
            my_renderer, sample
        )

        final1 = np.hstack((annotated_img1, annotated_img2))
        final2 = np.hstack((annotated_img3, sample.segmented_img))
        final = np.vstack((final1, final2))

        # >>> VIDEO ADDITION
        if video_writer is None:
            h, w, _ = final.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                video_path, fourcc, fps, (w, h)
            )

        video_writer.write(final)
        # <<< VIDEO ADDITION

    # >>> VIDEO ADDITION
    if video_writer is not None:
        video_writer.release()
        print(f"Video written to {video_path}")
        
        # Play the video using ffmpeg
        import subprocess
        print(f"Playing video with ffmpeg...")
        try:
            subprocess.run(
                ["ffplay", "-autoexit", "-loop", "0", video_path],
                check=True
            )
        except FileNotFoundError:
            print("ffplay not found. Install ffmpeg to play the video.")
            print(f"You can manually play the video at: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error playing video: {e}")
        except KeyboardInterrupt:
            print("\nVideo playback interrupted by user.")
    # <<< VIDEO ADDITION


if __name__ == "__main__":
    main()'''
