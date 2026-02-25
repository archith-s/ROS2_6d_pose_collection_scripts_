from pathlib import Path
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
    dataset_path = file_path / "../../dataset"
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
    main()
