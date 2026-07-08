# collect_data.py
# python ROS2_6d_pose_collection_scripts_/scripts/collect_data.py --path ./dataset --scene_id 1

from pathlib import Path
import sys
import time
import signal
import click

import rclpy
from rclpy import ok as ros_ok

from ambf6dpose import (
    SimulatorDataProcessor,
    AbstractSimulationClient,
    SyncRosInterface,
)
from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import YamlSampleSaver
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver, MultiCameraSaver
from ambf6dpose.DataCollection.BOPSaver.BopSaver import BopSampleSaver


# ───────────────────────── signal handler ──────────────────────────

def signal_handler(sig, frame):
    """
    Signals the ROS2 context to shut down.
    This will cause 'ros_ok()' to return False, breaking the while loop.
    """
    print("\n[SIGINT] Stop requested. Finalizing dataset...")
    if rclpy.ok():
        rclpy.shutdown()


# ───────────────────────── helper factories ────────────────────────

def create_multi_camera_saver(
    root: Path,
    saver_type: str,
    scene_id: int,
    camera_names: list,
) -> MultiCameraSaver:
    """
    Build a MultiCameraSaver that creates one sub-dataset per camera.

    Directory layout:
        root/
            camera_l/    ← full BOP or YAML dataset for camera_l
            camera_0/    ← full BOP or YAML dataset for camera_0
            camera_1/    ← full BOP or YAML dataset for camera_1

    Parameters
    ----------
    root : Path
        Top-level output directory supplied on the CLI.
    saver_type : str
        "bop" or "yaml".
    scene_id : int
        Passed through to BopSampleSaver (ignored by YamlSampleSaver).
    camera_names : list[str]
        Ordered camera names from SimulatorDataProcessor.camera_registry.
    """
    if saver_type == "bop":
        def factory(camera_root: Path) -> AbstractSaver:
            return BopSampleSaver(camera_root, scene_id=scene_id)
    elif saver_type == "yaml":
        def factory(camera_root: Path) -> AbstractSaver:
            return YamlSampleSaver(camera_root)
    else:
        raise ValueError(f"Unknown sample saver type: {saver_type!r}")

    return MultiCameraSaver(
        root=root,
        camera_names=camera_names,
        saver_factory=factory,
    )


def wait_for_data(client: AbstractSimulationClient):
    try:
        print('here1')
        client.wait_for_data()
        print('here2')
    except TimeoutError:
        print(
            "ERROR: Timeout exception triggered. ROS message filter did not receive any data.",
            file=sys.stderr,
        )
        sys.exit(1)


# ————————————————————————— collection loop —————————————————————————

def start_collection(
    samples_generator: SimulatorDataProcessor,
    saver: MultiCameraSaver,
    sample_time: float,
):
    last_time = time.time()
    count = 0

    # The 'with' block ensures that even if an error occurs,
    # each child saver's __exit__ is called to flush JSON metadata.
    with saver:
        print("Starting collection loop. Press Ctrl+C to save and exit.")
        while ros_ok():
            current_time = time.time()
            if current_time - last_time > sample_time:
                wait_for_data(samples_generator.simulation_client)
                print('here')

                if not ros_ok():
                    break

                # generate_dataset_sample() returns one DatasetSample per
                # camera, all derived from the same synchronised snapshot.
                samples = samples_generator.generate_dataset_sample()

                # Delegate each sample to the appropriate per-camera saver.
                saver.save_samples(samples)

                print(
                    f" Saved Sample: {count} | "
                    f"Interval: {current_time - last_time:0.3f}s"
                )

                last_time = time.time()
                count += 1
            else:
                time.sleep(0.01)

        print("\nExited loop. Finalizing files on disk...")


# ————————————————————————— Click CLI ———————————————————————————————

@click.command()
@click.option("--path",        required=True,  help="Path to save dataset")
@click.option("--scene_id",    required=True,  help="scene_id")
@click.option("--sample_time", default=0.5,    help="Sample every n seconds")
@click.option(
    "--saver_type",
    default="bop",
    show_default=True,
    type=click.Choice(["bop", "yaml"], case_sensitive=False),
    help="Dataset format to write",
)
def collect_data(path: str, scene_id: int, sample_time: float, saver_type: str) -> None:
    signal.signal(signal.SIGINT, signal_handler)

    path     = Path(path).resolve()
    scene_id = int(scene_id)

    # SyncRosInterface calls rclpy.init() internally
    client = SyncRosInterface()
    print(f"Connected to ROS2: {client}")

    samples_generator = SimulatorDataProcessor(client)

    # Derive the ordered camera names directly from the registry so the
    # saver order always stays in sync with generate_dataset_sample().
    camera_names = [cfg.name for cfg in samples_generator.camera_registry]
    print(f"Cameras registered for collection: {camera_names}")

    saver = create_multi_camera_saver(path, saver_type, scene_id, camera_names)

    start_collection(samples_generator, saver, sample_time)

    print(f"Collection finished. Dataset saved to: {path}")

'''# collect_data.py
# python ROS2_6d_pose_collection_scripts_/scripts/collect_data.py --path ./ROS2_6d_pose_collection_scripts_/dataset --scene_id 1

from pathlib import Path
import sys
import time
import signal
import click

import rclpy
from rclpy.node import Node
from rclpy import ok as ros_ok          # keep call-site name short

from ambf6dpose import (
    SimulatorDataProcessor,
    AbstractSimulationClient,
    SyncRosInterface,
)
from ambf6dpose.DataCollection.CustomYamlSaver.YamlSaver import YamlSampleSaver
from ambf6dpose.DataCollection.ReaderSaverUtils import AbstractSaver
from ambf6dpose.DataCollection.BOPSaver.BopSaver import BopSampleSaver


# ───────────────────────── signal handler ──────────────────────────


''''''def signal_handler(sig, frame):
    print("\nClosing collection script")
    time.sleep(10)
    if rclpy.ok():  # Only shutdown if still active
        rclpy.shutdown()''''''
                            # replaces rospy.signal_shutdown


# ───────────────────────── helper factories ────────────────────────
def create_sample_saver(root: Path, type: str, scene_id: int) -> AbstractSaver:
    if type == "bop":
        return BopSampleSaver(root, scene_id=scene_id)
    elif type == "yaml":
        return YamlSampleSaver(root)
    else:
        raise ValueError(f"Unknown sample saver {type}")


def wait_for_data(client: AbstractSimulationClient):
    try:
        print('here1')
        client.wait_for_data()
        print('here2')
    except TimeoutError:
        print(
            "ERROR: Timeout exception triggered. ROS message filter did not receive any data.",
            file=sys.stderr,
        )
        sys.exit(1)

# ————————————————————————— signal handler ——————————————————————————
def signal_handler(sig, frame):
    """
    Signals the ROS2 context to shut down. 
    This will cause 'ros_ok()' to return False, breaking the while loop.
    """
    print("\n[SIGINT] Stop requested. Finalizing dataset...")
    if rclpy.ok():
        rclpy.shutdown()

# ————————————————————————— collection loop —————————————————————————



def start_collection(
    samples_generator: SimulatorDataProcessor,
    saver: AbstractSaver,
    sample_time: float,
):
    last_time = time.time()
    count = 0
    
    # The 'with' block ensures that even if an error occurs, 
    # the saver's __exit__ method is called to write the JSON metadata.
    with saver:
        print("Starting collection loop. Press Ctrl+C to save and exit.")
        while ros_ok():
            current_time = time.time()
            if current_time - last_time > sample_time:
                # This call waits for fresh ROS data
                wait_for_data(samples_generator.simulation_client)
                print('here')
                # Double-check ROS status after waiting
                if not ros_ok():
                    break

                sample = samples_generator.generate_dataset_sample()
                saver.save_sample(sample)
                
                print(
                    f" Saved Sample: {count} | "
                    f"Interval: {current_time - last_time:0.3f}s"
                )
                
                last_time = time.time()
                count += 1
            else:
                # Prevents the script from hogging 100% CPU while waiting
                time.sleep(0.01)
        
        print("\nExited loop. Finalizing files on disk...")

# ————————————————————————— Click CLI ———————————————————————————————
@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--scene_id", required=True, help="scene_id")
@click.option("--sample_time", default=0.5, help="Sample every n seconds")
def collect_data(path: str, scene_id: int, sample_time: float) -> None:
    # Set up SIGINT handler
    signal.signal(signal.SIGINT, signal_handler)
    
    path = Path(path).resolve()
    scene_id = int(scene_id)
    
    # SyncRosInterface calls rclpy.init() internally
    client = SyncRosInterface()
    print(f"Connected to ROS2: {client}")
    
    samples_generator = SimulatorDataProcessor(client)
    saver = create_sample_saver(path, "bop", scene_id)
    
    # This will now exit cleanly when rclpy.shutdown() is called in the handler
    start_collection(samples_generator, saver, sample_time)
    
    print(f"Collection finished. Dataset saved to: {path}")'''


# ───────────────────────── collection loop ─────────────────────────
'''def start_collection(
    samples_generator: SimulatorDataProcessor,
    saver: AbstractSaver,
    sample_time: float,
):
    last_time = time.time() + sample_time
    count = 0
    with saver:
        while ros_ok():
            if time.time() - last_time > sample_time:
                wait_for_data(samples_generator.simulation_client)
                # SIGINT can occur while waiting for data
                print('Out of wait_for_data loop')
                if ros_ok():
                    sample = samples_generator.generate_dataset_sample()
                    saver.save_sample(sample)
                    print(
                        f" Sample: {count}  "
                        f"Time from last sample: {time.time() - last_time:0.3f}s"
                    )
                    last_time = time.time()
                    count += 1


# ───────────────────────── Click CLI ───────────────────────────────
@click.command()
@click.option("--path", required=True, help="Path to save dataset")
@click.option("--scene_id", required=True, help="scene_id")
@click.option("--sample_time", default=0.5, help="Sample every n seconds")
def collect_data(path: str, scene_id: int, sample_time: float) -> None:
    """
    6-D pose data-collection script.
    Instructions: (1) run AMBF simulation  (2) run recorded motions
    (3) run this collection script.
    """

    # Set up SIGINT handler only
    signal.signal(signal.SIGINT, signal_handler)
    path = Path(path).resolve()
    scene_id = int(scene_id)
    client = SyncRosInterface()                            # rclpy.init() already called inside
    print(client)
    samples_generator = SimulatorDataProcessor(client)
    print(samples_generator)
    saver = create_sample_saver(path, "bop", scene_id)
    start_collection(samples_generator, saver, sample_time)
    # Normal shutdown if the loop exits
    print('Out of start collection loop')
    rclpy.shutdown()'''

if __name__ == "__main__":
    collect_data()

