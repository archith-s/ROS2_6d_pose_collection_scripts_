# collect_data_ros2.py
# python ROS2_6d_pose_collection_scripts_/scripts/collect_data.py --path ./ROS2_6d_pose_collection_scripts_/dataset --scene_id 1

from pathlib import Path
import sys
import time
import signal
import click

import rclpy
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


'''def signal_handler(sig, frame):
    print("\nClosing collection script")
    time.sleep(10)
    if rclpy.ok():  # Only shutdown if still active
        rclpy.shutdown()'''
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
        client.wait_for_data()
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
    
    print(f"Collection finished. Dataset saved to: {path}")


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

