#python3 simple_replay.py --bag_path /home/user/my_test_bag
#ros2 bag record /psm1/setpoint_js /psm1/jaw/setpoint_js /psm2/setpoint_js /psm2/jaw/setpoint_js /ecm/setpoint_js -o my_test_bag
from dataclasses import dataclass
import signal
import time
import cv2
import click
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

import ambf6dpose.RosBagReplay.RosbagUtils as rosbagutils
from ambf6dpose.DataCollection.Rostopics import _make_image_processor, RosTopics

default_path = "/home/juan1995/research/accelnet_grant/6d_pose_collection_scripts/test_replay/src_env2_v1.1.3_rec03_jack.bag"

class ROS2ImageSubscriber(Node):
    def __init__(self, output_path: Path):
        super().__init__('simple_image_subscriber')
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.img_processor = _make_image_processor()
        self.count = 0
        self.saving_mode = False

        topic, msg_type = RosTopics.CAMERA_L_IMAGE.value
        self.subscriber = self.create_subscription(
            msg_type,
            topic,
            self.topic_cb,
            10
        )

    def start_saving(self):
        self.saving_mode = True

    def stop_saving(self):
        self.saving_mode = False

    def save_frame(self, img):
        out_path = str(self.output_path / f"frame_{self.count:05d}.png")
        cv2.imwrite(out_path, img)
        self.count += 1

    def topic_cb(self, msg):
        img = self.img_processor(msg)
        if self.saving_mode:
            self.save_frame(img)

@dataclass
class SimpleImgSubs:
    output_path: Path

    def __post_init__(self):
        assert self.output_path.exists()
        self.node = ROS2ImageSubscriber(self.output_path)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def start_saving(self):
        self.node.start_saving()

    def stop_saving(self):
        self.node.stop_saving()

    def __enter__(self):
        self.start_saving()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_saving()
        self.executor.shutdown()
        self.node.destroy_node()

def setup_sigint_handler(bag_player: rosbagutils.RosbagReplayer):
    def signal_handler(sig, frame):
        bag_player.run = False
        print("\nClosing player")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)

class FloatListParamType(click.ParamType):
    name = "float_list"

    def __init__(self, sep=" ", ignore_empty=False):
        self.sep = sep
        self.ignore_empty = ignore_empty

    def convert(self, value, param, ctx):
        try:
            if self.ignore_empty and not value:
                return []
            return [float(x) for x in value.split(self.sep)]
        except Exception:
            self.fail(f"{value!r} is not a valid list of floats", param, ctx)

@click.command()
@click.option("--bag_path", default=default_path, help="Path to bag file", type=click.Path(exists=True))
@click.option("--percent_to_replay", default=1.0, help="Percentage of motion to replay")
@click.option("--ecm_pos", default="", help="ECM joint position as 4 floats", type=FloatListParamType(" ", ignore_empty=True))
@click.option("-r", "record_im", is_flag=True, default=False, help="Record images")
@click.option("-o", "--output_p", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), help="Required if recording is enabled")
@click.option("--show_images", is_flag=True, default=False, help="Show CameraL image summary")
@click.option("--show_poses", is_flag=True, default=False, help="Show toolpitch pose summary")
def single_replay(bag_path, percent_to_replay, ecm_pos, record_im, output_p: Path, show_images, show_poses):
    assert len(ecm_pos) == 4 or len(ecm_pos) == 0, "ecm jp needs 4 values"
    if record_im:
        assert output_p is not None, "output_p must be set to record data"
        img_recorder = SimpleImgSubs(output_p)

    rclpy.init()

    bag_path = Path(bag_path)
    bag_player = rosbagutils.RosbagReplayer()
    if len(ecm_pos) == 4:
        bag_player.move_cam(ecm_pos)
    setup_sigint_handler(bag_player)

    # ➜ Read data
    ecm_pos_list, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    # ➜ Access new topics
    image_data_L = rosbagutils.read_rosbag.image_data_L
    toolpitch_poses = rosbagutils.read_rosbag.toolpitch_poses

    if show_images:
        print(f"Number of CameraL images: {len(image_data_L)}")
        for i, img in enumerate(image_data_L[:5]):  # show first 5 as sample
            print(f"Frame {i}: {img.height}x{img.width}, encoding: {img.encoding}")

    if show_poses:
        print(f"Number of toolpitch poses: {len(toolpitch_poses)}")
        for i, pose in enumerate(toolpitch_poses[:5]):  # show first 5 as sample
            print(f"Pose {i}: Position (x,y,z): {pose[:3]}, Orientation (x,y,z,w): {pose[3:]}")

    if record_im:
        with img_recorder:
            bag_player.run_replay(psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay)
    else:
        bag_player.run_replay(psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay)

    input("Press enter to return to home ")
    bag_player.return_psm_to_home()
    bag_player.reset_bodies()

    rclpy.shutdown()

@click.command()
@click.option("--bag_path", default=default_path, help="Path to bag file")
@click.option("--n_replays", default=2, help="Number of replays")
@click.option("--percent_to_replay", default=1.0, help="Percentage of motion to replay")
def loop_replay(bag_path, n_replays, percent_to_replay):
    rclpy.init()
    bag_player = rosbagutils.RosbagReplayer()
    setup_sigint_handler(bag_player)

    ecm_pos_list, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw = rosbagutils.read_rosbag(bag_path)

    try:
        for i in range(n_replays):
            print(f"Replay {i}")
            bag_player.move_psm_to_start(psm1_pos[0], psm2_pos[0])
            bag_player.run_replay(psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay)
            time.sleep(0.3)
            bag_player.return_psm_to_home()
            bag_player.reset_bodies()
    except KeyboardInterrupt:
        print("finishing")
    finally:
        rclpy.shutdown()

@click.group(help="Replay motions from rosbag")
def main():
    pass

if __name__ == "__main__":
    main.add_command(single_replay)
    main.add_command(loop_replay)
    main()

