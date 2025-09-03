import time
import sys
from typing import List
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from ambf_msgs.msg import RigidBodyState
from std_msgs.msg import Float64MultiArray, Float64

def read_rosbag(rosbag_name):
    storage_options = StorageOptions(uri=str(rosbag_name), storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_type_map = {t.name: t.type for t in topics_and_types}

    psm1_pos, psm2_pos, psm1_jaw, psm2_jaw, ecm_pos = [], [], [], [], []

    # NEW holders for extra topics
    image_data_L = []
    toolpitch_poses = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic == "/psm1/setpoint_js":
            msg = deserialize_message(data, Float64MultiArray)
            psm1_pos.append(list(msg.data))
        elif topic == "/psm1/jaw/setpoint_js":
            msg = deserialize_message(data, Float64)
            psm1_jaw.append(msg.data)
        elif topic == "/psm2/setpoint_js":
            msg = deserialize_message(data, Float64MultiArray)
            psm2_pos.append(list(msg.data))
        elif topic == "/psm2/jaw/setpoint_js":
            msg = deserialize_message(data, Float64)
            psm2_jaw.append(msg.data)
        elif topic == "/ecm/setpoint_js":
            msg = deserialize_message(data, Float64MultiArray)
            ecm_pos.append(list(msg.data))

        # NEW: cameraL ImageData
        elif topic == "/ambf/env/cameras/cameraL/ImageData":
            msg = deserialize_message(data, Image)
            image_data_L.append(msg)

        # NEW: toolpitchlink State
        elif topic == "/ambf/env/psm1/toolpitchlink/State":
            msg = deserialize_message(data, RigidBodyState)
            position = msg.pose.position
            orientation = msg.pose.orientation
            pose_data = [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w]
            toolpitch_poses.append(pose_data)

    print("psm 1 record count:", len(psm1_pos))
    print("psm 1 jaw record count:", len(psm1_jaw))
    print("psm 2 record count:", len(psm2_pos))
    print("psm 2 jaw record count:", len(psm2_jaw))
    print("ecm record count:", len(ecm_pos))
    print("Camera L image frames count:", len(image_data_L))
    print("Toolpitchlink pose count:", len(toolpitch_poses))

    # Attach extra data to the module for downstream access if needed
    read_rosbag.image_data_L = image_data_L
    read_rosbag.toolpitch_poses = toolpitch_poses

    return ecm_pos, psm1_pos, psm2_pos, psm1_jaw, psm2_jaw

class RosbagReplayer:
    def __init__(self):
        from surgical_robotics_challenge.psm_arm import PSM
        from surgical_robotics_challenge.ecm_arm import ECM
        from surgical_robotics_challenge.simulation_manager import SimulationManager

        self.simulation_manager = SimulationManager("record_test")
        time.sleep(0.2)
        self.w = self.simulation_manager.get_world_handle()
        time.sleep(0.2)
        self.w.reset_bodies()
        time.sleep(0.2)
        self.cam = ECM(self.simulation_manager, "CameraFrame")
        time.sleep(0.2)
        self.psm1 = PSM(self.simulation_manager, "psm1", add_joint_errors=False)
        time.sleep(0.2)
        self.psm2 = PSM(self.simulation_manager, "psm2", add_joint_errors=False)
        time.sleep(0.2)
        self.run = True

    def move_cam(self, ecm_jp: List[float]):
        self.cam.servo_jp(ecm_jp)

    def reset_bodies(self):
        self.w.reset_bodies()

    def run_replay(self, psm1_pos, psm1_jaw, psm2_pos, psm2_jaw, percent_to_replay: float = 1.0):
        assert 0.0 <= percent_to_replay <= 1.0, "percent_to_replay must be between 0 and 1"
        self.run = True
        count = 0
        total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
        total_num = int(total_num * percent_to_replay)

        for i in range(total_num):
            if not self.run:
                break
            self.psm1.servo_jp(psm1_pos[i])
            self.psm1.set_jaw_angle(psm1_jaw[i])
            self.psm2.servo_jp(psm2_pos[i])
            self.psm2.set_jaw_angle(psm2_jaw[i])
            time.sleep(0.01)
            count += 1
            sys.stdout.write(f"\r Run Progress: {count} / {total_num}")
            sys.stdout.flush()

    def stop_replay(self):
        self.run = False

    def move_psm_to_start(self, psm1_pos, psm2_pos):
        self.psm1.move_jp(psm1_pos, execute_time=1.0)
        self.psm2.move_jp(psm2_pos, execute_time=1.0)
        time.sleep(2.0)

    def return_psm_to_home(self):
        self.psm1.set_jaw_angle(0.8)
        self.psm2.set_jaw_angle(0.8)
        time.sleep(2.2)
        self.psm1.move_jp([0.0] * 6, execute_time=0.8)
        self.psm1.set_jaw_angle(0.0)
        self.psm2.move_jp([0.0] * 6, execute_time=0.8)
        self.psm2.set_jaw_angle(0.0)
        time.sleep(2.2)
