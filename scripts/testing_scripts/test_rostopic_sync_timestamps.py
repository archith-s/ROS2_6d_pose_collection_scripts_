import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from ambf_msgs.msg import RigidBodyState, CameraState

class TimestampDebugger(Node):
    def __init__(self):
        super().__init__('timestamp_debugger')

        self.subs = {
            "CAMERA_L_STATE":      ("/ambf/env/cameras/cameraL/State", CameraState),
            "CAMERA_FRAME":        ("/ambf/env/phantom/CameraFrame/State", RigidBodyState),
            "NEEDLE":              ("/ambf/env/phantom/Needle/State", RigidBodyState),
            "CAMERA_L_IMAGE":      ("/ambf/env/cameras/cameraL/ImageData", Image),
            "CAMERA_L_SEG_IMAGE":  ("/ambf/env/cameras/cameraL2/ImageData", Image),
            "CAMERA_L_DEPTH":      ("/ambf/env/cameras/cameraL/DepthData", PointCloud2),
            "PSM1_PITCH":          ("/ambf/env/psm1/toolpitchlink/State", RigidBodyState),
            "PSM2_PITCH":          ("/ambf/env/psm2/toolpitchlink/State", RigidBodyState),
            "PSM1_YAW":            ("/ambf/env/psm1/toolyawlink/State", RigidBodyState),
            "PSM2_YAW":            ("/ambf/env/psm2/toolyawlink/State", RigidBodyState),
        }

        for label, (topic, msg_type) in self.subs.items():
            self.create_subscription(msg_type, topic, self.make_cb(label), 10)

    def make_cb(self, label):
        def cb(msg):
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                sec = msg.header.stamp.sec
                nsec = msg.header.stamp.nanosec
                t = sec + nsec * 1e-9
                self.get_logger().info(f"[{label}] {t:.6f}")
            else:
                self.get_logger().error(f"[{label}] ❌ No header.stamp")
        return cb

def main():
    rclpy.init()
    node = TimestampDebugger()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
