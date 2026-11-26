#!/usr/bin/env python3
"""
Unity -> Nav2 glue. Unity gives a perfect pose on /odom_unity (already in the map frame = ROS FLU
of the capture), so there's no AMCL/SLAM: map->odom is identity and odom->base_link is that pose.

Publishes
  /tf          odom -> base_link   (from /odom_unity, restamped with this machine's clock)
  /tf_static   map -> odom         (identity)
  /odom        nav_msgs/Odometry, frame odom / child base_link (Nav2's odom_topic)

python3 tools/unity_nav2_bridge.py
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


class Bridge(Node):
    def __init__(self):
        super().__init__("unity_nav2_bridge")
        self.tf = TransformBroadcaster(self)
        st = TransformStamped()
        st.header.stamp = self.get_clock().now().to_msg()
        st.header.frame_id, st.child_frame_id = "map", "odom"
        st.transform.rotation.w = 1.0
        self.static = StaticTransformBroadcaster(self); self.static.sendTransform(st)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(Odometry, "/odom_unity", self.on_odom, qos_profile_sensor_data)
        self.n = 0
        self.get_logger().info("waiting for /odom_unity ...")

    def on_odom(self, msg):
        now = self.get_clock().now().to_msg()
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id, t.child_frame_id = "odom", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, 0.0
        t.transform.rotation = q
        self.tf.sendTransform(t)
        o = Odometry()
        o.header.stamp = now
        o.header.frame_id, o.child_frame_id = "odom", "base_link"
        o.pose.pose = msg.pose.pose; o.pose.pose.position.z = 0.0
        o.twist = msg.twist
        self.odom_pub.publish(o)
        self.n += 1
        if self.n == 1:
            self.get_logger().info(f"first pose x={p.x:.2f} y={p.y:.2f}: publishing TF map->odom->base_link")


def main():
    rclpy.init()
    n = Bridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
