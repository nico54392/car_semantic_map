// RGBDPublisher.cs: makes a Unity camera look like a RealSense on ROS, so tools/semantic_mapper.py
// (and anything else) can't tell sim from the real robot.
// Put it on the ROBOT's camera (it only reads the camera, never moves it). Needs ROS-TCP-Connector.
//
// Publishes (defaults = realsense2_camera-style names, change in the Inspector):
//   /camera/color/image_raw       sensor_msgs/Image rgb8
//   /camera/depth/image_raw       sensor_msgs/Image 32FC1, metres, z-depth, ALIGNED to the colour image
//                                 (same size/intrinsics, like align_depth on a RealSense); 0 = no hit
//   /camera/color/camera_info     sensor_msgs/CameraInfo
//   /camera_pose_unity            geometry_msgs/PoseStamped: camera pose in the map frame (ROS FLU),
//                                 OpenCV optical axes (x right, y down, z forward). Use the mapper with
//                                 pose_source:=topic, no TF needed.
// Same conventions as SceneCapture.cs (Unity (x,y,z) -> ROS (z,-x,y), fx = fy from the vertical FOV).
// Depth comes from raycasts, so everything that should appear needs a collider.
// Physical Camera must be OFF (lens shift would break the intrinsics).
using System;
using Unity.Collections;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class RGBDPublisher : MonoBehaviour
{
    public string rgbTopic = "/camera/color/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    public string infoTopic = "/camera/color/camera_info";
    public string poseTopic = "/camera_pose_unity";
    public string frameId = "camera_color_optical_frame";
    public string mapFrame = "map";
    public int width = 320;
    public int height = 240;
    [Tooltip("Hz. The mapper only keeps keyframes anyway, 2-5 Hz is plenty")]
    public float rate = 3f;
    public float maxDistance = 20f;
    public LayerMask raycastMask = ~0;

    Camera cam;
    ROSConnection ros;
    float next;
    RenderTexture rt;
    Texture2D tex;
    byte[] rgbBuf, depthBuf;

    void Start()
    {
        cam = GetComponent<Camera>();
        if (cam.usePhysicalProperties)
            Debug.LogWarning("RGBDPublisher: turn Physical Camera off, the intrinsics would be wrong");
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>(rgbTopic);
        ros.RegisterPublisher<ImageMsg>(depthTopic);
        ros.RegisterPublisher<CameraInfoMsg>(infoTopic);
        ros.RegisterPublisher<PoseStampedMsg>(poseTopic);
        rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        rgbBuf = new byte[width * height * 3];
        depthBuf = new byte[width * height * 4];
    }

    void OnDestroy()
    {
        if (rt) rt.Release();
    }

    void LateUpdate()
    {
        if (Time.time < next) return;
        next = Time.time + 1f / Mathf.Max(rate, 0.1f);
        var header = Header();
        PublishRGB(header);
        PublishDepth(header);
        PublishInfo(header);
        PublishPose();
    }

    HeaderMsg Header()
    {
        // wall clock, so the stamps are comparable with the ROS side on the same machine
        long ns = (DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()) * 1000000L;
        return new HeaderMsg { stamp = new TimeMsg { sec = (int)(ns / 1000000000L), nanosec = (uint)(ns % 1000000000L) },
                               frame_id = frameId };
    }

    void PublishRGB(HeaderMsg header)
    {
        var prev = cam.targetTexture;
        cam.targetTexture = rt;
        cam.aspect = (float)width / height;
        cam.Render();
        var prevActive = RenderTexture.active;
        RenderTexture.active = rt;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();
        RenderTexture.active = prevActive;
        cam.targetTexture = prev;
        cam.ResetAspect();
        var raw = tex.GetRawTextureData();              // rows bottom-up
        int row = width * 3;
        for (int v = 0; v < height; v++)
            Buffer.BlockCopy(raw, (height - 1 - v) * row, rgbBuf, v * row, row);
        ros.Publish(rgbTopic, new ImageMsg { header = header, height = (uint)height, width = (uint)width,
                                             encoding = "rgb8", is_bigendian = 0, step = (uint)row, data = rgbBuf });
    }

    void PublishDepth(HeaderMsg header)
    {
        int n = width * height;
        var cmds = new NativeArray<RaycastCommand>(n, Allocator.TempJob);
        var hits = new NativeArray<RaycastHit>(n, Allocator.TempJob);
        var qp = new QueryParameters(raycastMask, false, QueryTriggerInteraction.Ignore, false);
        float aspect = cam.aspect;
        cam.aspect = (float)width / height;
        for (int v = 0; v < height; v++)
            for (int u = 0; u < width; u++)
            {
                Ray r = cam.ViewportPointToRay(new Vector3((u + 0.5f) / width, 1f - (v + 0.5f) / height, 0f));
                cmds[v * width + u] = new RaycastCommand(r.origin, r.direction, qp, maxDistance);
            }
        cam.aspect = aspect;
        RaycastCommand.ScheduleBatch(cmds, hits, 256, 1).Complete();
        var depth = new float[n];
        Vector3 p0 = cam.transform.position, fwd = cam.transform.forward;
        for (int i = 0; i < n; i++)
            depth[i] = hits[i].collider != null ? Vector3.Dot(hits[i].point - p0, fwd) : 0f;
        cmds.Dispose(); hits.Dispose();
        Buffer.BlockCopy(depth, 0, depthBuf, 0, depthBuf.Length);
        ros.Publish(depthTopic, new ImageMsg { header = header, height = (uint)height, width = (uint)width,
                                               encoding = "32FC1", is_bigendian = 0, step = (uint)(width * 4), data = depthBuf });
    }

    void PublishInfo(HeaderMsg header)
    {
        double fy = height * 0.5 / Math.Tan(cam.fieldOfView * 0.5 * Math.PI / 180.0), fx = fy;
        double cx = (width - 1) * 0.5, cy = (height - 1) * 0.5;
        ros.Publish(infoTopic, new CameraInfoMsg {
            header = header, height = (uint)height, width = (uint)width, distortion_model = "plumb_bob",
            d = new double[5],
            k = new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 },
            r = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
            p = new double[] { fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 } });
    }

    static Vector3 U2R(Vector3 v) => new Vector3(v.z, -v.x, v.y);

    void PublishPose()
    {
        // OpenCV camera axes in the ROS map frame = the columns of the rotation
        Vector3 c0 = U2R(cam.transform.right), c1 = U2R(-cam.transform.up), c2 = U2R(cam.transform.forward);
        Vector3 t = U2R(cam.transform.position);
        float m00 = c0.x, m01 = c1.x, m02 = c2.x, m10 = c0.y, m11 = c1.y, m12 = c2.y, m20 = c0.z, m21 = c1.z, m22 = c2.z;
        float tr = m00 + m11 + m22, qw, qx, qy, qz;
        if (tr > 0) { float s = Mathf.Sqrt(tr + 1f) * 2; qw = 0.25f * s; qx = (m21 - m12) / s; qy = (m02 - m20) / s; qz = (m10 - m01) / s; }
        else if (m00 > m11 && m00 > m22) { float s = Mathf.Sqrt(1f + m00 - m11 - m22) * 2; qw = (m21 - m12) / s; qx = 0.25f * s; qy = (m01 + m10) / s; qz = (m02 + m20) / s; }
        else if (m11 > m22) { float s = Mathf.Sqrt(1f + m11 - m00 - m22) * 2; qw = (m02 - m20) / s; qx = (m01 + m10) / s; qy = 0.25f * s; qz = (m12 + m21) / s; }
        else { float s = Mathf.Sqrt(1f + m22 - m00 - m11) * 2; qw = (m10 - m01) / s; qx = (m02 + m20) / s; qy = (m12 + m21) / s; qz = 0.25f * s; }
        var h = Header(); h.frame_id = mapFrame;
        ros.Publish(poseTopic, new PoseStampedMsg { header = h, pose = new PoseMsg {
            position = new PointMsg(t.x, t.y, t.z), orientation = new QuaternionMsg(qx, qy, qz, qw) } });
    }
}
