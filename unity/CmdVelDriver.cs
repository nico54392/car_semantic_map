using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

// Drives the skid-steer robot from /cmd_vel (geometry_msgs/Twist).
// While commands are coming in it takes over from RobotControls (keyboard),
// and hands control back once it has come to rest after `timeout` s without messages.
//
// Low-level control, in this order every physics step:
//   1. target (v, w) from /cmd_vel, or 0 when the commands go stale
//   2. acceleration limits: the command ramps toward the target (no jumps, smooth stop)
//   3. closedLoop on: PI on the MEASURED forward speed and yaw rate of the robot body,
//      so it turns/drives as asked whatever the tyre slip or terrain. Off: plain open loop
//   4. skid-steer kinematics -> wheel speeds (track width * turnSlipFactor = effective track)
// ROS convention: v forward (m/s), w positive = turn LEFT (rad/s).
// Measurements come from the root ArticulationBody/Rigidbody, else from the transform.
public class CmdVelDriver : MonoBehaviour
{
    [Header("ROS")]
    public string topic = "/cmd_vel";
    public float timeout = 0.5f;          // s without a msg -> ramp to zero, then give keyboard back

    [Header("Wheels")]
    public ArticulationBody frontLeftWheel;
    public ArticulationBody frontRightWheel;
    public ArticulationBody rearLeftWheel;
    public ArticulationBody rearRightWheel;
    public float wheelRadius = 0.079f;    // m, same as RobotControls
    public float trackWidth = 0f;         // m, left-right wheel distance. 0 = measure at start
    public float turnSlipFactor = 1.0f;   // effective track / real track (skid steer scrub). With closedLoop the PI fixes the rest

    [Header("Limits")]
    public float maxAccel = 1.0f;         // m/s^2
    public float maxAngAccel = 3.0f;      // rad/s^2
    public float maxSpeed = 1.0f;         // m/s
    public float maxYawRate = 2.0f;       // rad/s

    [Header("Closed loop (PI on measured v and yaw rate)")]
    public bool closedLoop = true;
    public float kpV = 1.0f, kiV = 1.0f;
    public float kpW = 1.0f, kiW = 2.0f;
    public float maxCorrectionV = 0.5f;   // m/s, clamp on the integral (anti-windup)
    public float maxCorrectionW = 3.0f;   // rad/s
    [Range(0f, 1f)] public float measureFilter = 0.3f;   // low-pass on the measurements (1 = raw)

    [Header("Optional")]
    public Transform chassis;              // the body that actually moves (auto: root ArticulationBody in children)
    public RobotControls keyboardControls; // gets disabled while ROS is driving

    [Header("Debug (read only)")]
    public float targetV, targetW;        // from /cmd_vel
    public float cmdV, cmdW;              // after acceleration limits
    public float measV, measW;            // measured, filtered
    public float outV, outW;              // sent to the wheels (after PI)
    public string measSource;             // which body the measurement comes from

    float lastMsgTime = -999f;
    bool rosDriving;
    float iV, iW;
    ArticulationBody bodyAB; Rigidbody bodyRB;
    Vector3 lastPos; float lastYaw;

    void Start()
    {
        ROSConnection.GetOrCreateInstance().Subscribe<TwistMsg>(topic, OnCmdVel);

        if (trackWidth <= 0f && frontLeftWheel && frontRightWheel)
            trackWidth = Vector3.Distance(frontLeftWheel.transform.position, frontRightWheel.transform.position);

        if (!keyboardControls) keyboardControls = GetComponent<RobotControls>();
        // find the body that actually moves: often a child (base_link), not this GameObject
        if (chassis) { bodyAB = chassis.GetComponent<ArticulationBody>(); bodyRB = chassis.GetComponent<Rigidbody>(); }
        if (!bodyAB && !bodyRB)
        {
            foreach (var ab in GetComponentsInChildren<ArticulationBody>())
                if (ab.isRoot) { bodyAB = ab; break; }
        }
        if (!bodyAB && !bodyRB) bodyRB = GetComponentInChildren<Rigidbody>();
        if (!chassis) chassis = bodyAB ? bodyAB.transform : bodyRB ? bodyRB.transform : transform;
        measSource = (bodyAB ? "ArticulationBody " : bodyRB ? "Rigidbody " : "transform ") + chassis.name;
        Debug.Log("[CmdVelDriver] measuring from " + measSource);
        lastPos = chassis.position; lastYaw = chassis.eulerAngles.y;
    }

    void OnCmdVel(TwistMsg msg)
    {
        targetV = Mathf.Clamp((float)msg.linear.x, -maxSpeed, maxSpeed);
        targetW = Mathf.Clamp((float)msg.angular.z, -maxYawRate, maxYawRate);
        lastMsgTime = Time.time;
    }

    void Measure(float dt)
    {
        Vector3 vel, angVel;
        if (bodyAB) { vel = bodyAB.velocity; angVel = bodyAB.angularVelocity; }
        else if (bodyRB) { vel = bodyRB.linearVelocity; angVel = bodyRB.angularVelocity; }
        else
        {
            vel = (chassis.position - lastPos) / dt;
            angVel = new Vector3(0f, Mathf.DeltaAngle(lastYaw, chassis.eulerAngles.y) * Mathf.Deg2Rad / dt, 0f);
        }
        lastPos = chassis.position; lastYaw = chassis.eulerAngles.y;
        float v = Vector3.Dot(vel, chassis.forward);
        float w = -Vector3.Dot(angVel, chassis.up);    // Unity +yaw = clockwise from above = right turn
        measV = Mathf.Lerp(measV, v, measureFilter);
        measW = Mathf.Lerp(measW, w, measureFilter);
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime;
        Measure(dt);

        bool fresh = Time.time - lastMsgTime < timeout;
        if (fresh && !rosDriving) SetRosDriving(true);
        if (!rosDriving) return;
        if (!fresh) { targetV = 0f; targetW = 0f; }

        // acceleration limits
        cmdV = Mathf.MoveTowards(cmdV, targetV, maxAccel * dt);
        cmdW = Mathf.MoveTowards(cmdW, targetW, maxAngAccel * dt);

        if (!fresh && Mathf.Abs(cmdV) < 1e-3f && Mathf.Abs(cmdW) < 1e-3f)
        {   // came to rest after the commands stopped: release
            Drive(0f, 0f); iV = iW = 0f; outV = outW = 0f;
            SetRosDriving(false);
            return;
        }

        outV = cmdV; outW = cmdW;
        if (closedLoop)
        {
            float eV = cmdV - measV, eW = cmdW - measW;
            bool resting = Mathf.Abs(cmdV) < 1e-3f && Mathf.Abs(cmdW) < 1e-3f;
            if (resting) { iV = 0f; iW = 0f; }
            else
            {
                iV = Mathf.Clamp(iV + kiV * eV * dt, -maxCorrectionV, maxCorrectionV);
                iW = Mathf.Clamp(iW + kiW * eW * dt, -maxCorrectionW, maxCorrectionW);
            }
            outV = cmdV + kpV * eV + iV;
            outW = cmdW + kpW * eW + iW;
        }

        // skid-steer kinematics: positive w = turn left
        float half = 0.5f * trackWidth * turnSlipFactor;
        float left = (outV - outW * half) / wheelRadius;    // rad/s
        float right = (outV + outW * half) / wheelRadius;
        Drive(left * Mathf.Rad2Deg, right * Mathf.Rad2Deg); // articulation drives want deg/s
    }

    void SetRosDriving(bool on)
    {
        rosDriving = on;
        if (!on) { cmdV = cmdW = 0f; iV = iW = 0f; }
        if (keyboardControls) keyboardControls.enabled = !on;
    }

    void Drive(float leftDeg, float rightDeg)
    {
        Set(frontLeftWheel, leftDeg);
        Set(rearLeftWheel, leftDeg);
        Set(frontRightWheel, rightDeg);
        Set(rearRightWheel, rightDeg);
    }

    static void Set(ArticulationBody wheel, float velDeg)
    {
        if (!wheel) return;
        var d = wheel.xDrive;
        d.stiffness = 0f;
        d.damping = 10f;
        d.forceLimit = 10000f;
        d.targetVelocity = velDeg;
        wheel.xDrive = d;
    }
}
