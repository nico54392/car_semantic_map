// SceneCapture.cs
// Dumps posed RGB + metric depth + GT labels from a Unity camera, plus a labeled
// point cloud of the map, in the format the python harness (harness/) reads.
//
// Setup:
//   1. Put this on a Camera. Drag your Terrain into "terrain".
//   2. Fill "classes" (id + name). 0 is void/sky, keep it.
//   3. Fill "terrainLayerToClass": entry i = class id for terrain layer i
//      (same order as the layers in the Terrain's Paint Texture tab).
//   4. Non-terrain stuff (rocks, logs, bushes) gets a TerrainClass component.
//      Everything you want in depth/labels needs a collider.
//   5. Play. C = capture this view, M = export map, or tick autoCaptureOnStart.
//
// Conventions (also written into scene.json):
//   world  = ROS FLU (x forward, y left, z up), meters. Unity (x,y,z) -> ROS (z,-x,y)
//   camera = OpenCV (x right, y down, z forward)
//   T_world_cam = 4x4 row-major, camera -> world
//   depth = z-depth along the camera axis in meters, float32, row-major from the
//           top-left pixel, 0 where the ray hit nothing
//   label png = class id in the R channel
// Floats are written with InvariantCulture on purpose; a Spanish locale would
// otherwise write "0,5" and break the json.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Unity.Collections;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class SceneCapture : MonoBehaviour
{
    [Serializable] public class ClassEntry { public int id; public string name; }

    [Header("Output")]
    [Tooltip("Empty = <project>/Captures/scene_<timestamp>")]
    public string outputFolder = "";
    public int width = 640;
    public int height = 480;

    [Header("Ground truth")]
    public List<ClassEntry> classes = new List<ClassEntry> {
        new ClassEntry { id = 0, name = "void" },
        new ClassEntry { id = 1, name = "grass" },
        new ClassEntry { id = 2, name = "dirt" },
        new ClassEntry { id = 3, name = "mud" },
        new ClassEntry { id = 4, name = "gravel" },
        new ClassEntry { id = 5, name = "rock" },
    };
    public Terrain terrain;
    [Tooltip("Index = terrain layer index, value = class id")]
    public List<int> terrainLayerToClass = new List<int>();
    public LayerMask raycastMask = ~0;
    public float maxDistance = 200f;

    [Header("Map export")]
    public float mapCellSize = 0.1f;

    [Header("Auto capture")]
    public bool autoCaptureOnStart = false;
    public int autoViews = 30;
    public float cameraHeight = 0.6f;          // roughly where the robot's camera sits
    public Vector2 pitchDownDeg = new Vector2(10f, 35f);
    public float edgeMargin = 5f;

    Camera cam;
    string dir;
    int frame = 0;
    float[,,] alpha;
    readonly Dictionary<Collider, int> classCache = new Dictionary<Collider, int>();
    static readonly CultureInfo IC = CultureInfo.InvariantCulture;

    void Start()
    {
        Init();
        if (autoCaptureOnStart) StartCoroutine(AutoCapture());
    }

    void Update()
    {
        bool cap = false, map = false;
#if ENABLE_INPUT_SYSTEM
        var kb = UnityEngine.InputSystem.Keyboard.current;
        if (kb != null) { cap = kb.cKey.wasPressedThisFrame; map = kb.mKey.wasPressedThisFrame; }
#else
        cap = Input.GetKeyDown(KeyCode.C);
        map = Input.GetKeyDown(KeyCode.M);
#endif
        if (cap) CaptureView();
        if (map) ExportMap();
    }

    void Init()
    {
        if (cam != null) return;
        cam = GetComponent<Camera>();
        if (cam.usePhysicalProperties)
            Debug.LogWarning("SceneCapture: physical camera is on, lens shift / gate fit will make the intrinsics wrong. Turn it off.");
        dir = string.IsNullOrEmpty(outputFolder)
            ? Path.Combine(Application.dataPath, "..", "Captures", "scene_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"))
            : outputFolder;
        Directory.CreateDirectory(Path.Combine(dir, "views"));
        if (terrain != null)
        {
            var td = terrain.terrainData;
            alpha = td.GetAlphamaps(0, 0, td.alphamapWidth, td.alphamapHeight);
        }
        WriteSceneJson();
        Debug.Log("SceneCapture writing to " + Path.GetFullPath(dir));
    }

    // ------------------------------------------------------------------ views

    [ContextMenu("Capture view")]
    public void CaptureView()
    {
        Init();
        string stem = Path.Combine(dir, "views", frame.ToString("D6"));

        // RGB
        var rt = RenderTexture.GetTemporary(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var prevTarget = cam.targetTexture;
        cam.targetTexture = rt;
        cam.aspect = (float)width / height;
        cam.Render();
        var prevActive = RenderTexture.active;
        RenderTexture.active = rt;
        var rgb = new Texture2D(width, height, TextureFormat.RGB24, false);
        rgb.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        rgb.Apply();
        RenderTexture.active = prevActive;
        cam.targetTexture = prevTarget;
        RenderTexture.ReleaseTemporary(rt);
        File.WriteAllBytes(stem + "_rgb.png", rgb.EncodeToPNG());
        Kill(rgb);

        // depth + labels: one ray per pixel center, same projection as the render
        int n = width * height;
        var cmds = new NativeArray<RaycastCommand>(n, Allocator.TempJob);
        var hits = new NativeArray<RaycastHit>(n, Allocator.TempJob);
        var qp = new QueryParameters(raycastMask, false, QueryTriggerInteraction.Ignore, false);
        for (int v = 0; v < height; v++)
            for (int u = 0; u < width; u++)
            {
                Ray r = cam.ViewportPointToRay(new Vector3((u + 0.5f) / width, 1f - (v + 0.5f) / height, 0f));
                cmds[v * width + u] = new RaycastCommand(r.origin, r.direction, qp, maxDistance);
            }
        RaycastCommand.ScheduleBatch(cmds, hits, 256, 1).Complete();

        var depth = new float[n];
        var lab = new Color32[n];
        Vector3 camPos = cam.transform.position, fwd = cam.transform.forward;
        for (int i = 0; i < n; i++)
        {
            int v = i / width, u = i % width;
            var hit = hits[i];
            byte id = 0;
            if (hit.collider != null)
            {
                // measured from the camera center, not the near-plane ray origin
                depth[i] = Vector3.Dot(hit.point - camPos, fwd);
                id = (byte)ClassAt(hit);
            }
            lab[(height - 1 - v) * width + u] = new Color32(id, id, id, 255); // texture rows go bottom-up
        }
        cmds.Dispose();
        hits.Dispose();
        cam.ResetAspect();

        var bytes = new byte[n * 4];
        Buffer.BlockCopy(depth, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(stem + "_depth.bin", bytes);

        var lt = new Texture2D(width, height, TextureFormat.RGBA32, false);
        lt.SetPixels32(lab);
        lt.Apply();
        File.WriteAllBytes(stem + "_label.png", lt.EncodeToPNG());
        Kill(lt);

        File.WriteAllText(stem + ".json", ViewJson());
        Debug.Log($"SceneCapture: view {frame:D6} saved");
        frame++;
    }

    string ViewJson()
    {
        float fy = height * 0.5f / Mathf.Tan(cam.fieldOfView * 0.5f * Mathf.Deg2Rad);
        float fx = fy;                       // square pixels
        float cx = (width - 1) * 0.5f, cy = (height - 1) * 0.5f;

        // OpenCV camera axes expressed in Unity world, then mapped to ROS world
        Vector3 c0 = U2R(cam.transform.right);
        Vector3 c1 = U2R(-cam.transform.up);
        Vector3 c2 = U2R(cam.transform.forward);
        Vector3 t = U2R(cam.transform.position);
        float[] T = {
            c0.x, c1.x, c2.x, t.x,
            c0.y, c1.y, c2.y, t.y,
            c0.z, c1.z, c2.z, t.z,
            0, 0, 0, 1 };

        var sb = new StringBuilder();
        sb.Append("{\n");
        sb.Append($"  \"width\": {width},\n  \"height\": {height},\n");
        sb.Append($"  \"fx\": {F(fx)}, \"fy\": {F(fy)}, \"cx\": {F(cx)}, \"cy\": {F(cy)},\n");
        sb.Append("  \"T_world_cam\": [");
        for (int i = 0; i < 16; i++) sb.Append(F(T[i])).Append(i < 15 ? ", " : "");
        sb.Append("]\n}\n");
        return sb.ToString();
    }

    // ------------------------------------------------------------------ map

    [ContextMenu("Export map")]
    public void ExportMap()
    {
        Init();
        if (terrain == null) { Debug.LogError("SceneCapture: need a terrain for the map export"); return; }
        var td = terrain.terrainData;
        Vector3 o = terrain.transform.position;
        int nx = Mathf.FloorToInt(td.size.x / mapCellSize), nz = Mathf.FloorToInt(td.size.z / mapCellSize);
        int n = nx * nz;
        float top = o.y + td.size.y + 50f;

        // straight-down rays: you get the top surface (tree canopy, not the ground under it)
        var cmds = new NativeArray<RaycastCommand>(n, Allocator.TempJob);
        var hits = new NativeArray<RaycastHit>(n, Allocator.TempJob);
        var qp = new QueryParameters(raycastMask, false, QueryTriggerInteraction.Ignore, false);
        for (int iz = 0; iz < nz; iz++)
            for (int ix = 0; ix < nx; ix++)
            {
                var from = new Vector3(o.x + (ix + 0.5f) * mapCellSize, top, o.z + (iz + 0.5f) * mapCellSize);
                cmds[iz * nx + ix] = new RaycastCommand(from, Vector3.down, qp, td.size.y + 100f);
            }
        RaycastCommand.ScheduleBatch(cmds, hits, 256, 1).Complete();

        var body = new StringBuilder();
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            var hit = hits[i];
            if (hit.collider == null) continue;
            Vector3 p = U2R(hit.point);
            body.Append(F(p.x)).Append(' ').Append(F(p.y)).Append(' ').Append(F(p.z)).Append(' ')
                .Append(ClassAt(hit).ToString(IC)).Append('\n');
            count++;
        }
        cmds.Dispose();
        hits.Dispose();

        string header = "ply\nformat ascii 1.0\n" +
                        $"element vertex {count}\n" +
                        "property float x\nproperty float y\nproperty float z\nproperty int label\nend_header\n";
        File.WriteAllText(Path.Combine(dir, "map.ply"), header + body);
        Debug.Log($"SceneCapture: map.ply with {count} points");
    }

    // ------------------------------------------------------------------ auto

    IEnumerator AutoCapture()
    {
        yield return null;
        ExportMap();
        if (terrain == null) yield break;
        var td = terrain.terrainData;
        Vector3 o = terrain.transform.position;
        for (int k = 0; k < autoViews; k++)
        {
            float x = o.x + UnityEngine.Random.Range(edgeMargin, td.size.x - edgeMargin);
            float z = o.z + UnityEngine.Random.Range(edgeMargin, td.size.z - edgeMargin);
            float y = terrain.SampleHeight(new Vector3(x, 0, z)) + o.y + cameraHeight;
            transform.SetPositionAndRotation(new Vector3(x, y, z),
                Quaternion.Euler(UnityEngine.Random.Range(pitchDownDeg.x, pitchDownDeg.y), UnityEngine.Random.Range(0f, 360f), 0f));
            yield return null;
            CaptureView();
        }
        Debug.Log("SceneCapture: auto capture done");
    }

    // ------------------------------------------------------------------ helpers

    int ClassAt(RaycastHit hit)
    {
        if (hit.collider is TerrainCollider && alpha != null)
        {
            var td = terrain.terrainData;
            Vector3 p = hit.point - terrain.transform.position;
            int ax = Mathf.Clamp(Mathf.RoundToInt(p.x / td.size.x * (td.alphamapWidth - 1)), 0, td.alphamapWidth - 1);
            int az = Mathf.Clamp(Mathf.RoundToInt(p.z / td.size.z * (td.alphamapHeight - 1)), 0, td.alphamapHeight - 1);
            int best = 0; float bw = -1f;
            for (int l = 0; l < alpha.GetLength(2); l++)
                if (alpha[az, ax, l] > bw) { bw = alpha[az, ax, l]; best = l; }
            return best < terrainLayerToClass.Count ? terrainLayerToClass[best] : 0;
        }
        var key = hit.collider;
        if (!classCache.TryGetValue(key, out int id))
        {
            var tc = hit.collider.GetComponentInParent<TerrainClass>();
            id = tc != null ? tc.classId : 0;
            classCache[key] = id;
        }
        return id;
    }

    static void Kill(UnityEngine.Object o) { if (Application.isPlaying) Destroy(o); else DestroyImmediate(o); }
    static Vector3 U2R(Vector3 v) => new Vector3(v.z, -v.x, v.y);
    static string F(float f) => f.ToString("R", IC);

    void WriteSceneJson()
    {
        var sb = new StringBuilder("{\n  \"classes\": {");
        for (int i = 0; i < classes.Count; i++)
            sb.Append($"\"{classes[i].id}\": \"{classes[i].name}\"").Append(i < classes.Count - 1 ? ", " : "");
        sb.Append("},\n");
        sb.Append("  \"world_frame\": \"ROS FLU (x forward, y left, z up), meters; unity (x,y,z) -> (z,-x,y)\",\n");
        sb.Append("  \"camera_frame\": \"OpenCV (x right, y down, z forward)\",\n");
        sb.Append("  \"depth\": \"float32 z-depth in meters, row-major from top-left, 0 = no hit\",\n");
        sb.Append("  \"label\": \"class id in R channel\",\n");
        sb.Append("  \"source\": \"unity\"\n}\n");
        File.WriteAllText(Path.Combine(dir, "scene.json"), sb.ToString());
    }
}
