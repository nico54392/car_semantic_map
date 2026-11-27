// CourseTools.cs: Unity menu "CAR > Course" to build the test course without hand-wiring classes.
// Goes in Assets/Editor/ (any folder called Editor). Needs SceneCapture.cs + TerrainClass.cs in the project.
//
// 1. python tools/megascans_to_unity.py --out <Project>/Assets/Course/Surfaces grass=... dirt=... mud=... gravel=... puddle=...
// 2. CAR > Course > 1. Create Terrain Layers        Assets/Course/Surfaces/<class>/ -> Assets/Course/Layers/<class>.terrainlayer
//                                                   (named after the class, so TerrainFriction finds each layer's mu)
// 3. Select the Terrain, CAR > Course > 2. Put Layers on Terrain   (same order as surfaces.json; layer 0 = base everywhere)
// 4. Paint the course (Paint Texture).
// 5. Select rocks / logs / bench / trees, CAR > Course > 3. Tag Selection...   -> TerrainClass + colliders
// 6. CAR > Course > 4. Sync SceneCapture Classes    classes list + terrainLayerToClass from the layer names and tags
// City / mesh scenes: CAR > Course > Add Colliders to Selection on the city root, so depth and the wheels see it.
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public static class CourseTools
{
    const string SurfacesDir = "Assets/Course/Surfaces";
    const string LayersDir = "Assets/Course/Layers";

    [System.Serializable] class Surface { public string cls; public float tile = 2f; }
    [System.Serializable] class SurfaceList { public string[] order; public Surface[] surfaces; }

    static SurfaceList ReadSurfaces()
    {
        var path = Path.Combine(SurfacesDir, "surfaces.json");
        if (!File.Exists(path)) { Debug.LogError($"CourseTools: {path} not found. Run tools/megascans_to_unity.py --out {SurfacesDir} first."); return null; }
        return JsonUtility.FromJson<SurfaceList>(File.ReadAllText(path));
    }

    static T LoadTex<T>(string path) where T : Object => File.Exists(path) ? AssetDatabase.LoadAssetAtPath<T>(path) : null;

    static void SetImport(string path, bool normal, bool srgb)
    {
        var imp = AssetImporter.GetAtPath(path) as TextureImporter;
        if (imp == null) return;
        bool dirty = false;
        if (normal && imp.textureType != TextureImporterType.NormalMap) { imp.textureType = TextureImporterType.NormalMap; dirty = true; }
        if (imp.sRGBTexture != srgb) { imp.sRGBTexture = srgb; dirty = true; }
        if (imp.wrapMode != TextureWrapMode.Repeat) { imp.wrapMode = TextureWrapMode.Repeat; dirty = true; }
        if (dirty) imp.SaveAndReimport();
    }

    [MenuItem("CAR/Course/1. Create Terrain Layers")]
    static void CreateLayers()
    {
        var list = ReadSurfaces(); if (list == null) return;
        AssetDatabase.Refresh();
        Directory.CreateDirectory(LayersDir);
        int n = 0;
        foreach (var s in list.surfaces)
        {
            string d = $"{SurfacesDir}/{s.cls}";
            string alb = $"{d}/{s.cls}_albedo.png", nrm = $"{d}/{s.cls}_normal.png", msk = $"{d}/{s.cls}_mask.png";
            SetImport(alb, false, true); SetImport(nrm, true, false); SetImport(msk, false, false);
            string lp = $"{LayersDir}/{s.cls}.terrainlayer";
            var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(lp);
            bool isNew = layer == null;
            if (isNew) layer = new TerrainLayer();
            layer.name = s.cls;
            layer.diffuseTexture = LoadTex<Texture2D>(alb);
            layer.normalMapTexture = LoadTex<Texture2D>(nrm);
            layer.maskMapTexture = LoadTex<Texture2D>(msk);
            layer.tileSize = new Vector2(s.tile, s.tile);
            if (isNew) AssetDatabase.CreateAsset(layer, lp); else EditorUtility.SetDirty(layer);
            n++;
        }
        AssetDatabase.SaveAssets();
        Debug.Log($"CourseTools: {n} terrain layers in {LayersDir} ({string.Join(", ", list.order)})");
    }

    static Terrain PickTerrain()
    {
        var t = Selection.activeGameObject ? Selection.activeGameObject.GetComponent<Terrain>() : null;
        return t ? t : Terrain.activeTerrain;
    }

    [MenuItem("CAR/Course/2. Put Layers on Terrain")]
    static void PutLayers()
    {
        var list = ReadSurfaces(); if (list == null) return;
        var terrain = PickTerrain();
        if (!terrain) { Debug.LogError("CourseTools: select a Terrain (or have one in the scene)"); return; }
        var layers = list.order.Select(c => AssetDatabase.LoadAssetAtPath<TerrainLayer>($"{LayersDir}/{c}.terrainlayer"))
                               .Where(l => l != null).ToArray();
        if (layers.Length > 8) Debug.LogWarning("CourseTools: more than 8 layers is slow in URP; keep it to the classes you need");
        Undo.RecordObject(terrain.terrainData, "Course terrain layers");
        terrain.terrainData.terrainLayers = layers;
        EditorUtility.SetDirty(terrain.terrainData);
        Debug.Log($"CourseTools: terrain '{terrain.name}' layers = {string.Join(", ", layers.Select(l => l.name))}. " +
                  "Layer 0 covers everything until you paint. Then: CAR > Course > 4. Sync SceneCapture Classes");
    }

    // ---------------------------------------------------------------- classes

    static SceneCapture FindCapture()
    {
        var sc = Object.FindFirstObjectByType<SceneCapture>();
        if (!sc) Debug.LogError("CourseTools: no SceneCapture in the scene (put it on the capture camera)");
        return sc;
    }

    static int EnsureClass(SceneCapture sc, string name)
    {
        name = name.Trim().ToLower();
        var e = sc.classes.FirstOrDefault(c => c.name.ToLower() == name);
        if (e != null) return e.id;
        int id = sc.classes.Count == 0 ? 1 : sc.classes.Max(c => c.id) + 1;
        if (!sc.classes.Any(c => c.id == 0)) sc.classes.Insert(0, new SceneCapture.ClassEntry { id = 0, name = "void" });
        sc.classes.Add(new SceneCapture.ClassEntry { id = id, name = name });
        return id;
    }

    [MenuItem("CAR/Course/4. Sync SceneCapture Classes")]
    static void SyncClasses()
    {
        var sc = FindCapture(); if (!sc) return;
        var terrain = sc.terrain ? sc.terrain : PickTerrain();
        Undo.RecordObject(sc, "Sync SceneCapture classes");
        var tagged = Object.FindObjectsByType<TerrainClass>(FindObjectsSortMode.None);
        string layers = "no Terrain (area mode: classes come from tagged objects only)";
        if (terrain)
        {
            sc.terrain = terrain;
            sc.terrainLayerToClass = new List<int>();
            foreach (var l in terrain.terrainData.terrainLayers)
                sc.terrainLayerToClass.Add(l ? EnsureClass(sc, l.name) : 0);
            layers = $"terrainLayerToClass = [{string.Join(", ", sc.terrainLayerToClass)}] for layers " +
                     $"[{string.Join(", ", terrain.terrainData.terrainLayers.Select(l => l ? l.name : "-"))}]";
        }
        EditorUtility.SetDirty(sc);
        Debug.Log($"CourseTools: {layers}; classes: {string.Join(", ", sc.classes.Select(c => $"{c.id}={c.name}"))}; " +
                  $"{tagged.Length} tagged objects. Class names must match config/ontology.json.");
    }

    // ---------------------------------------------------------------- colliders only

    [MenuItem("CAR/Course/Add Colliders to Selection (and children)")]
    static void AddColliders()
    {
        int added = 0, had = 0;
        foreach (var go in Selection.gameObjects)
            foreach (var mf in go.GetComponentsInChildren<MeshFilter>(true))
            {
                if (mf.sharedMesh == null) continue;
                if (mf.GetComponent<Collider>() != null) { had++; continue; }
                Undo.AddComponent<MeshCollider>(mf.gameObject).sharedMesh = mf.sharedMesh;
                added++;
            }
        Debug.Log($"CourseTools: added {added} MeshColliders ({had} meshes already had a collider). " +
                  "Depth (RGBDPublisher / SceneCapture) and the robot's wheels only see things with colliders.");
    }

    // ---------------------------------------------------------------- tag objects

    public class TagWindow : EditorWindow
    {
        string cls = "rock";
        bool addColliders = true, visualOnly = false;

        [MenuItem("CAR/Course/3. Tag Selection...")]
        static void Open() => GetWindow<TagWindow>("Tag selection").minSize = new Vector2(320, 150);

        void OnGUI()
        {
            EditorGUILayout.LabelField($"{Selection.gameObjects.Length} object(s) selected");
            cls = EditorGUILayout.TextField("Class (ontology name)", cls);
            addColliders = EditorGUILayout.Toggle("Add MeshColliders if missing", addColliders);
            visualOnly = EditorGUILayout.Toggle(new GUIContent("Visual only (no collider)",
                "e.g. the water plane over a puddle: the rays go through to the 'puddle' terrain layer"), visualOnly);
            if (GUILayout.Button("Tag")) Tag();
        }

        void Tag()
        {
            var sc = FindCapture(); if (!sc) return;
            Undo.RecordObject(sc, "Tag selection");
            int id = EnsureClass(sc, cls);
            int n = 0;
            foreach (var go in Selection.gameObjects)
            {
                if (visualOnly)
                {
                    foreach (var c in go.GetComponentsInChildren<Collider>()) Undo.DestroyObjectImmediate(c);
                    n++; continue;
                }
                var tc = go.GetComponent<TerrainClass>();
                if (tc == null) tc = Undo.AddComponent<TerrainClass>(go);   // no ?? on Unity objects
                Undo.RecordObject(tc, "Tag");
                tc.classId = id;
                if (addColliders && go.GetComponentInChildren<Collider>() == null)
                    foreach (var mf in go.GetComponentsInChildren<MeshFilter>())
                        Undo.AddComponent<MeshCollider>(mf.gameObject).sharedMesh = mf.sharedMesh;
                // child colliders need nothing: SceneCapture uses GetComponentInParent<TerrainClass>()
                n++;
            }
            EditorUtility.SetDirty(sc);
            Debug.Log(visualOnly ? $"CourseTools: removed colliders from {n} object(s) (visual only)"
                                 : $"CourseTools: {n} object(s) tagged '{cls}' (id {id})");
        }
    }
}
