// TerrainFriction.cs: each terrain layer gets its own wheel friction, so the robot actually slips on mud.
// Put it on the robot root (A4WD3), next to CmdVelDriver. Drag the Terrain in.
//
// "layers": one entry per terrain layer, SAME ORDER as Paint Texture (asphalt, dirt, grass, gravel, mud on FlatTest).
// It fills itself when you add the component (or right-click the component > "Fill from terrain"), from the
// terrain's layer names + the ontology's first-guess mu. Then edit the mu sliders in the Inspector (NOT in Play mode,
// Play-mode edits are lost when you stop).
//
// How: every physics step, look up the paint weights under each wheel (cached alphamap, no raycast),
// mu = weighted mix of the layers' mu, and set it on that wheel's own PhysicsMaterial.
// Wheels use frictionCombine = Multiply and the terrain gets a friction-1 material, so the contact friction = mu.
// Turn "applyFriction" off to get the old behaviour (everything like asphalt) for A/B tests.
using System;
using System.Collections.Generic;
using UnityEngine;

public class TerrainFriction : MonoBehaviour
{
    [Serializable] public class LayerFriction { public string name; [Range(0f, 1.5f)] public float mu = 0.6f; }

    public Terrain terrain;
    public bool applyFriction = true;
    public float offTerrainMu = 0.7f;
    public List<LayerFriction> layers = new List<LayerFriction>();

    [Header("Debug (read only)")]
    public string[] wheelSurface = new string[4];   // FL, RL, FR, RR
    public float[] wheelMu = new float[4];

    // first-guess mu, same as config/ontology.json
    static readonly Dictionary<string, float> Defaults = new Dictionary<string, float> {
        {"asphalt", 0.7f}, {"concrete", 0.7f}, {"dirt", 0.5f}, {"grass", 0.35f}, {"gravel", 0.45f},
        {"mud", 0.2f}, {"puddle", 0.2f}, {"rubble", 0.5f}, {"sand", 0.4f}, {"snow", 0.2f}, {"ice", 0.08f} };

    ArticulationBody[] wheels;
    Collider[][] wheelCols;
    PhysicsMaterial[] mats;
    float[,,] alpha;
    TerrainData td;

    void Start()
    {
        var drv = GetComponent<CmdVelDriver>();
        if (!drv) { Debug.LogError("[TerrainFriction] needs CmdVelDriver on the same object (for the wheel list)"); enabled = false; return; }
        if (!terrain) terrain = Terrain.activeTerrain;
        td = terrain.terrainData;
        alpha = td.GetAlphamaps(0, 0, td.alphamapWidth, td.alphamapHeight);

        if (layers.Count == 0) Fill();
        if (layers.Count != td.alphamapLayers)
            Debug.LogWarning($"[TerrainFriction] {layers.Count} entries but the terrain has {td.alphamapLayers} layers");

        // terrain side: friction 1, multiplied by the wheel's mu
        var ground = new PhysicsMaterial("terrain_mu1") { dynamicFriction = 1f, staticFriction = 1f,
            frictionCombine = PhysicsMaterialCombine.Multiply, bounceCombine = PhysicsMaterialCombine.Minimum };
        terrain.GetComponent<TerrainCollider>().sharedMaterial = ground;

        wheels = new[] { drv.frontLeftWheel, drv.rearLeftWheel, drv.frontRightWheel, drv.rearRightWheel };
        wheelCols = new Collider[4][]; mats = new PhysicsMaterial[4];
        for (int i = 0; i < 4; i++)
        {
            mats[i] = new PhysicsMaterial("wheel_" + i) { frictionCombine = PhysicsMaterialCombine.Multiply,
                bounceCombine = PhysicsMaterialCombine.Minimum, dynamicFriction = 0.6f, staticFriction = 0.6f };
            wheelCols[i] = wheels[i] ? wheels[i].GetComponentsInChildren<Collider>() : new Collider[0];
            foreach (var c in wheelCols[i]) c.sharedMaterial = mats[i];
        }
        string s = "";
        for (int k = 0; k < layers.Count; k++) s += $"{k}:{layers[k].name}={layers[k].mu:0.00}  ";
        Debug.Log($"[TerrainFriction] layers {s}| wheel colliders {wheelCols[0].Length},{wheelCols[1].Length},{wheelCols[2].Length},{wheelCols[3].Length}");
    }

    void Reset() { terrain = Terrain.activeTerrain; Fill(); }      // runs when the component is added in the editor

    [ContextMenu("Fill from terrain")]
    void Fill()
    {
        if (!terrain) terrain = Terrain.activeTerrain;
        if (!terrain) return;
        layers.Clear();
        foreach (var tl in terrain.terrainData.terrainLayers)
        {
            string n = tl ? tl.name.ToLower() : "?";
            float mu = 0.6f;
            foreach (var kv in Defaults) if (n.Contains(kv.Key)) { mu = kv.Value; n = kv.Key; break; }
            layers.Add(new LayerFriction { name = n, mu = mu });
        }
    }

    void FixedUpdate()
    {
        if (td == null) return;
        for (int i = 0; i < 4; i++)
        {
            if (!wheels[i]) continue;
            float mu = offTerrainMu; string surf = "off terrain";
            if (applyFriction && Sample(wheels[i].transform.position, out float m, out string best)) { mu = m; surf = best; }
            if (!applyFriction) { mu = offTerrainMu; surf = "friction off"; }
            if (Mathf.Abs(mats[i].dynamicFriction - mu) > 0.005f) { mats[i].dynamicFriction = mu; mats[i].staticFriction = mu; }
            wheelMu[i] = mu; wheelSurface[i] = surf;
        }
    }

    bool Sample(Vector3 world, out float mu, out string best)
    {
        mu = offTerrainMu; best = "off terrain";
        Vector3 p = world - terrain.transform.position;
        float u = p.x / td.size.x, v = p.z / td.size.z;
        if (u < 0f || u > 1f || v < 0f || v > 1f) return false;
        int x = Mathf.Clamp((int)(u * (td.alphamapWidth - 1)), 0, td.alphamapWidth - 1);
        int z = Mathf.Clamp((int)(v * (td.alphamapHeight - 1)), 0, td.alphamapHeight - 1);
        float sum = 0f, wsum = 0f, wbest = -1f;
        for (int k = 0; k < td.alphamapLayers && k < layers.Count; k++)
        {
            float w = alpha[z, x, k];                       // alphamap is [y(z), x, layer]
            sum += w * layers[k].mu; wsum += w;
            if (w > wbest) { wbest = w; best = layers[k].name; }
        }
        if (wsum > 1e-4f) mu = sum / wsum;
        return true;
    }
}
