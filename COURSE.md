# Test course (Unity): realistic, traversable, built for "go to the puddle"

Why our own course and not a demo scene: the AI needs real-looking ground (Fantasy Forest put PE-B at the
"everything is grass" level), the demos have no puddles or mud, and here every surface's class, traversability
and friction is known exactly, so the ground truth is trustworthy (unlike RELLIS's mud/dirt).

## Layout (~60 x 60 m terrain, flat-ish with gentle bumps)

```
 N
 +------------------------------------------------------------+
 |  trees + bushes (edge)                     rock garden      |
 |    ~~ grass ~~~~~~~~~~~~~~~~~~~~~~~~        (3-5 rocks)      |
 |   bench + table          puddle B (big)                     |
 |      \                      ::::                            |
 |  ====dirt track loop==================== gravel section ==  |
 |  ||    mud patch ON the track (5 m)     ######             || |
 |  ||        puddle A (small) beside it                      || |
 |  ||                                     fallen logs (2)    || |
 |  ====================================== across the track ==  |
 |   START (robot spawn)        puddle C        bushes          |
 +------------------------------------------------------------+
```

What each part tests:

- **Dirt track loop.** The default "good" surface. Path planning should prefer it.
- **Mud patch on the track.** A low-traversability, low-μ choice between going through and going around. This is where the wheel → wheg switch belongs.
- **Gravel section.** Medium traversability and μ 0.45.
- **Puddles A, B and C, of different sizes.** These exercise "closest / biggest / farthest puddle".
- **Logs across the track.** Rigid obstacles the robot has to climb or avoid.
- **Rock garden.** Rigid obstacles.
- **Bench, table, trees.** Goal objects ("go to the bench") and references ("the mud near the bench").
- **Duplicates of each class** (3 puddles, 2 mud patches, 2 gravel areas). Without duplicates the state-aware prompts have nothing to choose between.

## What to download (Fab, Megascans filter, all free since 2026-11-05)

Download as **FBX**, quality tier **2K** (4K is wasted at robot camera range and slows Unity).
Pick tileable "Surface" assets. The search terms below work; choose by eye, since names change.

| Class (= terrain layer name) | Fab search | Pick for |
|---|---|---|
| `dirt` | "forest path", "dirt road", "soil" | dry, compacted, light brown. Must look clearly different from mud |
| `grass` | "grass", "meadow", "lawn" | short grass (tall grass is better as detail meshes) |
| `mud` | "mud", "wet mud", "muddy ground" | dark, glossy, wet |
| `gravel` | "gravel", "pebbles" | small loose stones |
| `puddle` | "wet ground", "muddy puddle", "wet soil" | dark and shiny. A water plane goes on top of it, see below |

3D assets: 3–5 rocks/boulders, 1–2 "log" / "tree trunk" / "fallen tree", 2–3 bushes.
Trees, bench and table can come from Megascans ("bench", "picnic table") or the free Unity Terrain Sample Asset Pack.

## Build steps

1. **Pack the surfaces.** Run this on the machine with the Unity project:
   ```bash
   python tools/megascans_to_unity.py --out <UnityProject>/Assets/Course/Surfaces \
       dirt=<dl>/Forest_Path_xxx grass=<dl>/Grass_xxx mud=<dl>/Mud_xxx gravel=<dl>/Gravel_xxx puddle=<dl>/Wet_Ground_xxx
   ```
   This writes the albedo, the normal map (green flipped to Unity's convention) and the URP mask map (smoothness = 1 − roughness).
2. **Copy the editor script.** Put `unity/Editor/CourseTools.cs` into `Assets/Editor/`. A **CAR > Course** menu appears.
3. **Create the terrain.** New 60 × 60 m Terrain. Then run **1. Create Terrain Layers**, select the terrain, and run **2. Put Layers on Terrain**.
4. **Paint the layout above** with Paint Texture.
5. **Add the puddles.** Paint the `puddle` layer in slight dips. Then add a Plane slightly above it: a dark, very smooth material, alpha around 0.6. Select the plane and run **3. Tag Selection…** with **Visual only** ticked, which removes its collider. The rays then pass through to the `puddle` layer, so the labels and the μ of 0.2 are right.
6. **Drop the objects in.** Add the rocks, logs, bench and trees. Select each group and run **3. Tag Selection…** with the class (`rock`, `log`, `bench`, `tree`, `bush`). This adds a `TerrainClass` and colliders.
7. **Run 4. Sync SceneCapture Classes.** This fills `classes` and `terrainLayerToClass` from the layer names and the tags.
8. **Friction.** On the robot, `TerrainFriction` > "Fill from terrain". It picks up each layer's μ by name.
9. **Capture and check.** Follow MONDAY.md steps 2–5: `autoViews` ~80, `scene_eval.py check`, `feature_map.py build --backbone pe-b`.
10. **Drive it.**
    ```bash
    python tools/prompt_drive.py $S --controller pp
    ```
    Then try "the closest puddle", "the mud near the bench", "the farthest gravel", "standing water"…

Keep it to **at most 8 terrain layers**, because URP slows down past that.
