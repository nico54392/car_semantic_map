using UnityEngine;

// Put this on any object (rock, log, bush...) to give it a ground-truth class id.
// Ids have to match "classes" in scene.json (SceneCapture writes that file).
// Terrain surfaces don't need this, SceneCapture reads their terrain layers instead.
public class TerrainClass : MonoBehaviour
{
    public int classId = 0;
}
