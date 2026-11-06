"""Read-only access to Jaime's ros2_segmentation_vlm. Nothing here edits his files."""
import os
import sys
from pathlib import Path


def add_jaime_server_to_path(server_dir=None):
    """Put Jaime's server/ folder on sys.path so `inference.sam3_multi_prompt` imports."""
    server_dir = server_dir or os.environ.get("JAIME_SERVER")
    if not server_dir:
        raise RuntimeError("Set JAIME_SERVER to .../ros2_segmentation_vlm/ros2_segmentation_vlm/server")
    server_dir = str(Path(server_dir).expanduser().resolve())
    if not Path(server_dir, "inference", "sam3_multi_prompt.py").exists():
        raise RuntimeError(f"No inference/sam3_multi_prompt.py under {server_dir}")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    return server_dir
