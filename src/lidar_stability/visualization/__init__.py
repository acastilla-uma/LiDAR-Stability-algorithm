"""
Terrain Impact Visualization Module

Analyzes model performance by segment and visualizes LiDAR terrain data with GPS routes.
Interactive visualization of positive and negative impact segments with 3D point cloud overlays.
"""

__version__ = "0.1.0"
__author__ = "LiDAR Stability Team"

from .segment_ranking import SegmentScore, load_and_rank_segments
from .segment_loader import SegmentData, load_segment_data, slice_segment_data
from .point_cloud_processor import PointCloudData, process_segment_point_cloud
from .map_builder import create_segment_visualization

__all__ = [
    "SegmentScore",
    "SegmentData",
    "PointCloudData",
    "load_and_rank_segments",
    "load_segment_data",
    "slice_segment_data",
    "process_segment_point_cloud",
    "create_segment_visualization",
]
