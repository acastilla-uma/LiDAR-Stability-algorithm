"""
Segment Ranking by Model Error

Loads holdout predictions and ranks segments by prediction error (MAE).
Identifies positive impact (low error) and negative impact (high error) segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentScore:
    """Score information for a segment."""
    segment_id: str
    device: str
    date: str
    mean_error: float
    std_error: float
    num_samples: int
    num_points_available: int
    impact_threshold: Optional[float] = field(default=None, init=False)
    
    @property
    def impact_type(self) -> str:
        """Classify as positive (low error) or negative (high error)."""
        if self.impact_threshold is None:
            return "negative"  # Default if threshold not set
        return "positive" if self.mean_error < self.impact_threshold else "negative"
    
    def __str__(self) -> str:
        return (f"{self.segment_id} | "
                f"Device: {self.device} | Date: {self.date} | "
                f"Error: {self.mean_error:.4f}±{self.std_error:.4f} | "
                f"Samples: {self.num_samples}")


def _extract_segment_info(filename: str) -> tuple[str, str, str]:
    """
    Extract segment ID, device, and date from filename.
    
    Expected format: DOBACK024_20251007_seg28.csv
    
    Args:
        filename: CSV filename
        
    Returns:
        Tuple of (segment_id, device, date)
    """
    name = Path(filename).stem
    parts = name.split("_")
    
    if len(parts) >= 3:
        device = parts[0]  # e.g., DOBACK024
        date = parts[1]    # e.g., 20251007
        # Preserve full stem to avoid collisions and support extra suffixes.
        segment_id = name
        return segment_id, device, date
    if len(parts) == 2:
        # Some sources are not segmented: DOBACK023_20251012.csv
        device = parts[0]
        date = parts[1]
        segment_id = name
        return segment_id, device, date
    else:
        # Fallback
        return name, "UNKNOWN", "UNKNOWN"


def load_and_rank_segments(
    model_dir: Path,
    model_name: str = "all-devices-no-imu",
    top_n: int = 5,
    featured_data_dir: Optional[Path] = None,
) -> tuple[list[SegmentScore], list[SegmentScore]]:
    """
    Load holdout predictions and rank segments by error.
    
    Args:
        model_dir: Path to models directory (e.g., output/models/extra_trees/)
        model_name: Name of the model subdirectory
        top_n: Number of top positive/negative segments to return
        featured_data_dir: Path to featured data directory (for point counts)
        
    Returns:
        Tuple of (positive_segments, negative_segments), each sorted by error ascending
        
    Raises:
        FileNotFoundError: If model predictions file not found
    """
    model_path = model_dir / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Find holdout predictions file
    holdout_files = list(model_path.glob("*_holdout_predictions.csv"))
    if not holdout_files:
        raise FileNotFoundError(f"No holdout_predictions.csv found in {model_path}")
    
    holdout_file = holdout_files[0]
    logger.info(f"Loading predictions from: {holdout_file.name}")
    
    # Load predictions
    df = pd.read_csv(holdout_file, low_memory=False)
    
    # Verify required columns
    required_cols = ["__source_file", "y_true", "y_pred", "abs_residual"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in holdout predictions: {missing}")
    
    # Group by segment and calculate error statistics
    segment_scores: dict[str, SegmentScore] = {}
    
    for filename, group in df.groupby("__source_file"):
        segment_id, device, date = _extract_segment_info(filename)
        
        mean_error = group["abs_residual"].mean()
        std_error = group["abs_residual"].std()
        num_samples = len(group)
        
        # Try to get point count from featured CSV if available
        num_points = num_samples  # Default: same as samples
        if featured_data_dir:
            featured_file = featured_data_dir / filename
            if featured_file.exists():
                try:
                    featured_df = pd.read_csv(featured_file, low_memory=False)
                    num_points = len(featured_df)
                except Exception as e:
                    logger.warning(f"Could not load point count from {filename}: {e}")
        
        segment_scores[segment_id] = SegmentScore(
            segment_id=segment_id,
            device=device,
            date=date,
            mean_error=mean_error,
            std_error=std_error,
            num_samples=num_samples,
            num_points_available=num_points,
        )
    
    logger.info(f"Ranked {len(segment_scores)} segments")
    
    # Calculate percentile-based threshold (median error)
    all_segments = list(segment_scores.values())
    all_errors = [s.mean_error for s in all_segments]
    threshold = np.percentile(all_errors, 50)  # Median
    logger.info(f"Impact threshold (median error): {threshold:.2f}")
    
    # Re-classify with calculated threshold
    for segment in all_segments:
        segment.impact_threshold = threshold
    
    # Separate into positive and negative
    positive = sorted([s for s in all_segments if s.impact_type == "positive"],
                     key=lambda x: x.mean_error)
    negative = sorted([s for s in all_segments if s.impact_type == "negative"],
                     key=lambda x: x.mean_error, reverse=True)  # Highest error first
    
    logger.info(f"Positive impact: {len(positive)} segments | "
               f"Negative impact: {len(negative)} segments")
    
    return positive[:top_n], negative[:top_n]
