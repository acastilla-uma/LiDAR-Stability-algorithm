"""Track and persist dropped rows during raw-to-processed filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class OutlierTracker:
    """Collect dropped rows and filter-step metrics for one source key."""

    enabled: bool = True
    drop_records: list[pd.DataFrame] = field(default_factory=list)
    filter_steps: list[dict[str, Any]] = field(default_factory=list)

    def log_filter_step(
        self,
        *,
        source_key: str,
        stage: str,
        filter_name: str,
        before_rows: int,
        dropped_rows: int,
    ) -> None:
        """Record row counts before/after each filtering rule."""
        if not self.enabled:
            return

        after_rows = max(0, int(before_rows) - int(dropped_rows))
        self.filter_steps.append(
            {
                "source_key": source_key,
                "stage": stage,
                "filter_name": filter_name,
                "before_rows": int(before_rows),
                "dropped_rows": int(dropped_rows),
                "after_rows": int(after_rows),
            }
        )

    def log_drops(
        self,
        *,
        source_key: str,
        stage: str,
        reason: str,
        source_file: str,
        dropped_df: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store detailed dropped rows for later export."""
        if not self.enabled or dropped_df is None or dropped_df.empty:
            return

        frame = dropped_df.copy()
        frame.insert(0, "source_key", source_key)
        frame.insert(1, "stage", stage)
        frame.insert(2, "reason", reason)
        frame.insert(3, "source_file", source_file)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, pd.Series):
                    frame[key] = value.reindex(frame.index).values
                elif isinstance(value, (list, tuple)) and len(value) == len(frame):
                    frame[key] = list(value)
                else:
                    frame[key] = value

        self.drop_records.append(frame)

    def export(self, output_dir: Path, source_key: str) -> tuple[Path, Path]:
        """Write dropped rows and summary stats to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        detail_path = output_dir / f"{source_key}_dropped_rows.csv"
        summary_path = output_dir / f"{source_key}_drop_summary.json"

        if self.drop_records:
            detail_df = pd.concat(self.drop_records, ignore_index=True)
        else:
            detail_df = pd.DataFrame(
                columns=[
                    "source_key",
                    "stage",
                    "reason",
                    "source_file",
                ]
            )

        detail_df.to_csv(detail_path, index=False)

        if detail_df.empty:
            by_reason: list[dict[str, Any]] = []
        else:
            by_reason_df = (
                detail_df.groupby(["stage", "reason"], dropna=False)
                .size()
                .reset_index(name="dropped_rows")
                .sort_values(["stage", "dropped_rows"], ascending=[True, False])
            )
            by_reason = by_reason_df.to_dict(orient="records")

        summary = {
            "source_key": source_key,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_dropped_rows": int(len(detail_df)),
            "drops_by_stage_and_reason": by_reason,
            "filter_steps": self.filter_steps,
        }

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return detail_path, summary_path
