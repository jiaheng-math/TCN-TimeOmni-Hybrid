from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_float(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def build_engine_summary(
    unit_frame: pd.DataFrame,
    feature_columns: list[str],
    history_cycles: int,
    top_k_features: int,
) -> dict:
    recent = unit_frame.tail(history_cycles).copy()
    numeric_columns = [column for column in feature_columns if column in recent.columns]

    feature_rows = []
    for column in numeric_columns:
        series_full = unit_frame[column].astype(float)
        series_recent = recent[column].astype(float)
        start = float(series_recent.iloc[0])
        end = float(series_recent.iloc[-1])
        delta = end - start
        std_full = float(series_full.std(ddof=0))
        recent_std = float(series_recent.std(ddof=0))
        score = abs(delta) / (std_full + 1.0e-6)
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        feature_rows.append(
            {
                "feature": column,
                "start": _safe_float(start),
                "end": _safe_float(end),
                "delta": _safe_float(delta),
                "recent_std": _safe_float(recent_std),
                "score": _safe_float(score),
                "direction": direction,
            }
        )

    feature_rows.sort(key=lambda item: item["score"], reverse=True)
    selected_rows = feature_rows[:top_k_features]

    trend_lines = []
    for item in selected_rows:
        trend_lines.append(
            (
                f"{item['feature']}: {item['direction']}, "
                f"delta={item['delta']:.4f}, recent_std={item['recent_std']:.4f}, "
                f"start={item['start']:.4f}, end={item['end']:.4f}"
            )
        )

    cycle_start = int(recent["cycle"].iloc[0])
    cycle_end = int(recent["cycle"].iloc[-1])
    return {
        "observed_cycles": int(unit_frame["cycle"].max()),
        "summary_window": {
            "history_cycles": int(len(recent)),
            "start_cycle": cycle_start,
            "end_cycle": cycle_end,
        },
        "top_features": selected_rows,
        "trend_lines": trend_lines,
    }
