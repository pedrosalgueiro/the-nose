from __future__ import annotations

import numpy as np
import pandas as pd

#from config import SENSOR_COLUMNS, WINDOW_SIZE

try:
    from .config import SENSOR_COLUMNS, WINDOW_SIZE
except ImportError:
    from config import SENSOR_COLUMNS, WINDOW_SIZE


# Protocol used for real data collection:
#   0-30 s    clean-air baseline
#   30-90 s   sample exposure
#   90-180 s  recovery
BASELINE_END_S = 30.0
EXPOSURE_START_S = 30.0
EXPOSURE_END_S = 90.0
RECOVERY_START_S = 90.0
RECOVERY_END_S = 180.0
FINAL_WINDOW_S = 15.0


def _safe_numeric(series: pd.Series | np.ndarray, length: int | None = None) -> np.ndarray:
    """Return a float numpy array with missing values filled safely."""
    if isinstance(series, np.ndarray):
        s = pd.Series(series)
    else:
        s = series.copy()
    values = pd.to_numeric(s, errors="coerce").ffill().bfill().fillna(0.0).astype(float).to_numpy()
    if length is not None and len(values) == 0:
        return np.zeros(length, dtype=float)
    return values


def _safe_slope(values: np.ndarray, x: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    if x is None:
        x = np.arange(len(values), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    # Avoid singular fits if timestamps are repeated.
    if len(np.unique(x)) < 2:
        x = np.arange(len(values), dtype=float)
    try:
        return float(np.polyfit(x, values, 1)[0])
    except Exception:
        return 0.0


def _safe_auc(values: np.ndarray, x: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    if x is None:
        x = np.arange(len(values), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    try:
        return float(np.trapz(values, x))
    except Exception:
        return 0.0


def _time_axis_seconds(df: pd.DataFrame) -> np.ndarray:
    """Return elapsed seconds for a session/window.

    Prefer the collect script's `elapsed_s`. For live app records, derive a relative
    axis from `timestamp_ms`. If neither exists, use row index as seconds.
    """
    if "elapsed_s" in df.columns:
        t = pd.to_numeric(df["elapsed_s"], errors="coerce").ffill().bfill()
        if t.notna().any():
            arr = t.fillna(0.0).astype(float).to_numpy()
            return arr - float(arr[0])

    if "timestamp_ms" in df.columns:
        t = pd.to_numeric(df["timestamp_ms"], errors="coerce").ffill().bfill()
        if t.notna().any():
            arr = t.fillna(0.0).astype(float).to_numpy()
            return (arr - float(arr[0])) / 1000.0

    return np.arange(len(df), dtype=float)


def _segment_mask(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return (t >= start_s) & (t < end_s)


def _segment_values(values: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    seg = values[mask]
    if len(seg) == 0:
        return fallback if len(fallback) else values
    return seg


def extract_segmented_features(session: pd.DataFrame) -> dict:
    """Extract one feature vector from one full collection session.

    This function is aligned with the recommended sampling protocol:
    baseline -> exposure -> recovery. It uses baseline-relative features, so the
    model learns the response shape rather than absolute raw sensor levels only.
    """
    session = session.reset_index(drop=True).copy()
    t = _time_axis_seconds(session)

    baseline_mask = _segment_mask(t, 0.0, BASELINE_END_S)
    exposure_mask = _segment_mask(t, EXPOSURE_START_S, EXPOSURE_END_S)
    recovery_mask = _segment_mask(t, RECOVERY_START_S, RECOVERY_END_S + 1e-9)
    final_mask = t >= max(0.0, float(np.nanmax(t)) - FINAL_WINDOW_S) if len(t) else np.array([], dtype=bool)

    features: dict[str, float] = {
        "session_duration_s": float(np.nanmax(t) - np.nanmin(t)) if len(t) else 0.0,
        "session_rows": float(len(session)),
    }

    for col in SENSOR_COLUMNS:
        if col not in session.columns:
            values = np.zeros(len(session), dtype=float)
        else:
            values = _safe_numeric(session[col])

        if len(values) == 0:
            values = np.zeros(len(session), dtype=float)

        baseline = _segment_values(values, baseline_mask, values[: max(1, min(len(values), 5))])
        exposure = _segment_values(values, exposure_mask, values)
        recovery = _segment_values(values, recovery_mask, values[-max(1, min(len(values), 5)):])
        final = _segment_values(values, final_mask, values[-max(1, min(len(values), 5)):])

        baseline_t = _segment_values(t, baseline_mask, t[: len(baseline)])
        exposure_t = _segment_values(t, exposure_mask, t[: len(exposure)])
        recovery_t = _segment_values(t, recovery_mask, t[-len(recovery):])

        b_mean = float(np.mean(baseline))
        b_std = float(np.std(baseline))
        exp_mean = float(np.mean(exposure))
        exp_max = float(np.max(exposure))
        exp_min = float(np.min(exposure))
        rec_mean = float(np.mean(recovery))
        final_mean = float(np.mean(final))
        global_mean = float(np.mean(values))
        global_max = float(np.max(values))
        global_min = float(np.min(values))
        peak_idx = int(np.argmax(values)) if len(values) else 0
        peak_time = float(t[peak_idx]) if len(t) else 0.0
        peak_value = float(values[peak_idx]) if len(values) else 0.0

        # Signed and positive response above baseline. Some sensors, such as
        # BME688 gas resistance, may decrease on VOC exposure, so we keep both.
        exposure_delta = exposure - b_mean
        recovery_delta = recovery - b_mean
        values_delta = values - b_mean

        peak_delta = peak_value - b_mean
        final_delta = final_mean - b_mean
        abs_peak_delta = float(np.max(np.abs(values_delta))) if len(values_delta) else 0.0

        response_den = abs(peak_delta) if abs(peak_delta) > 1e-9 else 0.0
        recovery_fraction = 0.0
        if response_den > 0:
            # 1.0 means it returned to baseline; 0.0 means no recovery.
            recovery_fraction = float(1.0 - (abs(final_delta) / response_den))

        features.update({
            f"{col}_global_mean": global_mean,
            f"{col}_global_std": float(np.std(values)),
            f"{col}_global_min": global_min,
            f"{col}_global_max": global_max,
            f"{col}_global_range": global_max - global_min,

            f"{col}_baseline_mean": b_mean,
            f"{col}_baseline_std": b_std,
            f"{col}_baseline_slope": _safe_slope(baseline, baseline_t),

            f"{col}_exposure_mean": exp_mean,
            f"{col}_exposure_std": float(np.std(exposure)),
            f"{col}_exposure_min": exp_min,
            f"{col}_exposure_max": exp_max,
            f"{col}_exposure_delta_mean": exp_mean - b_mean,
            f"{col}_exposure_delta_max": exp_max - b_mean,
            f"{col}_exposure_delta_min": exp_min - b_mean,
            f"{col}_exposure_slope": _safe_slope(exposure, exposure_t),
            f"{col}_exposure_auc_signed": _safe_auc(exposure_delta, exposure_t),
            f"{col}_exposure_auc_abs": _safe_auc(np.abs(exposure_delta), exposure_t),

            f"{col}_recovery_mean": rec_mean,
            f"{col}_recovery_std": float(np.std(recovery)),
            f"{col}_recovery_delta_mean": rec_mean - b_mean,
            f"{col}_recovery_slope": _safe_slope(recovery, recovery_t),
            f"{col}_recovery_auc_signed": _safe_auc(recovery_delta, recovery_t),
            f"{col}_recovery_auc_abs": _safe_auc(np.abs(recovery_delta), recovery_t),

            f"{col}_final_mean": final_mean,
            f"{col}_final_delta": final_delta,
            f"{col}_peak_value": peak_value,
            f"{col}_peak_delta": peak_delta,
            f"{col}_abs_peak_delta": abs_peak_delta,
            f"{col}_time_to_peak_s": peak_time,
            f"{col}_recovery_fraction": recovery_fraction,
        })

    return features


# Backwards-compatible alias used by some older scripts.
def extract_features_from_window(window: pd.DataFrame) -> dict:
    return extract_segmented_features(window)


def _iter_sessions(df: pd.DataFrame):
    # Always prefer session-level examples. Each session becomes one training row.
    if "session_id" in df.columns and "label" in df.columns:
        return df.groupby(["label", "session_id"], sort=False)
    if "trial" in df.columns and "label" in df.columns:
        return df.groupby(["label", "trial"], sort=False)
    if "label" in df.columns:
        return df.groupby("label", sort=False)
    return [(None, df)]


def make_feature_table(df: pd.DataFrame, window_size: int = WINDOW_SIZE, step_size: int | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Create a feature table with one row per labelled collection session.

    `window_size` and `step_size` are accepted for compatibility, but the default
    training path no longer makes sliding windows. This matches the real data
    collection protocol and avoids treating baseline/recovery as separate labels.
    """
    df = df.copy()
    for col in SENSOR_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    rows: list[dict] = []
    labels: list[str] = []
    has_label = "label" in df.columns

    for group_key, group in _iter_sessions(df):
        group = group.reset_index(drop=True)
        # Skip extremely short / accidental captures.
        if len(group) < 10:
            continue
        rows.append(extract_segmented_features(group))
        if has_label:
            labels.append(str(group["label"].iloc[0]))

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0.0)
    y = pd.Series(labels, name="label") if has_label else None
    return X, y


def features_for_latest_window(records: list[dict], window_size: int = WINDOW_SIZE) -> pd.DataFrame | None:
    """Features for the live app.

    The model was trained with a complete baseline/exposure/recovery session, so
    the app should ideally wait for ~180 seconds of data. If a shorter window is
    passed, the function still returns features, but predictions will be less
    reliable.
    """
    if len(records) < min(window_size, 30):
        return None
    window = pd.DataFrame(records[-window_size:]).copy()
    for col in SENSOR_COLUMNS:
        if col not in window.columns:
            window[col] = 0.0
    return pd.DataFrame([extract_segmented_features(window)]).replace([np.inf, -np.inf], 0).fillna(0.0)



def extract_fast_features(session: pd.DataFrame) -> dict:
    """
    Extract features for fast prediction.

    Protocol:
    0–25 s   baseline
    35–60 s  early exposure
    """

    session = session.sort_values("elapsed_s").copy()

    if "elapsed_s" not in session.columns:
        raise ValueError("Fast features require elapsed_s column.")

    t = session["elapsed_s"].astype(float).to_numpy()

    baseline_mask = (t >= 0.0) & (t < 25.0)
    exposure_mask = (t >= 35.0) & (t <= 60.0)

    features = {}

    for col in SENSOR_COLUMNS:
        if col not in session.columns:
            continue

        values = pd.to_numeric(session[col], errors="coerce").to_numpy(dtype=float)

        baseline = values[baseline_mask]
        exposure = values[exposure_mask]
        exposure_t = t[exposure_mask]

        baseline = baseline[np.isfinite(baseline)]
        exposure = exposure[np.isfinite(exposure)]

        if len(baseline) == 0 or len(exposure) == 0:
            continue

        b_mean = float(np.mean(baseline))
        b_std = float(np.std(baseline)) if len(baseline) > 1 else 0.0

        e_mean = float(np.mean(exposure))
        e_std = float(np.std(exposure)) if len(exposure) > 1 else 0.0
        e_min = float(np.min(exposure))
        e_max = float(np.max(exposure))

        delta_mean = e_mean - b_mean
        delta_min = e_min - b_mean
        delta_max = e_max - b_mean

        ratio_mean = e_mean / b_mean if abs(b_mean) > 1e-9 else 0.0
        ratio_max = e_max / b_mean if abs(b_mean) > 1e-9 else 0.0

        if len(exposure_t) >= 2 and len(exposure) >= 2:
            try:
                slope = float(np.polyfit(exposure_t[:len(exposure)], exposure, 1)[0])
            except Exception:
                slope = 0.0

            delta_series = exposure - b_mean

            auc_delta = safe_auc(delta_series, exposure_t[:len(exposure)])
            auc_abs_delta = safe_auc(np.abs(delta_series), exposure_t[:len(exposure)])

        else:
            slope = 0.0
            auc_delta = 0.0
            auc_abs_delta = 0.0

        features[f"{col}_baseline_mean"] = b_mean
        features[f"{col}_baseline_std"] = b_std

        features[f"{col}_early_mean"] = e_mean
        features[f"{col}_early_std"] = e_std
        features[f"{col}_early_min"] = e_min
        features[f"{col}_early_max"] = e_max

        features[f"{col}_early_delta_mean"] = delta_mean
        features[f"{col}_early_delta_min"] = delta_min
        features[f"{col}_early_delta_max"] = delta_max

        features[f"{col}_early_ratio_mean"] = ratio_mean
        features[f"{col}_early_ratio_max"] = ratio_max

        features[f"{col}_early_slope"] = slope
        features[f"{col}_early_auc_delta"] = auc_delta
        features[f"{col}_early_auc_abs_delta"] = auc_abs_delta

    return features


def make_fast_feature_table(df: pd.DataFrame):
    rows = []
    labels = []

    for (label, session_id), group in df.groupby(["label", "session_id"], sort=False):
        feats = extract_fast_features(group)
        feats["session_id"] = session_id
        rows.append(feats)
        labels.append(label)

    X = pd.DataFrame(rows).fillna(0.0)
    session_ids = X.pop("session_id")
    y = pd.Series(labels, name="label")

    return X, y, session_ids


def features_for_fast_window(records: list[dict]) -> pd.DataFrame | None:
    """
    Create fast model features from app live records.
    Needs around 60 seconds of data.
    """

    if len(records) < 60:
        return None

    df = pd.DataFrame(records[-70:]).copy()

    if "elapsed_s" not in df.columns:
        df["elapsed_s"] = np.arange(len(df), dtype=float)

    feats = extract_fast_features(df)
    if not feats:
        return None

    return pd.DataFrame([feats]).fillna(0.0)



def safe_auc(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]

    if len(y) < 2:
        return 0.0

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))
