# -*- coding: utf-8 -*-
"""
25Hz 手腕六轴 IMU 计步项目 - 数据预处理脚本

作用：
1. 读取原始 CSV
2. 做数据体检
3. 生成 8 维特征：ax, ay, az, gx, gy, gz, acc_mag, gyro_mag
4. 低通滤波：默认 7Hz，适配 25Hz
5. 将硬标签 label=1 转成高斯软标签，sigma=2 帧
6. 切 64 帧窗口，stride=5
7. 生成 step 标签和 gait 门控标签
8. 按每个文件的时间顺序切 train/val/test
9. 只用训练集计算标准化参数，并保存处理后的 npy 文件

运行方法：
python preprocess.py --input_dir ./dataset_raw --output_dir ./processed_dataset

注意：
- 当前是原型训练版本，默认使用离线 filtfilt/sosfiltfilt 滤波。
- 后续做实时设备端时，滤波方式需要换成 causal 实时滤波。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# =========================
# 1. 全局参数：25Hz 版本
# =========================
FS_TARGET = 25.0
WINDOW_SIZE = 64        # 64 帧 = 2.56 秒
STRIDE = 5              # 5 帧 = 0.20 秒
GAUSSIAN_SIGMA = 2      # 2 帧 = 80ms
LOWPASS_CUTOFF = 7.0    # 25Hz 下建议 6~8Hz，先用 7Hz

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_GAP_WINDOWS = 3   # 切分边界附近丢几个窗口，减少重叠泄漏

RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "seq_id", "unix_time", "time",
    "ax", "ay", "az", "gx", "gy", "gz", "label"
]
IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]
FEATURE_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"]



@dataclass
class FileSpec:
    canonical_name: str
    action: str
    kind: str  # "positive" or "negative"
    aliases: Tuple[str, ...] = ()


# canonical_name 是我建议你最终改成的文件名。
# aliases 是为了兼容你当前已经上传的中文文件名。
FILE_SPECS: List[FileSpec] = [
    FileSpec("walk_normal.csv", "normal_walk", "positive", ("正常步行(1).csv", "正常步行.csv")),
    FileSpec("walk_fast.csv", "fast_walk", "positive", ("快走_labeled.csv",)),
    FileSpec("jog_slow.csv", "slow_jog", "positive", ("慢跑_labeled.csv",)),
    FileSpec("run_fast.csv", "fast_run", "positive", ("快跑_labeled.csv",)),
    FileSpec("stairs_up.csv", "stairs_up", "positive", ("上楼_labeled.csv",)),
    FileSpec("stairs_down.csv", "stairs_down", "positive", ("下楼_labeled.csv",)),
    FileSpec("march_in_place.csv", "march_in_place", "positive", ("原地踏步_labeled.csv",)),

    FileSpec("stand_still.csv", "stand_still", "negative", ("静止站立.csv",)),
    FileSpec("sit_stand.csv", "sit_stand", "negative", ("坐下起立.csv",)),
    FileSpec("phone_shake.csv", "phone_shake", "negative", ("手机晃动.csv",)),

    # 以后补采坐车数据时，建议使用这个名字：
    FileSpec("vehicle.csv", "vehicle", "negative", ("坐车.csv", "公交.csv", "地铁.csv")),
]


def find_existing_file(input_dir: Path, spec: FileSpec) -> Path | None:
    """优先找 canonical_name，找不到就找 aliases。"""
    candidates = [spec.canonical_name, *spec.aliases]
    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p
    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少必要列：{missing}")

    df = df[REQUIRED_COLUMNS].copy()

    # 强制数值列转成数值，转失败会变成 NaN，后面体检会报出来。
    numeric_cols = ["seq_id", "time", *IMU_COLUMNS, "label"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def estimate_fs(time_values: np.ndarray) -> Tuple[float, float, float]:
    """返回 fs_est, median_dt, dt_std。"""
    time_values = np.asarray(time_values, dtype=float)
    diffs = np.diff(time_values)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    median_dt = float(np.median(diffs))
    dt_std = float(np.std(diffs))
    fs_est = float(1.0 / median_dt) if median_dt > 0 else float("nan")
    return fs_est, median_dt, dt_std


def build_quality_report(df: pd.DataFrame, path: Path, spec: FileSpec) -> Dict[str, object]:
    labels = df["label"].fillna(0).astype(int).to_numpy()
    time_arr = df["time"].to_numpy(dtype=float)
    fs_est, median_dt, dt_std = estimate_fs(time_arr)

    label_pos = np.where(labels == 1)[0]
    if len(label_pos) >= 2:
        intervals = np.diff(label_pos)
        min_interval = int(intervals.min())
        median_interval = float(np.median(intervals))
        max_interval = int(intervals.max())
    else:
        min_interval = None
        median_interval = None
        max_interval = None

    seq = df["seq_id"].to_numpy(dtype=float)
    seq_diff = np.diff(seq)
    seq_ok = bool(np.all(seq_diff == 1)) if len(seq_diff) else True

    time_diff = np.diff(time_arr)
    time_ok = bool(np.all(time_diff > 0)) if len(time_diff) else True

    allowed_labels = set(pd.Series(labels).dropna().unique().tolist())
    label_ok = allowed_labels.issubset({0, 1})

    duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) >= 2 else 0.0
    label_count = int((labels == 1).sum())
    steps_per_min = float(label_count / duration * 60.0) if duration > 0 else 0.0

    return {
        "source_file": path.name,
        "recommended_name": spec.canonical_name,
        "action": spec.action,
        "kind": spec.kind,
        "rows": int(len(df)),
        "duration_sec": round(duration, 3),
        "fs_est": round(fs_est, 3) if np.isfinite(fs_est) else None,
        "median_dt": round(median_dt, 5) if np.isfinite(median_dt) else None,
        "dt_std": round(dt_std, 5) if np.isfinite(dt_std) else None,
        "null_count": int(df.isna().sum().sum()),
        "seq_ok": seq_ok,
        "time_ok": time_ok,
        "label_ok": bool(label_ok),
        "label_1_count": label_count,
        "steps_per_min": round(steps_per_min, 2),
        "label_min_interval_frames": min_interval,
        "label_median_interval_frames": round(median_interval, 2) if median_interval is not None else None,
        "label_max_interval_frames": max_interval,
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """基础清洗：去空值、按 seq_id 排序、label 规整成 0/1。"""
    df = df.dropna(subset=["seq_id", "time", *IMU_COLUMNS, "label"]).copy()
    df = df.sort_values("seq_id").reset_index(drop=True)

    for col in IMU_COLUMNS:
        df[col] = df[col].astype(np.float32)

    # label 只保留 0/1。非 1 的都当 0。
    df["label"] = (df["label"].astype(float) == 1).astype(np.int64)
    df["time"] = df["time"].astype(np.float32)
    return df


def compute_features(df: pd.DataFrame) -> np.ndarray:
    raw = df[IMU_COLUMNS].to_numpy(dtype=np.float32)
    ax, ay, az, gx, gy, gz = [raw[:, i] for i in range(6)]
    acc_mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    gyro_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    features = np.column_stack([raw, acc_mag, gyro_mag]).astype(np.float32)
    return features


def lowpass_filter_features(features: np.ndarray, fs: float = FS_TARGET, cutoff: float = LOWPASS_CUTOFF) -> np.ndarray:
    """离线低通滤波。数据太短或没 scipy 时自动跳过。"""
    if not HAS_SCIPY:
        print("[WARN] 未安装 scipy，跳过低通滤波。建议安装：pip install scipy")
        return features.astype(np.float32)

    if len(features) < 20:
        return features.astype(np.float32)

    nyquist = fs / 2.0
    safe_cutoff = min(float(cutoff), nyquist * 0.85)
    sos = butter(N=2, Wn=safe_cutoff, btype="low", fs=fs, output="sos")

    try:
        filtered = sosfiltfilt(sos, features, axis=0)
        return filtered.astype(np.float32)
    except Exception as e:
        print(f"[WARN] 滤波失败，跳过滤波：{e}")
        return features.astype(np.float32)


def generate_gaussian_labels(hard_label: np.ndarray, sigma: int = GAUSSIAN_SIGMA) -> np.ndarray:
    hard_label = np.asarray(hard_label).astype(int)
    n = len(hard_label)
    soft = np.zeros(n, dtype=np.float32)
    peak_positions = np.where(hard_label == 1)[0]
    radius = int(5 * sigma)

    for pc in peak_positions:
        start = max(0, pc - radius)
        end = min(n, pc + radius + 1)
        t = np.arange(start, end)
        g = np.exp(-((t - pc) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
        soft[start:end] = np.maximum(soft[start:end], g)

    return soft


def create_windows(
    features: np.ndarray,
    soft_label: np.ndarray,
    hard_label: np.ndarray,
    time_arr: np.ndarray,
    source_file: str,
    action: str,
    kind: str,
    fs_est: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[Dict[str, object]]]:
    X_list: List[np.ndarray] = []
    Y_step_list: List[np.ndarray] = []
    Y_gait_list: List[int] = []
    meta_list: List[Dict[str, object]] = []

    n = len(features)
    if n < WINDOW_SIZE:
        return X_list, Y_step_list, Y_gait_list, meta_list

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        x = features[start:end]
        y_step = soft_label[start:end]
        hard_count = int(hard_label[start:end].sum())

        # gait 门控标签：
        # 正样本文件里，只要窗口内有真实步态事件，就认为这个窗口可计步。
        # 负样本文件里永远为 0。
        if kind == "positive" and hard_count >= 1:
            y_gait = 1
        else:
            y_gait = 0

        X_list.append(x)
        Y_step_list.append(y_step)
        Y_gait_list.append(y_gait)
        meta_list.append({
            "source_file": source_file,
            "action": action,
            "kind": kind,
            "start_frame": int(start),
            "end_frame": int(end - 1),
            "start_time": float(time_arr[start]),
            "end_time": float(time_arr[end - 1]),
            "hard_step_count_in_window": hard_count,
            "gait_label": int(y_gait),
            "fs_est": float(fs_est) if np.isfinite(fs_est) else None,
        })

    return X_list, Y_step_list, Y_gait_list, meta_list


def temporal_split_by_file(meta_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """每个文件内部按时间顺序切 train/val/test。"""
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, group in meta_df.groupby("source_file", sort=False):
        ids = group.index.to_numpy()
        n = len(ids)
        if n < 10:
            train_idx.extend(ids.tolist())
            continue

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        gap = SPLIT_GAP_WINDOWS if n > 50 else 0

        train_ids = ids[:max(0, train_end - gap)]
        val_ids = ids[min(n, train_end + gap):max(train_end + gap, val_end - gap)]
        test_ids = ids[min(n, val_end + gap):]

        # 如果因为文件太短导致某个集合为空，就退回无 gap 切分。
        if len(val_ids) == 0 or len(test_ids) == 0:
            train_ids = ids[:train_end]
            val_ids = ids[train_end:val_end]
            test_ids = ids[val_end:]

        train_idx.extend(train_ids.tolist())
        val_idx.extend(val_ids.tolist())
        test_idx.extend(test_ids.tolist())

    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }


def normalize_by_train(
    X: np.ndarray,
    splits: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    train_x = X[splits["train"]]
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.maximum(std, 1e-6)

    out = {}
    for name, idx in splits.items():
        out[name] = ((X[idx] - mean) / std).astype(np.float32)

    stats = {
        "feature_columns": FEATURE_COLUMNS,
        "mean": mean.astype(float).round(8).tolist(),
        "std": std.astype(float).round(8).tolist(),
        "fs_target": FS_TARGET,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "gaussian_sigma": GAUSSIAN_SIGMA,
        "lowpass_cutoff": LOWPASS_CUTOFF,
    }
    return out, stats


def save_outputs(
    output_dir: Path,
    X: np.ndarray,
    Y_step: np.ndarray,
    Y_gait: np.ndarray,
    meta_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = temporal_split_by_file(meta_df)
    X_norm_splits, norm_stats = normalize_by_train(X, splits)

    for split_name, idx in splits.items():
        np.save(output_dir / f"X_{split_name}.npy", X_norm_splits[split_name])
        np.save(output_dir / f"Y_step_{split_name}.npy", Y_step[idx].astype(np.float32))
        np.save(output_dir / f"Y_gait_{split_name}.npy", Y_gait[idx].astype(np.float32))

        meta_split = meta_df.iloc[idx].copy()
        meta_split.to_csv(output_dir / f"meta_{split_name}.csv", index=False, encoding="utf-8-sig")

    # 额外保存完整 meta 和报告
    meta_df.to_csv(output_dir / "meta_all.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(output_dir / "quality_report.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)

    summary = {
        "total_windows": int(len(X)),
        "train_windows": int(len(splits["train"])),
        "val_windows": int(len(splits["val"])),
        "test_windows": int(len(splits["test"])),
        "X_shape": list(X.shape),
        "Y_step_shape": list(Y_step.shape),
        "Y_gait_shape": list(Y_gait.shape),
        "positive_gait_windows": int(Y_gait.sum()),
        "negative_gait_windows": int(len(Y_gait) - Y_gait.sum()),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== 预处理完成 ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n输出目录：{output_dir.resolve()}")
    print("主要文件：")
    print("- X_train.npy / Y_step_train.npy / Y_gait_train.npy")
    print("- X_val.npy / Y_step_val.npy / Y_gait_val.npy")
    print("- X_test.npy / Y_step_test.npy / Y_gait_test.npy")
    print("- meta_all.csv")
    print("- quality_report.csv")
    print("- norm_stats.json")


def preprocess(input_dir: Path, output_dir: Path) -> None:
    all_X: List[np.ndarray] = []
    all_Y_step: List[np.ndarray] = []
    all_Y_gait: List[int] = []
    all_meta: List[Dict[str, object]] = []
    quality_rows: List[Dict[str, object]] = []

    used_files = set()

    for spec in FILE_SPECS:
        path = find_existing_file(input_dir, spec)
        if path is None:
            print(f"[INFO] 未找到 {spec.canonical_name}，跳过。")
            continue

        # 避免 正常步行.csv 和 正常步行(1).csv 同时存在时被重复处理。
        if path.resolve() in used_files:
            continue
        used_files.add(path.resolve())

        print(f"\n[READ] {path.name} -> action={spec.action}, kind={spec.kind}")
        df = read_csv_safely(path)
        quality = build_quality_report(df, path, spec)
        quality_rows.append(quality)

        df = clean_dataframe(df)
        fs_est, _, _ = estimate_fs(df["time"].to_numpy(dtype=float))

        features = compute_features(df)
        features = lowpass_filter_features(features, fs=FS_TARGET, cutoff=LOWPASS_CUTOFF)

        hard_label = df["label"].to_numpy(dtype=np.int64)
        soft_label = generate_gaussian_labels(hard_label, sigma=GAUSSIAN_SIGMA)
        time_arr = df["time"].to_numpy(dtype=np.float32)

        X_list, Y_step_list, Y_gait_list, meta_list = create_windows(
            features=features,
            soft_label=soft_label,
            hard_label=hard_label,
            time_arr=time_arr,
            source_file=path.name,
            action=spec.action,
            kind=spec.kind,
            fs_est=fs_est,
        )

        print(f"  rows={len(df)}, label_1={int(hard_label.sum())}, windows={len(X_list)}")

        all_X.extend(X_list)
        all_Y_step.extend(Y_step_list)
        all_Y_gait.extend(Y_gait_list)
        all_meta.extend(meta_list)

    if not all_X:
        raise RuntimeError("没有找到可处理的 CSV。请检查 input_dir 和文件名。")

    X = np.stack(all_X).astype(np.float32)
    Y_step = np.stack(all_Y_step).astype(np.float32)
    Y_gait = np.asarray(all_Y_gait, dtype=np.float32)
    meta_df = pd.DataFrame(all_meta)
    quality_df = pd.DataFrame(quality_rows)

    save_outputs(output_dir, X, Y_step, Y_gait, meta_df, quality_df)


def print_recommended_names() -> None:
    print("\n建议 CSV 文件名：")
    for spec in FILE_SPECS:
        alias_text = ", ".join(spec.aliases) if spec.aliases else "无"
        print(f"- {spec.canonical_name:20s}  action={spec.action:15s} kind={spec.kind:8s}  当前兼容名：{alias_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="25Hz IMU 计步数据预处理")
    parser.add_argument("--input_dir", type=str, default="./dataset_raw", help="原始 CSV 目录")
    parser.add_argument("--output_dir", type=str, default="./processed_dataset", help="输出目录")
    parser.add_argument("--print_names", action="store_true", help="只打印建议文件名，不处理数据")
    args = parser.parse_args()

    if args.print_names:
        print_recommended_names()
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print_recommended_names()
    preprocess(input_dir, output_dir)


if __name__ == "__main__":
    main()
