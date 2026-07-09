# -*- coding: utf-8 -*-
"""
train_model.py

StepPeakNet-25Hz 训练脚本
用途：读取 preprocess.py 生成的 processed_dataset，训练 CNN-TCN 双头计步模型。

输入目录需要包含：
- X_train.npy / Y_step_train.npy / Y_gait_train.npy
- X_val.npy   / Y_step_val.npy   / Y_gait_val.npy
- X_test.npy  / Y_step_test.npy  / Y_gait_test.npy
- meta_test.csv

输出目录会生成：
- best_model.pth              最优模型
- last_model.pth              最后一轮模型
- training_history.csv        训练过程记录
- loss_curve.png              loss 曲线
- test_count_report.csv       测试集按动作计步报告
- run_config.json             本次训练配置

推荐先测试能不能跑：
python train_model.py --data_dir ./processed_dataset --out_dir ./model_run_test --epochs 2 --max_train_batches 30 --max_val_batches 10

正式训练：
python train_model.py --data_dir ./processed_dataset --out_dir ./model_run_full --epochs 50 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


# =========================
# 1. 固定随机种子，方便复现实验
# =========================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================
# 2. 数据集读取
# =========================

class StepDataset(Dataset):
    """读取 X、Y_step、Y_gait 三个 npy 文件。"""

    def __init__(self, x_path: Path, y_step_path: Path, y_gait_path: Path):
        if not x_path.exists():
            raise FileNotFoundError(f"找不到输入文件: {x_path}")
        if not y_step_path.exists():
            raise FileNotFoundError(f"找不到逐帧标签文件: {y_step_path}")
        if not y_gait_path.exists():
            raise FileNotFoundError(f"找不到门控标签文件: {y_gait_path}")

        # mmap_mode='r' 的好处：不一次性把所有数据复制进内存，更稳。
        self.X = np.load(x_path, mmap_mode="r")
        self.Y_step = np.load(y_step_path, mmap_mode="r")
        self.Y_gait = np.load(y_gait_path, mmap_mode="r")

        if len(self.X) != len(self.Y_step) or len(self.X) != len(self.Y_gait):
            raise ValueError(
                f"X/Y 数量不一致: X={len(self.X)}, "
                f"Y_step={len(self.Y_step)}, Y_gait={len(self.Y_gait)}"
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # X: [64, 8]
        # Y_step: [64]
        # Y_gait: []
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y_step = torch.tensor(self.Y_step[idx], dtype=torch.float32)
        y_gait = torch.tensor(self.Y_gait[idx], dtype=torch.float32)
        return x, y_step, y_gait


# =========================
# 3. 模型定义：CNN-TCN 双头模型
# =========================

class TCNBlock(nn.Module):
    """TCN 残差块。"""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StepPeakNet25Hz(nn.Module):
    """
    StepPeakNet-25Hz

    输入：
        x: [batch, 64, 8]

    输出：
        step_logits: [batch, 64]   每一帧是不是一步附近
        gait_logits: [batch]       当前窗口能不能计步
    """

    def __init__(self, input_channels: int = 8, dropout: float = 0.2):
        super().__init__()

        # CNN：先看局部波形
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # TCN：再看时间节奏
        self.tcn = nn.Sequential(
            TCNBlock(128, dilation=1, dropout=dropout),
            TCNBlock(128, dilation=2, dropout=dropout),
            TCNBlock(128, dilation=4, dropout=dropout),
            TCNBlock(128, dilation=8, dropout=dropout),
            TCNBlock(128, dilation=16, dropout=dropout),
        )

        # step_head：输出每一帧的计步概率 logits
        self.step_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        # gait_head：输出整个窗口是否可计步 logits
        self.gait_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 128, 64] -> [B, 128, 1]
            nn.Flatten(),             # [B, 128]
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, 64, 8] -> [B, 8, 64]
        x = x.permute(0, 2, 1)

        feat = self.cnn(x)       # [B, 128, 64]
        feat = self.tcn(feat)    # [B, 128, 64]

        step_logits = self.step_head(feat).squeeze(1)       # [B, 64]
        gait_logits = self.gait_head(feat).squeeze(1)       # [B]

        return step_logits, gait_logits


# =========================
# 4. 损失函数
# =========================

def step_weighted_mse_loss(
    step_logits: torch.Tensor,
    y_step: torch.Tensor,
    alpha: float = 4.0,
) -> torch.Tensor:
    """
    逐帧计步损失。

    为什么用加权 MSE：
    - y_step 大部分位置接近 0
    - 真正的步态峰很少
    - 不加权时模型可能学成“全输出 0”
    """
    step_prob = torch.sigmoid(step_logits)
    weight = 1.0 + alpha * y_step
    loss = weight * (step_prob - y_step) ** 2
    return loss.mean()


def total_loss_fn(
    step_logits: torch.Tensor,
    gait_logits: torch.Tensor,
    y_step: torch.Tensor,
    y_gait: torch.Tensor,
    step_alpha: float = 4.0,
    gait_weight: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """总损失 = 计步损失 + 0.2 * 门控损失。"""
    step_loss = step_weighted_mse_loss(step_logits, y_step, alpha=step_alpha)
    gait_loss = nn.functional.binary_cross_entropy_with_logits(gait_logits, y_gait)
    total = step_loss + gait_weight * gait_loss
    return total, step_loss, gait_loss


# =========================
# 5. 单轮训练 / 验证
# =========================

@dataclass
class EpochResult:
    total_loss: float
    step_loss: float
    gait_loss: float
    gait_acc: float


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step_alpha: float = 4.0,
    gait_weight: float = 0.2,
    max_batches: Optional[int] = None,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_losses: List[float] = []
    step_losses: List[float] = []
    gait_losses: List[float] = []
    correct = 0
    count = 0

    for batch_idx, (x, y_step, y_gait) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y_step = y_step.to(device)
        y_gait = y_gait.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            step_logits, gait_logits = model(x)
            loss, step_loss, gait_loss = total_loss_fn(
                step_logits,
                gait_logits,
                y_step,
                y_gait,
                step_alpha=step_alpha,
                gait_weight=gait_weight,
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_losses.append(float(loss.detach().cpu()))
        step_losses.append(float(step_loss.detach().cpu()))
        gait_losses.append(float(gait_loss.detach().cpu()))

        gait_prob = torch.sigmoid(gait_logits)
        pred = (gait_prob >= 0.5).float()
        correct += int((pred == y_gait).sum().detach().cpu())
        count += int(y_gait.numel())

    return EpochResult(
        total_loss=float(np.mean(total_losses)) if total_losses else math.nan,
        step_loss=float(np.mean(step_losses)) if step_losses else math.nan,
        gait_loss=float(np.mean(gait_losses)) if gait_losses else math.nan,
        gait_acc=float(correct / max(count, 1)),
    )


# =========================
# 6. 测试集：按文件拼接预测并计步
# =========================

def min_distance_by_action(action: str) -> int:
    """25Hz 下，不同动作的最小步间距。"""
    if action in {"fast_run", "slow_jog"}:
        return 5
    return 7


def detect_peaks_1d(prob: np.ndarray, height: float, distance: int) -> np.ndarray:
    """一维概率序列找峰。"""
    if not SCIPY_AVAILABLE:
        # 没有 scipy 时，用一个很简单的找峰备用方案。
        peaks = []
        last = -10_000
        for i in range(1, len(prob) - 1):
            if prob[i] >= height and prob[i] >= prob[i - 1] and prob[i] >= prob[i + 1]:
                if i - last >= distance:
                    peaks.append(i)
                    last = i
        return np.asarray(peaks, dtype=np.int64)

    peaks, _ = find_peaks(prob, height=height, distance=distance)
    return peaks


@torch.no_grad()
def predict_all(
    model: nn.Module,
    dataset: StepDataset,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对整个数据集预测，返回 step_prob、gait_prob、true_step。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    step_probs: List[np.ndarray] = []
    gait_probs: List[np.ndarray] = []
    true_steps: List[np.ndarray] = []

    for x, y_step, _ in loader:
        x = x.to(device)
        step_logits, gait_logits = model(x)
        step_prob = torch.sigmoid(step_logits).detach().cpu().numpy()
        gait_prob = torch.sigmoid(gait_logits).detach().cpu().numpy()

        step_probs.append(step_prob.astype(np.float32))
        gait_probs.append(gait_prob.astype(np.float32))
        true_steps.append(y_step.numpy().astype(np.float32))

    return (
        np.concatenate(step_probs, axis=0),
        np.concatenate(gait_probs, axis=0),
        np.concatenate(true_steps, axis=0),
    )


def build_count_report(
    meta_path: Path,
    pred_step: np.ndarray,
    pred_gait: np.ndarray,
    true_step: np.ndarray,
    out_path: Path,
    step_threshold: float = 0.5,
    gait_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    根据 meta_test.csv 把窗口预测拼回每个原始文件的时间线，然后找峰计步。

    注意：这是第一版离线评估，不等于最终实时计步逻辑。
    但它能帮助我们快速判断模型有没有学会。
    """
    if not meta_path.exists():
        print(f"[WARN] 找不到 {meta_path}，跳过测试集计步报告。")
        return pd.DataFrame()

    meta = pd.read_csv(meta_path)
    if len(meta) != len(pred_step):
        print(f"[WARN] meta 行数和预测数量不一致: meta={len(meta)}, pred={len(pred_step)}")
        n = min(len(meta), len(pred_step))
        meta = meta.iloc[:n].copy()
        pred_step = pred_step[:n]
        pred_gait = pred_gait[:n]
        true_step = true_step[:n]

    rows = []

    # 按来源文件分组评估
    for source_file, group in meta.groupby("source_file", sort=False):
        idxs = group.index.to_numpy()
        action = str(group["action"].iloc[0]) if "action" in group.columns else "unknown"
        kind = str(group["kind"].iloc[0]) if "kind" in group.columns else "unknown"

        starts = group["start_frame"].to_numpy(dtype=int)
        ends = group["end_frame"].to_numpy(dtype=int)
        min_start = int(starts.min())
        max_end = int(ends.max())
        length = max_end - min_start + 1

        pred_sum = np.zeros(length, dtype=np.float64)
        pred_count = np.zeros(length, dtype=np.float64)
        true_sum = np.zeros(length, dtype=np.float64)
        true_count = np.zeros(length, dtype=np.float64)
        gait_sum = np.zeros(length, dtype=np.float64)
        gait_count = np.zeros(length, dtype=np.float64)

        for row_i, sample_i in enumerate(idxs):
            start = int(starts[row_i]) - min_start
            end = int(ends[row_i]) - min_start + 1

            pred_sum[start:end] += pred_step[sample_i]
            pred_count[start:end] += 1
            true_sum[start:end] += true_step[sample_i]
            true_count[start:end] += 1
            gait_sum[start:end] += float(pred_gait[sample_i])
            gait_count[start:end] += 1

        valid = pred_count > 0
        p_pred = np.zeros(length, dtype=np.float32)
        p_true = np.zeros(length, dtype=np.float32)
        p_gait = np.zeros(length, dtype=np.float32)
        p_pred[valid] = (pred_sum[valid] / pred_count[valid]).astype(np.float32)
        p_true[valid] = (true_sum[valid] / true_count[valid]).astype(np.float32)
        p_gait[valid] = (gait_sum[valid] / gait_count[valid]).astype(np.float32)

        # 门控：如果模型认为当前不是可计步状态，就压掉 step 概率。
        p_pred_gated = p_pred.copy()
        p_pred_gated[p_gait < gait_threshold] = 0.0

        distance = min_distance_by_action(action)
        true_peaks = detect_peaks_1d(p_true, height=0.5, distance=distance)
        pred_peaks = detect_peaks_1d(p_pred_gated, height=step_threshold, distance=distance)

        true_count_n = int(len(true_peaks))
        pred_count_n = int(len(pred_peaks))

        if true_count_n > 0:
            error_rate = abs(pred_count_n - true_count_n) / true_count_n
        else:
            error_rate = np.nan

        rows.append({
            "source_file": source_file,
            "action": action,
            "kind": kind,
            "test_frames": int(valid.sum()),
            "true_steps": true_count_n,
            "pred_steps": pred_count_n,
            "abs_error": int(abs(pred_count_n - true_count_n)),
            "error_rate": float(error_rate) if np.isfinite(error_rate) else "",
            "mean_gait_prob": float(np.mean(p_gait[valid])) if valid.any() else 0.0,
            "step_threshold": step_threshold,
            "gait_threshold": gait_threshold,
            "distance": distance,
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False, encoding="utf-8-sig")
    return report


# =========================
# 7. 保存图表
# =========================

def save_loss_curve(history: pd.DataFrame, out_path: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] 当前环境没有 matplotlib，跳过 loss_curve.png。")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_total_loss"], label="train_total_loss")
    plt.plot(history["epoch"], history["val_total_loss"], label="val_total_loss")
    plt.plot(history["epoch"], history["train_step_loss"], label="train_step_loss", alpha=0.7)
    plt.plot(history["epoch"], history["val_step_loss"], label="val_step_loss", alpha=0.7)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("StepPeakNet-25Hz Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# 8. 主函数
# =========================

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    step_alpha: float = 4.0
    gait_weight: float = 0.2
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    step_threshold: float = 0.5
    gait_threshold: float = 0.5


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="训练 StepPeakNet-25Hz 计步模型")
    parser.add_argument("--data_dir", type=str, default="./processed_dataset", help="preprocess.py 输出的数据目录")
    parser.add_argument("--out_dir", type=str, default="./model_output", help="训练结果输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数。先测试建议 2，正式训练建议 50")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小。CPU 建议 32 或 64，显卡可 128")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--step_alpha", type=float, default=4.0, help="步态峰值损失加权系数")
    parser.add_argument("--gait_weight", type=float, default=0.2, help="gait 门控损失权重")
    parser.add_argument("--patience", type=int, default=8, help="早停轮数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=0, help="Windows/PyCharm 建议保持 0")
    parser.add_argument("--max_train_batches", type=int, default=None, help="只跑前 N 个训练 batch，用于快速测试")
    parser.add_argument("--max_val_batches", type=int, default=None, help="只跑前 N 个验证 batch，用于快速测试")
    parser.add_argument("--step_threshold", type=float, default=0.5, help="测试报告中找 step 峰的阈值")
    parser.add_argument("--gait_threshold", type=float, default=0.5, help="测试报告中 gait 门控阈值")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("========== StepPeakNet-25Hz 训练开始 ==========")
    print(f"数据目录: {data_dir.resolve()}")
    print(f"输出目录: {out_dir.resolve()}")

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"显卡名称: {torch.cuda.get_device_name(0)}")

    # 读取数据
    train_ds = StepDataset(
        data_dir / "X_train.npy",
        data_dir / "Y_step_train.npy",
        data_dir / "Y_gait_train.npy",
    )
    val_ds = StepDataset(
        data_dir / "X_val.npy",
        data_dir / "Y_step_val.npy",
        data_dir / "Y_gait_val.npy",
    )
    test_ds = StepDataset(
        data_dir / "X_test.npy",
        data_dir / "Y_step_test.npy",
        data_dir / "Y_gait_test.npy",
    )

    print(f"训练样本数: {len(train_ds)}")
    print(f"验证样本数: {len(val_ds)}")
    print(f"测试样本数: {len(test_ds)}")
    print(f"单个输入形状: {train_ds.X.shape[1:]}  # 应该是 (64, 8)")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = StepPeakNet25Hz(input_channels=8, dropout=cfg.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history_rows: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_res = run_one_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            step_alpha=cfg.step_alpha,
            gait_weight=cfg.gait_weight,
            max_batches=cfg.max_train_batches,
        )
        val_res = run_one_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            step_alpha=cfg.step_alpha,
            gait_weight=cfg.gait_weight,
            max_batches=cfg.max_val_batches,
        )

        scheduler.step(val_res.total_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_total_loss": train_res.total_loss,
            "train_step_loss": train_res.step_loss,
            "train_gait_loss": train_res.gait_loss,
            "train_gait_acc": train_res.gait_acc,
            "val_total_loss": val_res.total_loss,
            "val_step_loss": val_res.step_loss,
            "val_gait_loss": val_res.gait_loss,
            "val_gait_acc": val_res.gait_acc,
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_res.total_loss:.5f} "
            f"step={train_res.step_loss:.5f} gait={train_res.gait_loss:.5f} "
            f"gait_acc={train_res.gait_acc:.3f} | "
            f"val_loss={val_res.total_loss:.5f} "
            f"step={val_res.step_loss:.5f} gait={val_res.gait_loss:.5f} "
            f"gait_acc={val_res.gait_acc:.3f} | "
            f"lr={lr_now:.6f}"
        )

        # 保存最优模型
        if val_res.total_loss < best_val:
            best_val = val_res.total_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "model_name": "StepPeakNet-25Hz",
                    "input_shape": [64, 8],
                    "feature_names": [
                        "ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag"
                    ],
                },
                out_dir / "best_model.pth",
            )
            print(f"  -> 保存最优模型 best_model.pth，val_loss={best_val:.5f}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  -> 早停：验证集连续 {cfg.patience} 轮没有提升。")
                break

    # 保存最后模型
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "epoch": history_rows[-1]["epoch"] if history_rows else 0,
            "model_name": "StepPeakNet-25Hz",
            "input_shape": [64, 8],
        },
        out_dir / "last_model.pth",
    )

    # 保存训练历史
    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    save_loss_curve(history_df, out_dir / "loss_curve.png")

    print("\n========== 训练完成 ==========")
    print(f"最优 epoch: {best_epoch}")
    print(f"最优 val_loss: {best_val:.6f}")

    # 加载最优模型做测试报告
    best_path = out_dir / "best_model.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\n========== 生成测试集计步报告 ==========")
    pred_step, pred_gait, true_step = predict_all(
        model,
        test_ds,
        device,
        batch_size=max(cfg.batch_size, 128),
    )

    report = build_count_report(
        meta_path=data_dir / "meta_test.csv",
        pred_step=pred_step,
        pred_gait=pred_gait,
        true_step=true_step,
        out_path=out_dir / "test_count_report.csv",
        step_threshold=cfg.step_threshold,
        gait_threshold=cfg.gait_threshold,
    )

    if len(report) > 0:
        print(report.to_string(index=False))
        print(f"\n测试计步报告已保存: {(out_dir / 'test_count_report.csv').resolve()}")
    else:
        print("未生成测试计步报告。")

    print("\n主要输出文件：")
    print(f"- {out_dir / 'best_model.pth'}")
    print(f"- {out_dir / 'training_history.csv'}")
    print(f"- {out_dir / 'loss_curve.png'}")
    print(f"- {out_dir / 'test_count_report.csv'}")
    print("\n队长，看到 best_model.pth 就说明第一版模型已经训练出来了。")


if __name__ == "__main__":
    # Windows/PyCharm 下保持这个入口，DataLoader 才稳定。
    main()
