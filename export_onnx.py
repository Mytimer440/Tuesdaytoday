# -*- coding: utf-8 -*-
import os
from pathlib import Path

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class StepPeakNet25Hz(nn.Module):
    def __init__(self, input_channels: int = 8, dropout: float = 0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.tcn = nn.Sequential(
            TCNBlock(128, dilation=1, dropout=dropout),
            TCNBlock(128, dilation=2, dropout=dropout),
            TCNBlock(128, dilation=4, dropout=dropout),
            TCNBlock(128, dilation=8, dropout=dropout),
            TCNBlock(128, dilation=16, dropout=dropout),
        )

        self.step_head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=1),
        )

        self.gait_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # 输入: [B, 64, 8]
        x = x.permute(0, 2, 1)  # [B, 64, 8] -> [B, 8, 64]
        feat = self.cnn(x)
        feat = self.tcn(feat)
        step_logits = self.step_head(feat).squeeze(1)  # [B, 64]
        gait_logits = self.gait_head(feat).squeeze(1)  # [B]
        return step_logits, gait_logits


def load_state_dict_safely(model_path):
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 兼容 DataParallel 保存出来的 module.xxx
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    return cleaned


def main():
    root = Path(__file__).resolve().parent

    model_path = root / "model_run_v2" / "best_model.pth"
    onnx_path = root / "model_run_v2" / "steppeaknet_25hz.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型文件：{model_path}")

    print(f"读取 PyTorch 模型：{model_path}")

    model = StepPeakNet25Hz(input_channels=8, dropout=0.2)
    state_dict = load_state_dict_safely(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 64, 8, dtype=torch.float32)

    print(f"正在导出 ONNX：{onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["step_logits", "gait_logits"],
        opset_version=12,
        do_constant_folding=True,
    )

    print("ONNX 导出成功。")

    # 简单检查 PyTorch 输出
    with torch.no_grad():
        step_logits, gait_logits = model(dummy_input)
        print("PyTorch step_logits shape:", tuple(step_logits.shape))
        print("PyTorch gait_logits shape:", tuple(gait_logits.shape))

    # 可选：如果安装了 onnxruntime，就顺便对比 ONNX 输出
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_step, ort_gait = sess.run(None, {"input": dummy_input.numpy()})

        torch_step = step_logits.numpy()
        torch_gait = gait_logits.numpy()

        step_diff = np.max(np.abs(torch_step - ort_step))
        gait_diff = np.max(np.abs(torch_gait - ort_gait))

        print("ONNX step_logits shape:", ort_step.shape)
        print("ONNX gait_logits shape:", ort_gait.shape)
        print("step 最大误差:", step_diff)
        print("gait 最大误差:", gait_diff)

        if step_diff < 1e-4 and gait_diff < 1e-4:
            print("ONNX 与 PyTorch 输出基本一致。")
        else:
            print("ONNX 与 PyTorch 有差异，但不一定是错误，需要进一步检查。")

    except Exception as e:
        print("ONNX Runtime 对比检查跳过：", e)

    print("完成。")


if __name__ == "__main__":
    main()