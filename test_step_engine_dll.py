# -*- coding: utf-8 -*-
import os
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent

DLL_DIR = ROOT / "cpp_step_engine" / "step_engine" / "x64" / "Release"
DLL_PATH = DLL_DIR / "step_engine.dll"
ONNX_PATH = ROOT / "model_run_v2" / "steppeaknet_25hz.onnx"


def main():
    print("DLL 路径:", DLL_PATH)
    print("ONNX 路径:", ONNX_PATH)

    if not DLL_PATH.exists():
        raise FileNotFoundError(f"找不到 step_engine.dll: {DLL_PATH}")

    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"找不到 ONNX 模型: {ONNX_PATH}")

    # Python 3.8+ 加载 DLL 依赖时需要把 DLL 目录加入搜索路径
    os.add_dll_directory(str(DLL_DIR))

    dll = ctypes.CDLL(str(DLL_PATH))

    # const char* get_last_error()
    dll.get_last_error.argtypes = []
    dll.get_last_error.restype = ctypes.c_char_p

    # int init_model(const wchar_t* onnx_path)
    dll.init_model.argtypes = [ctypes.c_wchar_p]
    dll.init_model.restype = ctypes.c_int

    # int run_model(const float* input_64x8, float* step_prob_out_64, float* gait_prob_out_1)
    dll.run_model.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.run_model.restype = ctypes.c_int

    # void release_model()
    dll.release_model.argtypes = []
    dll.release_model.restype = None

    ret = dll.init_model(str(ONNX_PATH))
    print("init_model 返回:", ret)

    if ret != 0:
        err = dll.get_last_error()
        print("错误信息:", err.decode("utf-8", errors="ignore") if err else "无")
        return

    # 构造一组假的 64x8 输入，先验证 DLL 跑通
    x = np.random.randn(64, 8).astype(np.float32)
    step_out = np.zeros(64, dtype=np.float32)
    gait_out = np.zeros(1, dtype=np.float32)

    ret = dll.run_model(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        step_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gait_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    print("run_model 返回:", ret)

    if ret != 0:
        err = dll.get_last_error()
        print("错误信息:", err.decode("utf-8", errors="ignore") if err else "无")
        return

    print("step_out shape:", step_out.shape)
    print("gait_out:", gait_out)
    print("step_out 前10个:", step_out[:10])
    print("step_out 最大值:", float(step_out.max()))
    print("DLL 模型推理测试成功。")

    dll.release_model()


if __name__ == "__main__":
    main()