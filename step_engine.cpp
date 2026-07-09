#include "pch.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>


// 这个 DLL 第一版只负责模型推理：
// 输入：已经标准化后的 64 * 8 float 数据
// 输出：64 个 step_prob + 1 个 gait_prob
//
// Python UI 仍然负责：
// 蓝牙接收、数据解析、norm_stats 标准化、64帧缓存、峰值判断、步数显示。

static std::unique_ptr<Ort::Env> g_env;
static std::unique_ptr<Ort::SessionOptions> g_session_options;
static std::unique_ptr<Ort::Session> g_session;
static std::mutex g_mutex;
static std::string g_last_error = "OK";


static float sigmoid_float(float x)
{
    if (x >= 0.0f)
    {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    else
    {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}


static void set_last_error(const std::string& msg)
{
    g_last_error = msg;
}


extern "C" __declspec(dllexport)
const char* get_last_error()
{
    return g_last_error.c_str();
}


extern "C" __declspec(dllexport)
int init_model(const wchar_t* onnx_path)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    try
    {
        if (onnx_path == nullptr)
        {
            set_last_error("onnx_path is null");
            return -1;
        }

        g_session.reset();
        g_session_options.reset();
        g_env.reset();

        g_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "step_engine");

        g_session_options = std::make_unique<Ort::SessionOptions>();
        g_session_options->SetIntraOpNumThreads(1);
        g_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        g_session = std::make_unique<Ort::Session>(
            *g_env,
            onnx_path,
            *g_session_options
        );

        set_last_error("OK");
        return 0;
    }
    catch (const Ort::Exception& e)
    {
        set_last_error(std::string("ONNX Runtime error: ") + e.what());
        return -2;
    }
    catch (const std::exception& e)
    {
        set_last_error(std::string("std error: ") + e.what());
        return -3;
    }
    catch (...)
    {
        set_last_error("unknown error in init_model");
        return -4;
    }
}


extern "C" __declspec(dllexport)
int run_model(
    const float* input_64x8,
    float* step_prob_out_64,
    float* gait_prob_out_1
)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    try
    {
        if (!g_session)
        {
            set_last_error("model not initialized");
            return -1;
        }

        if (input_64x8 == nullptr || step_prob_out_64 == nullptr || gait_prob_out_1 == nullptr)
        {
            set_last_error("null input or output pointer");
            return -2;
        }

        // 输入形状必须和导出 ONNX 时一致：[1, 64, 8]
        std::vector<int64_t> input_shape = { 1, 64, 8 };
        size_t input_tensor_size = 1 * 64 * 8;

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_64x8),
            input_tensor_size,
            input_shape.data(),
            input_shape.size()
        );

        const char* input_names[] = { "input" };
        const char* output_names[] = { "step_logits", "gait_logits" };

        auto output_tensors = g_session->Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );

        if (output_tensors.size() != 2)
        {
            set_last_error("unexpected output tensor count");
            return -3;
        }

        float* step_logits = output_tensors[0].GetTensorMutableData<float>();
        float* gait_logits = output_tensors[1].GetTensorMutableData<float>();

        // PyTorch 版里是 sigmoid(step_logits)，这里保持一致。
        for (int i = 0; i < 64; ++i)
        {
            step_prob_out_64[i] = sigmoid_float(step_logits[i]);
        }

        gait_prob_out_1[0] = sigmoid_float(gait_logits[0]);

        set_last_error("OK");
        return 0;
    }
    catch (const Ort::Exception& e)
    {
        set_last_error(std::string("ONNX Runtime error: ") + e.what());
        return -10;
    }
    catch (const std::exception& e)
    {
        set_last_error(std::string("std error: ") + e.what());
        return -11;
    }
    catch (...)
    {
        set_last_error("unknown error in run_model");
        return -12;
    }
}


extern "C" __declspec(dllexport)
void release_model()
{
    std::lock_guard<std::mutex> lock(g_mutex);

    g_session.reset();
    g_session_options.reset();
    g_env.reset();
    set_last_error("released");
}