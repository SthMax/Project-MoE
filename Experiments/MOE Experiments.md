# [[Chat Log]]

# [[实验 1.1：多数据集观察 expert activation 分布（“arrival process 长什么样”）]]
# 实验 1.2：改变 top-k，观察 perplexity 与 benchmark（“质量-稀疏度曲线”）

# 实验 1.3：Queue-aware 的新 Top-K policy（不考虑通信），看 ppl/跑分/吞吐

  

# System Spec

4xA100 40G@Ampere

Driver Version: 580.95.05 CUDA Version: 13.0

  

  

# Environments

Python 3.12.12
VLLM 0.13.0 https://github.com/SthMax/vllm/tree/v0.13.0-MoE

  

  

  

## Models

Qwen3-Next-80B-A3B-FP8 https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8

512 Experts, 10 Activated, 1 Shared

Qwen3-30B-A3B-2507 https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507

128 Experts, 8 Activated

GPT-OSS 120b https://huggingface.co/openai/gpt-oss-120b

128 Experts, 4 Activated