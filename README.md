# Project MoE

面向 **MoE（Mixture-of-Experts）推理系统 + 运筹/随机网络建模** 的研究型仓库。

本仓库包含：

- 论文/模型推导（`Model/`）
- vLLM 代码考古与工程落地评估（`Documentation/` + `external/vllm/`）
- 可复现实验与统计分析（`Code/Verification/`）
- 过程记录与周任务（`Chat/`, `Task/`）

## 快速导航

### 📐 PriceMoE 模型

| 文档 | 说明 | 适用场景 |
|------|------|---------|
| `Model/PriceMoE_Final_Model_v2.1.md` | OR 版本数学模型（主线） | MS/OR 投稿、理论推导 |
| `Model/Model_Coding_Version.md` | 工程适配版本 | 合作对齐、代码实现 |
| `Model/Constraint Survey.md`（**deprecated**） | **初版**约束体系调研 | 理解约束选择的背景 |

### 🔧 vLLM 工程分析

| 文档 | 说明 |
|------|------|
| `Documentation/PriceMoE Implementation Feasibility Analysis.md` | **建议优先阅读** ⭐ 结合代码分析约束的实际意义与实现路径 |
| `Documentation/PriceMoE Model Constraint Overview.md` | **最新的约束调研**：列举选择/不选择每个约束的原因 |
| `Documentation/vLLM MoE Inference Walkthrough (v0.13.0).md` | vLLM MoE 推理流程全景梳理 |
| `Documentation/vLLM Scheduler schedule() 逐段解读.md` | Scheduler 核心函数逐行解析 |
| `Documentation/vLLM Distributed Parallelism Guide.md` | TP/PP/EP/DP 分布式并行指南 |
| `Documentation/Expert Replication in vLLM Analysis.md` | 专家复制机制分析 |

### 🔬 实验

| 实验 | 目录 | 说明 |
|------|------|------|
| E1.1 | `Code/Verification/E1.1/` | 多数据集观察 expert activation 分布（验证 arrival process） |

### 📚 Q&A 与讨论记录

| 文档 | 说明 |
|------|------|
| `Documentation/Q&A/20260128 vLLM & Model Review.md` | vLLM 代码与模型对齐问题讨论 |

## Folder Structure

### Chat

LLM Chats & Guidance for the project
Naming Convention: `yyddmm Model-name (Optional Topic)`

### Code

Code Repository
Including:

1. **Verification Experiments**：`Code/Verification/`
   - `E1.1/`：多数据集观察 expert activation/route 分布；包含 configs、脚本与分析代码

> 注：目前主要可复现实验集中在 `Code/Verification/E1.1/`。

### Documentation

Documentations for the Project

1. **工程与代码分析**
   - `vLLM MoE Inference Walkthrough (v0.13.0).md`
   - `vLLM Distributed Parallelism Guide.md`
   - `Expert Replication in vLLM Analysis.md`
   - `PriceMoE Implementation Feasibility Analysis.md`（建议优先阅读）
2. **实验记录**：`Documentation/Experiment Documentations/`
3. **学术参考**：`Documentation/Academic References/`

### Model

数学模型/推导与约束梳理：

- `Fluid_MoE*.md`：不同版本的 Fluid/MoE 建模迭代
- `Constraint Survey.md`：约束体系调研
- `PriceMoE_Final_Model_v2.1.md`：当前主线模型（投稿 MS 方向）

### Paper

Paper Repository for the Project

目前包含一个占位 `README.md`，用于承载论文写作的结构与材料汇编。

### Task

周任务与里程碑：`Task/`

### external

外部依赖（只读镜像/子模块式引用）：

- `external/vllm/`：vLLM 源码（用于代码考古、定位实现点、评估工程改动成本）

## 实验入口（E1.1）

实验目录：`Code/Verification/E1.1/`

- 配置：`Code/Verification/E1.1/configs/*.json`
- 运行脚本：`Code/Verification/E1.1/scripts/`
  - `run_from_config.sh`：按 config 执行一次实验
  - `run_two_domains_smoke.sh`：快速 smoke
- 输出：`Code/Verification/E1.1/outputs/runs/`
- 分析代码：`Code/Verification/E1.1/src/`（如 `analyze_run.py`, `collect_routing_stats.py`, `plot_routing.py`）

> 运行方式与参数说明以 `Code/Verification/E1.1/README.md` 为准。

## 版本与复现说明

- 本仓库包含的 vLLM 代码位于 `external/vllm/`，用于阅读与对齐论文/工程分析；
- 实验脚本可能依赖具体的运行环境（GPU、模型权重、数据集路径等）。

