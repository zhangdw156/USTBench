# **Urban Spatiotemporal Reasoning Benchmark**

This repository provides a comprehensive framework for **urban spatiotemporal reasoning** tasks. It integrates **large language models (LLMs)** into urban science problems.

It includes **spatial-temporal reasoning QA** tasks to evaluate LLMs' ability in understanding and reasoning over spatial-temporal data, and **downstream tasks** to evaluate LLMs' ability in solving real-world urban tasks.

---

## 🎉 News

**We are excited to announce that our work has been accepted to [The Fourteenth International Conference on Learning Representations (ICLR 2026)](https://iclr.cc/)!**

This benchmark provides a comprehensive evaluation framework for large language models in urban spatiotemporal reasoning, bridging the gap between AI capabilities and real-world urban science applications.

---

## 🔧 Lightweight vLLM QA workflow with uv

This fork is optimized for evaluating only the USTBench question-answering subsets
`st_understanding` and `planning` against an already deployed OpenAI-compatible
vLLM service. The uv environment intentionally excludes local model inference,
PyTorch, Transformers, CityFlow, geospatial downstream-task dependencies, and
W&B.

```bash
uv sync
bash scripts/prepare_eval_data.sh
```

By default, `scripts/prepare_eval_data.sh` downloads the customized dataset
`zhangdw/USTBench-ST-Planning-10pct` with `uv run hf download`, keeps a stable
cache under `${TMPDIR:-/tmp}/ustbench-data/` for resumable re-runs, and installs
only the USTBench-compatible QA tree into:

```text
data/
```

Use another Hugging Face dataset repo if needed:

```bash
bash scripts/prepare_eval_data.sh --repo <user-or-org>/<dataset-repo> --force
```

Then evaluate a vLLM-served model:

```bash
uv run python scripts/evaluate_qa_vllm.py \
  --model <served-model-name> \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --datasets "st_understanding,planning" \
  --batch-size 32
```

`--tasks auto` is the default: the evaluator discovers task folders present in
`data/`. Results are written to:

```text
results/
```

`results/` is git-ignored. To publish local results to the Hugging Face bucket
`zhangdw/leo-benchmark` under `USTBench/results`, run:

```bash
bash scripts/sync_results_to_hf.sh --dry-run
bash scripts/sync_results_to_hf.sh
```

The original full-benchmark dependency list remains in `requirements.txt`, but
it is not needed for this uv/vLLM QA workflow.

---

## 🚀 Running a Task

> Note: the uv workflow maintained by this fork is the vLLM QA path above. The original full-benchmark runner below may require the legacy dependencies in `requirements.txt`.

### General Format

```bash
python run_UST_tasks.py --task <task_name> \
                        --batch_size <int> \
                        --llm_path_or_name <llm_model_path_or_hub_name> \
                        [--use_reflection true(default)/false ] \
                        [--other_task_specific_args]
```

### View Required Arguments

Each task has its own required parameters. You can inspect them by running:

```bash
python run_UST_tasks.py --task <task_name> --help
```

---

## 🧠 Task Examples

### 🎯 vLLM QA evaluation

The maintained lightweight entrypoint is:

```bash
bash ./scripts/run_spatiotemporal_reasoning_evaluation.sh \
  --model <served-model-name> \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --batch-size 32
```

This wrapper calls `scripts/evaluate_qa_vllm.py` and evaluates only
`st_understanding` and `planning`.

---

### 🔍 Downstream Task Examples

Each downstream task has specific required arguments:

#### Socio-Economic Prediction

```bash
python run_UST_tasks.py --task socio_ecomic_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Guangzhou" \
                        --use_reflection true
```

#### Congestion Prediction

```bash
python run_UST_tasks.py --task congestion_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Beijing" \
                        --use_reflection true
```

#### Road Planning

```bash
python run_UST_tasks.py --task road_planning \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --slum_name "CapeTown1" \
                        --use_reflection true
```

#### Urban Planning

```bash
python run_UST_tasks.py --task urban_planning \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --cfg "hlg" \
                        --use_reflection true
```

#### POI Placement

```bash
python run_UST_tasks.py --task poi_placement \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Qiaonan" \
                        --use_reflection true
```

#### Traffic Signal Control

```bash
python run_UST_tasks.py --task traffic_signal_control \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --dataset "hangzhou" \
                        --traffic_file "anon_4_4_hangzhou_real.json" \
                        --use_reflection true
```

#### Traffic Flow Prediction

```bash
python run_UST_tasks.py --task traffic_od_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Newyork" \
                        --use_reflection true
```

#### Route Planning

```bash
python run_UST_tasks.py --task route_planning \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Manhattan" \
                        --use_reflection true
```

#### Human Mobility Prediction

```bash
python run_UST_tasks.py --task next_poi_prediction \
                        --batch_size 32 \
                        --llm_path_or_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
                        --location "Newyork" \
                        --use_reflection true
```

Or just run from our script for all downstream tasks:

```bash
bash ./scripts/run_downstream_tasks.sh
```

---

## 📌 Supported Tasks and Arguments

| Task Name                 | Required Arguments        |
| ------------------------- | ------------------------- |
| `question_answering`      | `tasks`, `datasets`       |
| `next_poi_prediction`     | `location`                |
| `poi_placement`           | `location`                |
| `congestion_prediction`   | `location`                |
| `route_planning`          | `location`                |
| `socio_ecomic_prediction` | `location`                |
| `traffic_signal_control`  | `dataset`, `traffic_file` |
| `traffic_od_prediction`   | `location`                |
| `road_planning`           | `slum_name`               |
| `urban_planning`          | `cfg`                     |

---

## 📎 Notes

* `--use_reflection`: Enables reflective reasoning if set to `true`.
* All models can be replaced with any HuggingFace-compatible LLM or local path to a fine-tuned model.
* You can easily extend new tasks by modifying `UST_tasks/` and updating `TASK_CONFIG` in `utils/task_config.py`.
