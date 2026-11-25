# HPC2 上的 FastWAM CUDA 推理与视频评测

默认把 FastWAM 放在 Slurm 分配的一张 NVIDIA A40 上，本机继续跑 HandsOn 的 Isaac Sim、PhysX 和三路相机。也支持在同一计算节点上跑完整回合。两种模式使用同一个 step-100000 checkpoint、一份与任务提示匹配的文本缓存，以及相同的评测入口。

## 1. 准备工作目录和环境

按仓库首页还原 HandsOn 源码并合入 overlay。本机与 HPC 工作树应使用相同版本。下文 shell 命令均从**还原后的 HandsOn 根目录**运行；将示例路径改为实际共享盘路径。

- **推理端**使用独立的 Python 3.11 环境。基础 Python 需要匹配计算节点的 Linux/glibc。安装上游 FastWAM 锁定依赖，由计算节点驱动提供 CUDA。模型在单张可见 GPU 上运行 BF16。
- **渲染端**需要 Isaac Sim 5.1、Isaac Lab 2.3.2、项目机器人和场景资产。已有的笔记本仿真环境可以复用。在执行回合的环境中安装 FFmpeg；在调用 hpc/run_episode.py 的机器也要有 ffmpeg 和 ffprobe。
- **权重**来自 Cyan666/HandsOn-Checkpoints 下的 fastwam/step-100000/mp_rank_00_model_states.pt。启动时程序逐字节校验固定 SHA-256。将模型和文本缓存放在计算节点可读的共享盘。下载和安装可在有外网的准备环境完成。

```bash
python3.11 -m venv "$PWD/.venvs/fastwam-cuda"
export FASTWAM_PYTHON="$PWD/.venvs/fastwam-cuda/bin/python"
"$FASTWAM_PYTHON" -m pip install -r requirements/policies/fastwam.txt
export FASTWAM_CHECKPOINT=/shared/models/fastwam/step-100000/mp_rank_00_model_states.pt
export FASTWAM_TEXT_CACHE=/shared/models/fastwam/care-tasks.safetensors
export HANDSON_ROOT="$PWD"
```

已有的 safetensors 提示缓存可直接复制；可用来源文件的 SHA-256 校对副本。若要为新任务建立缓存，在获配 GPU 的计算节点中运行下例，增加任务时重复传入 --task。归档 trial 应使用覆盖其确切指令的严格缓存。

```bash
PYTHONPATH="$PWD/handson" "$FASTWAM_PYTHON" -m handson.policies.encode_text \
  --model fastwam --device cuda:0 --task t25_place_cup_on_plate \
  --encoder /shared/models/wan-text/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer /shared/models/wan-text/google/umt5-xxl --output "$FASTWAM_TEXT_CACHE"
```

## 2. HPC 推理，本机渲染

在 HPC **登录 shell** 提交。2026-11-25 在 qli 的 HPC2 登录环境查到了 i64m1tga40u 分区；以下资源请求用默认账户通过了 sbatch --test-only。实际时间和可用资源由 Slurm 决定。

```bash
sinfo -h -p i64m1tga40u -o '%P %G %l'
sbatch --test-only -p i64m1tga40u hpc/fastwam.sbatch serve
sbatch -p i64m1tga40u hpc/fastwam.sbatch serve
```

作业请求 1 张 GPU、8 个 CPU、64 GiB 内存、4 小时。脚本保留 Slurm 的 GPU 掩码。模型在该掩码内使用逻辑 cuda:0。用 squeue -j JOBID 查状态，分配节点后查看 handson-JOBID.log。日志中的 READY 行给出计算节点、端口、元数据文件绝对路径。每次启动使用独立结果目录。

在本机第一个终端，将 NODE、PORT 替换成 READY 中的值。以下转发使用已有的 qli 登录节点别名，经跳板连接已获配计算节点的 SSH 服务，模型只监听该计算节点的回环地址。

```bash
NODE=gpuX-Y
PORT=23456
ssh -o ExitOnForwardFailure=yes -N -J qli \
  -L "127.0.0.1:18633:127.0.0.1:$PORT" "qli736@$NODE"
```

在本机第二个终端，从共享盘取回**本次作业**的 metadata，运行单任务。运行目录必须是新路径，脚本会核对实时元数据与这份文件、checkpoint 身份及 CUDA 设备。

```bash
mkdir -p outputs/hpc-client
# 用 READY 中的绝对路径替换冒号后的路径
scp qli:/absolute/job-directory/policy-metadata.json outputs/hpc-client/job-metadata.json
export HANDSON_SIM_PYTHON=/home/quincylee/code/Robotics/.venvs/handson-sim/bin/python
python3 hpc/run_episode.py --url http://127.0.0.1:18633 \
  --metadata outputs/hpc-client/job-metadata.json --task t25_place_cup_on_plate \
  --seed 0 --output "outputs/hpc-client/cup-$(date +%Y%m%d-%H%M%S)"
```

同一个服务可连续跑多个任务。下面读取历史首轮的 12 个任务，沿用 seed 0。文本缓存需先覆盖全部 12 项。这段命令按顺序执行，遇到运行错误会停止；环境正常结束但任务未成功时，仍保留视频并继续。

```bash
(
  set -e
  batch="outputs/hpc-client/batch-$(date +%Y%m%d-%H%M%S)"
  tasks=$(python3 -c 'import json; print(" ".join(x["task"] for x in json.load(open("outputs/care-batch-20261106_152040_147176/SUMMARY.json"))))')
  for task in $tasks; do
    python3 hpc/run_episode.py --url http://127.0.0.1:18633 \
      --metadata outputs/hpc-client/job-metadata.json --task "$task" --seed 0 \
      --output "$batch/$task"
  done
)
```

完成后在登录节点用 scancel JOBID 释放独立推理服务。若集群限制从跳板 SSH 到计算节点，可用下面的同节点模式。

## 3. HPC 同节点推理和渲染

该模式在同一个 allocation 内启动模型与 Isaac Sim，自动通过计算节点回环完成评测，作业结束即释放服务。两者共享一张卡；实际显存余量取决于任务场景。已有可用的仿真 Python 时：

```bash
export HANDSON_SIM_PYTHON=/shared/venvs/handson-sim/bin/python
sbatch -p i64m1tga40u hpc/fastwam.sbatch episode --task t25_place_cup_on_plate --seed 0
```

使用已准备的 Isaac Lab 2.3.2 Singularity 容器时，设置以下变量再执行同一条 sbatch。容器内需有 /isaac-sim/python.sh、Isaac Sim 5.1、FFmpeg 及兼容的 HandsOn 依赖。额外的 Python 包可放在还原目录的 .sim-packages/。容器运行器默认使用在 HPC2 查到的 Singularity CE 4.1.3 路径，也可通过 HANDSON_CONTAINER_RUNTIME 指定。代码和输出以宿主机同名绝对路径挂载，Kit 缓存、home 和 /tmp 按作业隔离。

```bash
export HANDSON_SIM_PYTHON="$PWD/hpc/sim_container.sh"
export HANDSON_SIM_IMAGE=/shared/containers/isaac-lab_2.3.2_unpacked
```

重跑归档条件时，以 --paper-trial 0 替换 --seed 0。输出位于日志所示的 job 目录。policy.log 保存模型启动错误。episode/RESULT.json 给出任务成功状态，episode/policy_000/ 保存 main.mp4、wrist_left.mp4 和 wrist_right.mp4。验证器逐路解码，并将帧数和分辨率与回合记录比对。

## 验证范围

发布时验证了 Slurm 资源请求、入口参数、模拟服务的启动与清理，以及已有三路录像的解码。实际 A40 上的模型加载、GPU 显存容量和新回合视频仍需在准备好权重、缓存和仿真依赖后运行确认。
