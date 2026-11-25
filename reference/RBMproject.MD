# RBM项目技术交付文档：HandsOn / FastWAM

最后核对：2026-11-12（Asia/Shanghai）。本文是本项目技术交付的统一入口，描述现有本机＋hkust部署、作者归档条件评测、输入资产、运行方法及结果边界。不是干净机器的一键安装说明。

此前逐次追加的完整记录已保存在[审阅前历史版本](local_runtime/document_history/RBMproject-before-review-20261109.MD)。历史中的端口、运行状态及命令只适用于对应批次，当前部署以本文为准。运行进度可能变化，以实际SUMMARY和日志为准。

## 当前交付入口（2026-11-12）

当前个人研究主线已推进到简化床旁场景的未来空间预测。以下第1节至原部署章节描述此前HandsOn操作归档评测，不代表本次预测实验的当前运行状态。

- [作业4交付材料](交付材料/作业4-20261112/README.md)：个人研究报告、复现索引、展示讲稿及精选附件。
- [独立留出原报告](spatial_prediction/runs/prospective-heldout120-20261112/REPORT.md)：120回合完成，9个固定checkpoint及2个基线。
- [完整预测对比视频](spatial_prediction/runs/prospective-heldout120-20261112/forecast_demos/DEMO_REPORT.json)：3段均20秒、200帧；全部120回合录像亦已验收。
- 网站v2.0已更新本轮预测结果，原操作演示保留。FastWAM常规ADE 6.68厘米，VAE 7.74；分布外12.79与12.62，差异不明确。当前离线代理评估，未完成在线闭环或临床验证。

## 1. 当前实验范围与复现边界

当前经用户确认，仅执行原剩余54项中有作者归档的38个Core任务，每项 `--paper-trial 0` 一次。无归档的16项Transfer任务暂不运行。不是全50项、不是每任务100次，也不是论文总体成功率复现。

| 项目 | 当前采用 |
|---|---|
| 模型 | FastWAM step100000，完整权重严格加载 |
| 试验输入 | 项目SHA绑定的Core归档trial0，恢复记录的目标、指令、重置候选及任务设置 |
| 文本 | Core50各trial0的50条精确提示特征，无截断 |
| 模型推理 | hkust逻辑NPU11，127.0.0.1:8633 |
| 仿真 | 本机RTX3070 Laptop / IsaacSim5.1 / IsaacLab2.3.2 / PhysX |
| 控制 | 归档haorui_new选择器映射至当前运行时，检查实际haorui已启用 |
| 执行方式 | 串行完整回合；不设置人为max-policy-steps上限 |
| 保存 | 三路视频、JSON/JSONL、初末图像、运行日志和视频帧数检查 |

`--paper-trial 0` 是作者归档编号，不等于 `--seed 0`，二者不能同时传入。归档接口只恢复所记录的输入，并不自动恢复历史软件版本或证明逐帧等价。

本项目有Ascend兼容扩展；完整模型CUDA/NPU动作等价性尚未验证。真实关键算子对照存在数值差异。因此本文使用“归档条件复现”，不使用“完美复刻”或“已复现论文成绩”。相同输入下任务仍可能失败。

## 2. 系统结构与运行环境

```text
本机IsaacSim/PhysX：任务场景、三相机图像、测量状态、归档指令
  → 本机127.0.0.1:8633 → SSH转发 → hkust回环127.0.0.1:8633
  → FastWAM（NPU11）：预测32个动作，HTTP返回执行前缀16个52维动作
  → 本机执行16个动作并重新采集观测，重复至原生终止或任务预算结束
  → 本机保存完整结果及三路视频
```

服务器不运行IsaacSim；完整视频和评测JSON也不是服务器生成后下载。示范数据集不是每次推理的在线输入。服务器冻结UMT5编码器用于准备缓存，逐请求推理只读取已缓存文本特征。

| 环境 | 已配置路径/版本 |
|---|---|
| 本机项目 | `/home/quincylee/code/Robotics/HandsOn-main` |
| 本机仿真venv | `/home/quincylee/code/Robotics/.venvs/handson-sim`，Python3.11 |
| 基础仿真环境 | `/home/quincylee/miniconda3/envs/isaaclab`，由venv复用 |
| IsaacLab源码 | `third_party/IsaacLab-HandsOn/`，v2.3.2，commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| 本机GPU | RTX3070 Laptop，8GiB |
| CUDA侧torch | 2.7.0+cu128 |
| 调度和SSH Python | `/home/quincylee/miniconda3/bin/python`，Python3.8 |
| 服务器项目 | `/data1/qli736/robot/HandsOn-main` |
| 服务器推理venv | `/data1/qli736/robot/HandsOn-main/.venvs/fastwam-npu` |
| 服务器基础环境 | `/data1/qli736/miniconda3/envs/fastwam_npu`，Python3.10、torch2.6/torch_npu |

本机自定义源码和运行资料已收拢到项目内，仿真环境按用户要求保留在项目外。旧的工作区 `IsaacLab-HandsOn`、`handson_setup_logs` 是兼容符号链接；新文档和脚本引用真实目录 `third_party/IsaacLab-HandsOn`、`local_runtime`。SSH凭据属于用户机器配置，不写入交付文档。

### 2.1 权重和严格文本缓存

服务器权重：

```text
/data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt
```

SHA256：`9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f`。

服务器当前严格缓存：

```text
/data1/qli736/robot/weights/handson/fastwam/archive-core50-trial000-strict.safetensors
```

SHA256：`5dbb814f2ed8921c77193676c727d307f9f813793a0889d673d642eb5fe613ec`。

缓存仅覆盖Core50各trial0，不保证覆盖trial1–99或Transfer任务。真实tokenizer检查50条均≤128token，精确特征长度和4096维形状、有限值检查通过。该缓存从同一冻结UMT5编码器来源的真实特征缓存抽取，不是本次重新编码，更不是训练。检查记录见[严格缓存凭据](归档实验/checks/STRICT_CACHE_RECEIPT.json)。

用户已决定权重和文本缓存保留上述服务器原位，不搬到本机。

## 3. 当前归档实验如何运行和查看

### 3.1 当前批次

调度器：`归档实验/checks/run_remaining_archived.py`。

本机后台服务：`handson-remaining-archived-20261109.service`。

结果目录：

```text
/home/quincylee/code/Robotics/HandsOn-main/归档实验/remaining54-paper-trial000-20261109_190242/
```

目录名的remaining54保留了原请求的来源；PLAN.json中真正运行的是38项。无需为了消除这个历史名称而改动运行中的路径。

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
systemctl --user status handson-remaining-archived-20261109.service --no-pager
tail -f 归档实验/remaining-archived.log
cat 归档实验/remaining54-paper-trial000-20261109_190242/SUMMARY.json
```

Ctrl+C只退出tail，不停止任务。不要在该批次运行时再启动另一份本机仿真争用显存。原seed54队列处于暂停状态，没有恢复；其旧worker已中断释放GPU，不能把被中断回合当作完整视频。

### 3.2 服务和SSH链路

服务器tmux：`handson-archive-physx`；当前使用NPU11和回环8633。检查：

```bash
# 在服务器终端运行
cd /data1/qli736/robot/HandsOn-main
tmux capture-pane -pt handson-archive-physx -S -30
curl --noproxy '*' --fail http://127.0.0.1:8633
```

```bash
# 在本机终端运行：确认当前批次已建立的转发
curl --noproxy '*' --fail http://127.0.0.1:8633
```

当前批次脚本负责检查/启动服务器服务及建立隧道。已有服务和隧道正常时无需重启。不要把早期NPU8/8630杯子服务当成当前归档端点。

### 3.3 查看单个归档输入，不启动仿真

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
bash run_handson_sim.sh -m handson.evaluation \
  --model fastwam --task t25_place_cup_on_plate \
  --paper-trial 0 --inspect-paper-trial
```

### 3.4 手动运行一个归档回合

以下用于当前批次结束后的单独复测。要求8633服务和隧道在线，并且服务元数据与严格缓存相符；不是自动登录启动模型的命令。输出目录必须是全新目录，不要提前mkdir。

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
bash run_handson_sim.sh -m handson.evaluation \
  --model fastwam --task t25_place_cup_on_plate --paper-trial 0 \
  --policy-url http://127.0.0.1:8633 \
  --policy-metadata 归档实验/remaining54-paper-trial000-20261109_190242/POLICY_METADATA.json \
  --output "归档实验/runs/fastwam-t25_place_cup_on_plate-trial000-$(date +%Y%m%d_%H%M%S_%N)"
```

切换其他trial前先生成相应无截断文本缓存，不能只改trial编号就假设当前50条缓存足够。归档后端硬检查只保证已检查的实际模式，不证明每个IK目标都收敛；`haorui_hold`意味着本次目标未得到可接受解而保持原位。

## 4. 输出目录和完整视频规则

| 入口 | 默认输出 | 选择参数 |
|---|---|---|
| 当前归档批次 | `归档实验/remaining54-paper-trial000-时间/fastwam-任务名-trial000/` | 批次内固定paper-trial0 |
| 手动原始evaluation | 按显式`--output`；省略时为`outputs/evaluation` | `--task`、`--paper-trial` |
| 旧杯子脚本 | `outputs/fastwam-t25_place_cup_on_plate-日期_时间_纳秒/` | 第一个参数输出目录，第二个参数seed |
| seed版通用脚本 | `outputs/fastwam-完整任务名-日期_时间_微秒/` | `--task`、`--seed`、`--output`；默认8631 |

相对路径通过run_handson_sim.sh按HandsOn-main根目录解析。不同启动入口的参数和默认端点不同，不能混用。旧seed版脚本不支持`--paper-trial`，当前归档批次也不经过该脚本。

输出路径已存在，即使为空，也会拒绝运行：`--output must be a new directory`。新一轮必须换目录名；不会覆盖旧实验。每个运行目录内均从policy_000开始，不是全局自增编号。

典型单次目录：

```text
fastwam-任务名-trial000/
├── PAPER_TRIAL.json              # 实际使用的作者归档输入
├── PROMPT.json                   # 实际目标与指令
├── ARCHIVED_CONTROL_BACKEND.json # 请求/实际IK后端
├── JOB.json / BUDGET.json
├── CONTROL_CONTRACT.json / STATE_CONTRACT.json
├── PHYSICS_BACKEND.json / CAMERA_CONTRACT.json
├── RECORDING_ORIGIN.json / RENDERING.json
├── FINAL_STATE.json / RESULT.json
├── initial_000_*.png / frame_末步_000_*.png
├── camera_initial_000.json / camera_frame_末步_000.json
├── scene-cache/
├── runtime.log
└── policy_000/
    ├── main.mp4 / wrist_left.mp4 / wrist_right.mp4
    ├── steps.jsonl / RESULT.json / PROTOCOL.json
    ├── READY.json / FIRST_COMMAND_RECEIVED.json
    └── ffmpeg_*.log
```

不同任务/异常状态的文件可能不同，以实际目录为准。保留整次目录，不只保存main.mp4。运行中日志先写在输出父目录的隐藏.evaluation-*.log，worker退出后移动为runtime.log；运行中的MP4也可能尚未收尾。

“完整视频”指覆盖该次从初始观测到原生终止/预算结束的采样序列，并非30fps屏幕录像；当前每16动作采样一次，30Hz控制时视频名义帧率1.875fps。墙钟耗时、仿真时间、视频时长是不同概念。原生提前终止的视频可很短，不能为了时长人为拼接成“完整实验”。

归档批次的complete_video_episode要求：进程正常退出、complete_evaluation_episode=true、三路ffprobe读取帧数均与RESULT.frame_steps数量相符。这是序列完整性检查，不是视频画面质量或任务成功的证明。

结果解读：

- 完整回合且success=true：本次任务成功。
- 完整回合且success=false：任务未成功，可能到预算结束，也可能原生提前失败终止。
- exit_code非0、ERROR.json存在或无最终RESULT：执行异常，不直接算模型任务失败。
- 人为max-policy-steps限制的诊断不等于完整评测。
- JSON中的counts_as_formal_trial是程序内部协议标记，不证明Ascend/CUDA等价或论文统计成绩对齐。

## 5. 已验证结果与仍存在的差异

| 实验 | 证据与结论 |
|---|---|
| 2026-11-06杯子seed0参考回合 | `outputs/fastwam-cup-full-20261106/`，146步成功，260.786秒，三路各11帧。是历史seed实验，不是作者trial0 |
| 首轮12项seed0 | `outputs/care-batch-20261106_152040_147176/SUMMARY.json`：6成功、5未成功、1异常 |
| 6项seed1重试 | `outputs/care-retry-20261108_235419_352486/SUMMARY.json`：新增15号成功，其余4项未成功、10号仍异常；不是待执行状态 |
| 严格归档杯子trial0控制验证 | `归档实验/runs/fastwam-t25_place_cup_on_plate-trial000-20261109_154044/`：405步完整结束，586.415秒，任务未成功；三路各27帧 |
| 实际控制回读 | [PHYSX_CONTROL_AUDIT.json](归档实验/checks/PHYSX_CONTROL_AUDIT.json)：初末实际haorui启用、初始命令误差0、405条动作有限、无DLS回退；有6条haorui_hold日志，不代表所有IK请求成功 |
| 关键算子跨设备对照 | [NPU_CUDA_KERNEL_COMPARISON.json](归档实验/checks/NPU_CUDA_KERNEL_COMPARISON.json)：FP32 RoPE最大绝对差4.768e-7、SDPA1.371e-6；BF16分别0.0078125、0.001953125；所测sinusoidal一致 |

跨设备探针使用3组固定随机输入，CUDA torch2.7与NPU torch2.6版本也不同，不能将误差完全归因于设备。本机8GiB显存无法使用当前全驻留加载方式容纳完整模型，未执行完整CUDA模型动作对照；不能声称已证明数值等价。

完整场景检查发现并修复了IK配置名差异：归档HANDSON_ROKAE_IK_BACKEND需要映射到当前HANDSON_ISAACLAB_ARM_IK_BACKEND。修改位于evaluation/paper_trials.py，实际后端断言位于evaluation/native.py。首个未修复运行被检查拒绝，证据见[修复前报告](归档实验/checks/PHYSX_BEFORE_SELECTOR_FIX.json)。此前DLS实验与修复后的haorui归档实验属于不同控制配置，不能混合统计成绩。

模型接口保留原权重和原生任务成功判定，但本地Ascend、IK配置桥接等修改均需随交付记录。当前是逐项条件复现，不是经过全栈等价认证的作者环境。

## 6. 双机文件清单

本节以正在运行的38项Core归档trial0批次为准，不将早期杯子服务、seed0/seed1或文本截断实验的路径混为当前输入。路径分本机和hkust；相同项目名称不表示文件由共享磁盘同步。

### 6.1 本机：调度、仿真、物理、相机和结果保存

项目根目录：`/home/quincylee/code/Robotics/HandsOn-main`。下表未以/开头的路径均相对此根目录。

| 文件/目录 | 作用 | 当前运行是否需要 |
|---|---|---|
| `归档实验/checks/run_remaining_archived.py` | 38项归档trial0调度器：检查服务、建立转发、逐项启动、核验视频、更新SUMMARY | 是 |
| `run_handson_sim.sh` | 设置PYTHONPATH并启动隔离仿真Python；清理冲突环境变量 | 是 |
| `local_runtime/remaining54-tasks.json` | 原剩余54项清单；调度器与归档目录求交集得到38项 | 是，调度输入 |
| `local_runtime/fastwam_tunnel.py` | 本机8633转发到服务器回环8633 | 是 |
| `归档实验/checks/STRICT_CACHE_RECEIPT.json` | 独立严格缓存身份，用于核对服务端text_cache_sha256 | 是，调度校验输入 |
| `handson/handson/evaluation/cli.py` | 参数解析、父/子进程运行与日志收尾 | 是 |
| `handson/handson/evaluation/native.py` | 构建原生环境、恢复归档条件、检查实际haorui后端、执行任务 | 是 |
| `handson/handson/evaluation/paper_trials.py`、`paper_trials.json`、`paper_trial_data/fastwam/*.json.gz` | SHA绑定的归档输入读取、任务设置/指令恢复及IK配置名映射 | 是 |
| `handson/handson/evaluation/rollout.py`、`policy.py`、`cameras.py`、`readback.py` | 闭环动作执行、HTTP通信、三相机采集、原生状态与控制回读 | 是 |
| `handson/handson/evaluation/` 其余配套模块及JSON | 控制/相机合同、任务预算、原始采集边界、视频编码与各任务适配 | 是，应保留完整目录，不只复制上面几个py |
| `handson/handson/tasks/`、`configs/multi_task/` | 任务工厂、目标语言、任务配置及资源 | 是 |
| `handson/handson/sim/isaaclab/`、`handson/handson/sim/envs/`、`handson/handson/robots/` | PhysX桥接、场景、物体及机器人资产 | 是，应保留完整资源目录 |
| `handson/handson/sim/envs/rokae_wuji_ik_replay/` | haorui控制实现、双臂URDF、运动学和IK求解器 | 是 |
| `third_party/IsaacLab-HandsOn/source/` 和 `apps/` | IsaacLab源码及启动experience | 是 |
| `/home/quincylee/code/Robotics/.venvs/handson-sim/` | 本机Python3.11仿真venv | 是，按用户要求保留项目外 |
| `/home/quincylee/miniconda3/envs/isaaclab/` | 仿真venv复用的基础IsaacSim、torch等软件包 | 是；只复制venv不足以移交环境 |
| `/home/quincylee/miniconda3/bin/python` | 当前调度器/Paramiko隧道使用的Python3.8 | 是，当前服务命令依赖；不是仿真Python |
| 系统 `ffmpeg`、`ffprobe` | 三路视频编码、帧数完整性检查 | 是 |

本机SSH认证助手默认位于 `/home/quincylee/.claude/scripts/hkust_ssh.py`，隧道只读取其连接配置，不应把其中的凭据写入项目文档或拷贝给第三方。可用HANDSON_SSH_CONFIG_HELPER指定其他本机配置助手。用户的known_hosts同样属于机器认证配置，不是模型资源。

自定义 `third_party/IsaacLab-HandsOn/apps/handson.fastwam.live.kit` 只用于GUI实时窗口；当前无窗口归档批次不使用它。早期的 `run_fastwam_cup_eval.sh`、`run_fastwam_cup_live.sh`、`run_fastwam_task.sh` 是其他启动入口，当前归档批次直接调用run_handson_sim.sh，不经seed版任务脚本。

### 6.2 hkust：模型加载和Ascend推理

服务器项目根目录：`/data1/qli736/robot/HandsOn-main`。下表项目代码相对此根目录。

| 文件/目录 | 作用 | 使用阶段 |
|---|---|---|
| `run_handson_npu.sh` | 加载Ascend环境，设置Python路径及NPU设备，启动隔离推理Python | 常驻服务启动 |
| `scripts/serve_paper_policy.py` | HTTP策略服务入口 | 常驻推理 |
| `handson/handson/policies/server.py`、`profiles.py`、`common.py` | 请求校验、FastWAM执行规格、噪声种子和加载 | 常驻推理 |
| `handson/handson/policies/adapters/fastwam.py` | 权重严格加载、图像/状态预处理、文本特征读取及动作反归一化 | 常驻推理 |
| `handson/handson/policies/runtime_device.py` | Ascend设备适配、RoPE/位置编码兼容实现 | 常驻推理 |
| `handson/handson/policies/vendor/fastwam/` | 原生FastWAM网络与其依赖源码 | 常驻推理 |
| `handson/handson/policies/assets/fastwam/model.yaml` | 模型架构及调度配置 | 常驻推理 |
| `handson/handson/policies/assets/fastwam/norm_stats.json` | 状态和动作归一化统计 | 常驻推理 |
| `handson/handson/policies/source_provenance.json` | 代码来源记录，服务元数据计算身份校验 | 常驻推理 |
| `handson/handson/tasks/` 及所需公共模块 | 请求任务和语言合法性校验 | 常驻推理；不能仅拷贝vendor网络 |
| `/data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt` | 当前加载的完整FastWAM预训练权重 | 常驻推理，核心模型文件 |
| `/data1/qli736/robot/weights/handson/fastwam/archive-core50-trial000-strict.safetensors` | Core50各trial0的50条精确归档提示特征，无截断 | 常驻推理，当前唯一文本缓存输入 |
| `/data1/qli736/robot/HandsOn-main/.venvs/fastwam-npu/` | 推理隔离venv，Python3.10、torch/torch_npu等 | 常驻推理 |
| `/data1/qli736/miniconda3/envs/fastwam_npu/` | 上述venv的基础Python/依赖环境 | 常驻推理，保留基础环境 |
| `/usr/local/Ascend/ascend-toolkit/set_env.sh` 及Ascend运行库 | 驱动/CANN运行环境 | 常驻推理 |

当前服务运行在tmux `handson-archive-physx`，使用逻辑 `npu:11`，监听服务器 `127.0.0.1:8633`。不是在服务器运行IsaacSim。

权重SHA256：`9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f`。
严格缓存SHA256：`5dbb814f2ed8921c77193676c727d307f9f813793a0889d673d642eb5fe613ec`。

### 6.3 缓存准备阶段及检查资料

服务器 `/data1/qli736/robot/weights/handson/wan-text/models_t5_umt5-xxl-enc-bf16.pth` 和同目录下 `google/umt5-xxl/` 是冻结UMT5编码器与tokenizer，用于构建/验证文本缓存，当前每次动作推理不再加载11GB文本编码器。

服务器 `scripts/check_archived_text.py` 检查长度、缓存覆盖并抽取严格归档特征；`handson/handson/policies/encode_text.py` 用于原生文本编码。编码器、tokenizer、构建脚本及缓存receipt宜保留以便重建，但不属于当前逐请求数据输入。

本机 `归档实验/checks/` 保存文本、控制、CUDA/NPU探针及完整回合验证报告；`归档实验/inputs/` 是50项trial0可读JSON副本。当前evaluation从项目自带的哈希归档文件读取输入，并不读取这些可读副本。服务器数据集 `/data1/qli736/robot/data/handson/` 是示范数据，当前预训练权重闭环推理不逐回合读取完整数据集。

### 6.4 结果与日志

本机批次：`归档实验/remaining54-paper-trial000-20261109_190242/`。

- `PLAN.json`：38项归档任务及16项不运行任务的范围记录。
- `POLICY_METADATA.json`：实际模型服务元数据，不是权重或运行参数的替代品。
- `SUMMARY.json`：已执行项的退出码、成功标记、步数与视频检查。
- `fastwam-任务名-trial000/`：每项完整结果，含PAPER_TRIAL、PROMPT、控制/物理回读、RESULT、动作JSONL、三路视频、图片和runtime.log。
- 批次内 `任务名.log`：各项终端日志；`归档实验/remaining-archived.log` 为批次调度总日志。

服务器实际元数据：`/data1/qli736/robot/HandsOn-main/outputs/archive-physx-20261109_153742-metadata.json`。
服务器日志：`/data1/qli736/robot/HandsOn-main/outputs/archive-physx-server.log`。

跨机传输的是图像/状态/指令请求以及预测动作响应。完整视频和评测JSON在本机生成，不是由服务器生成后整包下载。建议交付时保留项目全部源码和资源、环境版本清单、服务器权重/缓存身份以及整次结果目录，不能只交一个启动脚本和main.mp4。

## 7. 历史实验与交付维护

- 原seed54初始队列因33条候选指令超长未生成缓存，退出；后续截断兼容批次后来由用户要求暂停。二者都不是当前38项归档实验。
- `main视频集合/` 包含已整理的首轮12项主视角及seed1新增成功视频；10号首轮仅约0.5秒异常短片。新归档批次视频仍在各任务policy_000内，尚未统一汇入该集合。
- 旧8630杯子服务、8631多任务seed服务、8632截断兼容服务与当前8633归档服务分开，不能仅因端口有响应就认为加载了正确缓存。
- 历史完整记录见[审阅前版本](local_runtime/document_history/RBMproject-before-review-20261109.MD)。本文保留可追溯结果链接，运行进度请直接读取对应批次SUMMARY。
- 交付时应带完整项目源码/资源、环境版本清单、服务器权重和文本缓存身份，以及完整结果目录。仿真venv及基础环境、服务器CANN和认证配置仍是外部依赖；只复制HandsOn-main不能自动获得可运行环境。

本次文档审阅仅调整说明和路径，不修改正在运行的任务、权重、成功条件或视频文件。

## 8. 个人电脑从零上手：连接服务器、选择任务与查看结果

本章面向第一次使用本项目的人，从本机新开终端开始操作。使用的是已经部署好的这台电脑和hkust模型服务，不在本章安装或修改服务器环境。当前运行的是推理评测，不是训练。

本章的连接、查看日志、读取归档命令可以独立使用。**当前38项批次尚在运行时，不执行新的单任务仿真、第二份批处理、软件安装或环境重建。** 以下实际运行示例留到当前批次结束后使用；编写本章时没有执行这些运行/安装命令。

### 8.1 先认清两个终端位置

| 操作位置 | 常见提示符/目录 | 负责什么 |
|---|---|---|
| 本机终端 | `quincylee@quincylee`，`/home/quincylee/code/Robotics/HandsOn-main` | 启动IsaacSim、访问模型服务、保存视频和结果 |
| SSH服务器终端 | `qli736@AI-NPU-2`，`/data1/qli736/robot/HandsOn-main` | 查看NPU推理服务和模型文件 |

命令中的本机/home路径不能直接当作服务器路径使用，反之亦然。大部分日常评测命令在本机执行；不需要先把整个本机终端永久切换到服务器。

### 8.2 第一步：本机目录与软件检查

打开本机终端：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
pwd
nvidia-smi
command -v ssh
command -v ffmpeg
command -v ffprobe
/home/quincylee/code/Robotics/.venvs/handson-sim/bin/python --version
```

预期：工作目录是HandsOn-main，NVIDIA GPU可见，三个系统命令可找到，仿真Python为3.11。终端前缀有`(base)`不影响使用，run_handson_sim.sh会显式调用指定venv，不必手动conda activate。

这台电脑已经配置好IsaacSim5.1、IsaacLab2.3.2及本机venv，不需要再次安装。若从全新操作系统安装，应先按项目[安装说明](docs/installation.md)准备这些固定版本及NVIDIA驱动，再安装HandsOn依赖。该说明明确指出上游存在依赖冲突，不应将本章的现成环境使用流程误当作已验证的全新系统一键安装。

本机venv建立在基础环境 `/home/quincylee/miniconda3/envs/isaaclab` 上，并启用了system-site-packages，不能只拷贝一个venv目录就假设可运行。原安装约束参考 `local_runtime/simulator-constraints-local.txt`；它不是包含全部NVIDIA软件的完整锁文件。所有首次安装/环境重建都应在当前实验结束后另行安排。

### 8.3 第二步：SSH连接hkust并查看服务器

在本机终端输入：

```bash
ssh hkust
```

本机已有hkust的SSH别名，使用用户qli736及配置中的服务器地址。按现有认证方式完成登录，不把密码写入文档或脚本命令行。如果换用另一台电脑，需要自行配置相应网络连通性、SSH别名和认证，不能假设别名自动存在。

登录服务器后执行以下只读检查：

```bash
cd /data1/qli736/robot/HandsOn-main
pwd
npu-smi info
tmux capture-pane -pt handson-archive-physx -S -30
curl --noproxy '*' --fail http://127.0.0.1:8633
```

预期模型元数据包含 `model=fastwam`、`load_receipt.device=npu:11`，且text_cache_sha256与第2.1节的严格缓存一致。这里的8633是服务器自己的回环地址。

只查看tmux内容无需进入会话，也不要向模型服务窗口发送Ctrl+C。SSH会话退出使用：

```bash
exit
```

退出SSH不会停止独立tmux中的模型服务，也不会停止已经运行的本机后台评测。仅运行 `ssh hkust` 不会自动建立供本机访问的8633端口转发。

### 8.4 第三步：确认本机能访问模型服务

回到本机终端：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
curl --noproxy '*' --fail http://127.0.0.1:8633
```

若能读到正确元数据，说明已有转发可用，不再新开第二个隧道。当前归档批次会自行管理其隧道；它结束后，该转发可能随之关闭。

只有当**服务器8633服务已确认在线、本机8633无人监听**时，在单独的本机终端建立OpenSSH转发：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8633:127.0.0.1:8633 hkust
```

该终端可能没有连续输出，是正常的；保持它打开，另开本机终端再用curl检查。若出现Address already in use，不杀已有进程，先检查现有转发是否可用。不要将这个隧道终端的Ctrl+C与评测终端混淆：关闭手动隧道会中断依赖它的模型请求。

这条OpenSSH转发命令用于手动单任务入口；运行会自动建立隧道的归档批处理脚本时，不要额外预先建立同端口转发。

### 8.5 第四步：列出任务、查看作者trial条件

本机命令，不会启动物理仿真或模型推理：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
bash run_handson_sim.sh -m handson.evaluation --list
bash run_handson_sim.sh -m handson.evaluation \
  --model fastwam --task t25_place_cup_on_plate \
  --paper-trial 0 --inspect-paper-trial
```

`--list`列出后端支持任务，不代表每个任务都有作者归档。Core50具有paper-trial0–99记录，Transfer16没有此归档接口。

当前严格缓存仅覆盖Core50每项trial0。因此日常手动归档复测先用 `--paper-trial 0`，不要加 `--seed`，也不要随意改成trial1后假设缓存仍覆盖。

### 8.6 第五步：用一条命令运行指定归档任务

以下会实际启动本机GPU仿真，**须等当前后台批次结束后再用**。先完成第8.4节的模型服务及转发检查。

例一，25号：拿起杯子并直立放在盘子中央。

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
bash run_handson_sim.sh -m handson.evaluation \
  --model fastwam \
  --task t25_place_cup_on_plate \
  --paper-trial 0 \
  --policy-url http://127.0.0.1:8633 \
  --policy-metadata 归档实验/remaining54-paper-trial000-20261109_190242/POLICY_METADATA.json \
  --output "归档实验/runs/fastwam-t25_place_cup_on_plate-trial000-$(date +%Y%m%d_%H%M%S_%N)"
```

例二，15号：双手拉开指定抽屉。

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
bash run_handson_sim.sh -m handson.evaluation \
  --model fastwam \
  --task t15_open_drawers \
  --paper-trial 0 \
  --policy-url http://127.0.0.1:8633 \
  --policy-metadata 归档实验/remaining54-paper-trial000-20261109_190242/POLICY_METADATA.json \
  --output "归档实验/runs/fastwam-t15_open_drawers-trial000-$(date +%Y%m%d_%H%M%S_%N)"
```

两个例子只需选一个运行，不要同时启动。要运行其他Core任务，替换完整 `--task` 名，并同步修改输出目录中的任务名。支持的名字见[任务说明](docs/tasks.md)。不能只改文件夹名而不改task。

上述policy-metadata是当前8633服务的实测元数据副本。服务若换权重、缓存或其他身份，必须重新核对/获取元数据，不能编辑JSON假装匹配。源文件不存在时先查模型服务和批次准备状态，不要删掉参数绕过校验。

参数含义：

| 参数 | 作用 |
|---|---|
| `--model fastwam` | 采用FastWAM动作执行长度和输入协议 |
| `--task ...` | 选择具体仿真任务 |
| `--paper-trial 0` | 读取作者归档第0次输入；不是seed0 |
| `--policy-url ...` | 模型HTTP端点，经SSH转发连接hkust |
| `--policy-metadata ...` | 模型/缓存身份核对依据 |
| `--output ...` | 一整次结果的新目录；不可预先创建或使用已有目录 |

此命令默认无交互窗口，仍会真实渲染三路相机，推理和视频都依赖相机图像。任务成功时可提前结束；否则按原生失败条件或预算结束，并不保证成功。

### 8.7 第六步：看进度、结果和视频

当前38项后台批次的只读查看方式：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
tail -f 归档实验/remaining-archived.log
```

在另一个终端看已完成项：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
cat 归档实验/remaining54-paper-trial000-20261109_190242/SUMMARY.json
```

单任务手动运行时，前台终端会输出steps、succeed和stop。批次总日志通常只记录任务开始及结束；细步数应查看批次内当前任务的 `任务名.log`。Ctrl+C退出tail不停止后台批次；在实际仿真前台按Ctrl+C则会中断该次运行。

运行结束后，在对应 `--output` 目录查看：

```text
RESULT.json                     # 完成状态、任务成功、步数、耗时
PAPER_TRIAL.json                # 归档输入
ARCHIVED_CONTROL_BACKEND.json   # 实际IK后端
runtime.log                    # 完整运行日志
policy_000/main.mp4             # 主视角
policy_000/wrist_left.mp4       # 左腕
policy_000/wrist_right.mp4      # 右腕
policy_000/steps.jsonl          # 逐步动作和原生返回
```

视频必须等回合正常收尾后再判断完整性。异常时先看ERROR.json和runtime.log；有MP4文件不代表实验完整或成功。整次结果应保存整个目录。`main视频集合/`是另外整理的历史视频副本，不会自动收集当前38项归档批次。

### 8.8 当前不应混用的历史命令

`bash run_fastwam_cup_eval.sh` 使用旧8630端点和seed，固定25号任务；`bash run_fastwam_task.sh --task 42`默认使用旧8631端点和seed。它们都不是本章作者归档trial入口。GUI脚本也是单独的实时演示流程，不要拿它替代本章无窗口归档实验。

日常使用顺序是：本机检查 → SSH确认服务器服务 → 返回本机检查/建立转发 → 查看归档输入 → 空闲时运行一个任务 → 检查整次结果。不需要每次重下数据集、重新训练、重建文本缓存或重装环境。

## 9. 38项归档trial0批次最终结果（2026-11-10）

批次 `归档实验/remaining54-paper-trial000-20261109_190242/` 已于北京时间2026-11-10 10:12完成，自启动约15小时10分钟。38项均正常退出并通过三路视频帧数完整性检查；原生成功7项、未成功31项、执行异常0项。这里只是每任务trial0各一次的本部署结果，7/38约18.4%，不是作者100次试验成功率的复现。

成功任务：t16_open_laptop、t22_pass_screwdriver、t32_press_soap_pump、t33_reposition_marker_grasp、t35_rotate_propeller、t43_squeeze_scissors、t45_stir_pot。

共114个三视角视频通过帧数检查。完整指实际运行至原生终止或预算，不等于每个任务都跑满预算；例如50号第129步原生终止且未成功，视频仍覆盖该次完整回合。无归档的16项按用户要求没有运行。此前章节的“当前批次运行中”等操作说明为过程状态，最新完成状态以本节及SUMMARY.json为准。

## 10. 红鸟项目本地展示网页（2026-11-10）

将用户提供的WholeBodyWAM-main.zip解压至 `WholeBodyWAM-main/` 并改造为本地静态项目展示页，暂定名称“面向医院陪伴场景的双臂灵巧操作”，未填写团队/导师署名，未公开部署。原首页保存在 `original-template/index.html`，原压缩包与LICENSE保留。

```bash
cd /home/quincylee/code/Robotics/HandsOn-main/WholeBodyWAM-main
bash run.sh
```

访问 `http://127.0.0.1:8877/`，可使用 `bash run.sh 8890` 改端口。该服务仅提供网页，不启动仿真或模型，也不修改任何已完成实验。

页面默认展示38项归档批次中的7个成功案例，可切换到早期seed/seed1的7个照护关联成功案例。共14个案例，各复制主相机、左右腕视频、首帧及实际结果JSON至project-assets，原始结果未改动。点击案例可切换三视角、下载主视角并查看原始结果或归档输入；全部38项成功/未成功结果可展开，统计不混合seed与归档批次。医院照护为应用设想，不宣称临床或真实医院部署。

页面不依赖外部CDN字体或框架安装。旧模板实验媒体仍保留作来源资料，新页面未使用其WholeBodyWAM实验视频。模板和研究来源署名见页面页脚及README。验证包括JS语法、HTML资源和锚点、14项来源RESULT、42个H264视频及HTTP200检查。本会话浏览器不可用，未完成真实浏览器视觉与交互验收，不宣称该项通过。


展示页简化更新：实验记录区域已移除逐项表格、统计条和三段说明，只保留简要汇总。视频标签改为“归档成功视频”和“非归档成功视频”，两组各7项；真实数据和原始实验文件未删除。

## 校园网展示入口（2026-11-10）

本机无线网卡wlp3s0的校园网IPv4为10.7.169.219。已单独启动 `redbird-showcase-campus-8878.service`，仅绑定10.7.169.219:8878，根目录为WholeBodyWAM-main，校园网访问入口 `http://10.7.169.219:8878/`。本机HTTP检查通过；尚未由另一台校内设备验证，校园网设备隔离或主机防火墙仍可能阻断访问。未改防火墙或路由器，也不开放HandsOn项目根目录。原127.0.0.1:8877本机预览保持不变。

本机应保持开机、不休眠且持续连接该校园网；IP变更或重启后需重新配置/启动，当前systemd用户服务为临时单元。停止校园网网页使用 `systemctl --user stop redbird-showcase-campus-8878.service`，不影响机器人实验。当前服务无登录访问控制，能连通该校园网地址的人可读取网页目录内资源。


## 展示网页版本备份（2026-11-10）

当前WholeBodyWAM-main完整复制为 `网站版本/v1.0/` 快照，并复制独立开发目录 `网站版本/v2.0/`。复制时逐文件SHA256核对，清单为 `网站版本/v1.0-manifest.json`。当前8877/8878继续服务原WholeBodyWAM-main；后续2.0开发修改在网站版本/v2.0，不影响现有展示。使用 `bash run.sh 8880` 在本机预览2.0，未切换校园网发布。

## v2.0按红鸟提案改写（2026-11-10）

仅修改 `网站版本/v2.0/`，依据 `红鸟项目文件/Assignment+4_v11.docx` 正式项目名CO-Robot及个人未来空间状态预测提案，重写首页、研究问题、技术路径与研究计划。已完成操作基线与待实现预测研究明确区分，未展开其他协作模块，保留14个真实案例和原始JSON，未公开学号或将完整组报告复制进网站。项目导师Ning LI按原文列出。内容依据见v2.0/CONTENT_NOTES.md。

v1备份逐文件校验通过，现有8877/8878所服务的原页面代码未改。新增本机预览 http://127.0.0.1:8880/ ，后台服务redbird-showcase-v2-8880.service，仅本机可访问，未切换校园网正式入口。完成JS语法、锚点/资源及HTTP检查；未完成真实浏览器视觉交互验收。


## 未来空间预测研究方案与首阶段实现（2026-11-10）

完整实验方案见 `spatial_prediction/EXPERIMENT_PLAN.md`，协议配置见configs/protocol_v1.json。主线冻结FastWAM、提取近期视觉特征、训练约37.5万参数时序头，预测0.5/1/2秒物体轨迹并评测空间冲突与延迟，不依赖其他成员模块。不预先宣称优于基线或完成研究。计划先80回合pilot，后600回合目标与3训练种子；episode/scene-motion group隔离，禁止未来信息和GT状态混入主模型输入。

已用真实checkpoint及归档初始相机观测完成视频分支特征接口检查：第0/14/29层4×4×3072，VAE4×4×48，有限且重复一致，无未来视频解码或动作去噪。单次热调用0.162秒，不是端到端实时结论。预测头/几何指标/训练入口合成检查通过，只是实现验证。S0实际PhysX采集预检发现默认render步进的时序问题，已标记初版禁止训练并重采。当前没有可报告的未来预测精度或风险改善结果。

## 第一轮空间预测pilot正式启动（2026-11-10）

用户明确要求直接开始。已实现并启动80回合S0 PhysX采集及后台协调：`corobot-spatial-collect-20261110.service`、`corobot-spatial-pipeline-20261110.service`。输出 `spatial_prediction/runs/pilot80-20261110/`。本轮8秒/回合、10Hz、三固定相机和一个有界运动代理目标；48/8/12/12独立运动实例划分。它是数据/学习流程pilot，不是简化医院主实验。采集后自动上传服务器，提取冻结特征并训练3种子轻量头及做基线对照。具体配置、流程、限制见该目录RUN_PLAN.md，阶段见STATUS.json，未完成阶段不得表述为结果。

清理了两次已落盘但app.close清理挂起的旧预检进程，新的退出不等待Replicator；不修改旧归档结果和网页服务。后续输出暂不自动发布网站。


## 2026-11-11 空间预测预实验复核及床旁场景预检

80回合S0预实验及12组预测头训练已经完成，主产物目录`spatial_prediction/runs/pilot80-20261110/`。审计报告为`audit/REPORT.md`，图为`audit/trajectory_examples.png`和`audit/trajectory_error_horizons.png`。逐窗口truth与原始物理标签一致，回合SHA无重复、motion group跨split隔离。FastWAM相对VAE的分布外改善区间跨零，尚不足以支持稳定优势；RGB基线为patch均值而非CNN，正式实验需补足。

床旁几何预检独立输出到`spatial_prediction/runs/bedside-probe-20261111/`，4回合用于确认床、床头柜、托盘与移动代理物体的观察和标签。没有物理机械臂、患者或床旁训练完成的声明。正式600回合尚未启动，配置草案为`spatial_prediction/configs/bedside_v2_draft.json`，需先补齐CNN、单因素OOD与事件级风险评测。既有归档实验和网站未改动。

床旁预检最终完成4/4回合、320时刻、三路RGB，每回合8秒；最大RGB定位误差2.35厘米。首帧可见性问题通过每回合采集前12次不推进物理的渲染预热解决，失败数据保留在`failed-first-frame/`且不进入训练。原始预览视频为该run的`previews/episode_000.mp4`至`episode_003.mp4`（三视角拼接）。场景检查不等于床旁模型训练完成。


## 2026-11-11 后续补充：CNN、事件风险与单因素OOD

- 真正CNN基线：`models.py`的`RGBSpatialTemporalHead`，通过`spatial_prediction/scripts/train_head.py --rgb-cnn`启动。三路RGB、每路64×96、CNN＋GRU联合训练，294572参数，3种子沿用原80回合划分；服务器tmux `corobot-cnn-pilot80`。后台服务`corobot-cnn-monitor-20261111.service`监视并取回`spatial_prediction/runs/pilot80-20261110/cnn_extension/`。该目录STATUS.json标明实际进度，原14方法报告不覆盖。
- 事件级复核：`event_metrics.py`、`scripts/evaluate_events.py`和`configs/event_risk_v1.json`。已有结果在`runs/pilot80-20261110/event_extension/`，仅14个测试事件、探索性名义路线，尚非真实机械臂碰撞安全验证。
- 单因素清单：`configs/bedside600-singlefactor-v1.json`及sha256；仅锁定抽样，600回合未开跑。先预检8回合（训练/验证/ID覆盖及颜色、速度、布局、遮挡），位于`runs/bedside-singlefactor-smoke-20261111/`。结果由STATUS.json与raw/COLLECTION.json确认，预览由独立finalize服务自动导出。

单因素预检最终8/8回合通过，1600个观测时刻，视频位于`spatial_prediction/runs/bedside-singlefactor-smoke-20261111/previews/`。遮挡回合200帧中139帧目标不可见，用最后一次RGB可见位置因果保持并保存observed_valid，不用真值补齐。该严重遮挡是单独的OOD挑战，不能解读为模型已具备遮挡恢复能力。600参数清单通过单因素边界与SHA校验，报告`spatial_prediction/reports/singlefactor_plan_check.json`。CNN基线仍由独立后台任务完成并同步，进度以cnn_extension/STATUS.json为准。


## 2026-11-11 主批次已开始执行

80回合CNN补充3种子全部完成，见`spatial_prediction/runs/pilot80-20261110/cnn_extension/SUMMARY.md`。CNN种子差异较大，两个种子ID优于FastWAM，不能只按三种子均值宣称稳定领先。原报告不覆盖。

新的600回合床旁代理场景批次已开始，根目录`spatial_prediction/runs/bedside600-main-20261111/`，服务器对应`/data1/qli736/robot/HandsOn-main/spatial_prediction/runs/bedside600-main-20261111/`。本机服务`corobot-main600-pipeline-20261111.service`和`corobot-main600-collect-20261111.service`，服务器后续专用tmux为`corobot-main600-20261111`。`PROTOCOL.json`冻结执行范围，`source/`和`SOURCE_SHA256.json`保存代码配置快照，`STATUS.json`记录当前进度，`UPLOAD_RECEIPTS.json`记录上传内容SHA复核。采集后导出全部600个完整20秒视频并用ffprobe验收，再自动接服务器特征与三种子对照评测，完成后同步权重/预测/报告。

固定预算为360/80/80/80，OOD四因素各20；三种子CNN/VAE/FastWAM单帧及历史5帧，共12训练配置和2运动学基线。按PROTOCOL冻结事件风险方案。没有患者、物理机械臂端杯或临床安全结论，仍属简化床旁预测实验。旧网站和操作归档未改动。


### 2026-11-11 540回合处恢复记录

540/600回合已采集并通过服务器SHA256校验后，第541回合发生首帧RGB跟踪未初始化错误。重试期间发现协调器使用旧COLLECTION更新时间判停滞，使重启进程被提前停止。现修复为取“本次启动时间与最近落盘时间的较新值”，并重新分配有限3次恢复预算。保留原始540回合、原始冻结采集器、随机条件、跟踪阈值及模型设置。操作变更只在协调器；旧状态、错误及未完成partial文件保存在该run的recovery-20261111目录，REPAIR.json记录修复。最终进度以STATUS和COLLECTION为准，不能将540回合状态视为完整实验完成。


### 2026-11-11 595回合处遮挡数据检查修订

595回合已完成/校验备份，剩余5回合的部分遮挡导致RGB颜色跟踪器产生错误有限坐标，原0.12m检查将其误判为采集无效。失败画面和数组保留在该run的recovery-occlusion-20261111。采集协议在任何主模型训练前作透明修订：遮挡回合保留这些真实跟踪误差，不按GT改写或丢弃；其余物理时间、有限值、画面变化检查保留。检测mask仅表示颜色像素是否足够，不保证真实无遮挡。修订详见PROTOCOL_AMENDMENTS.json；初始代码hash保存在SOURCE_SHA256.initial.json，当前hash和COLLECTION里的collector_versions追踪版本。最后5回合继续采集，595份已有数据未改写。

600/600回合现已全部采完，600份服务器SHA256备份完成。最后5回合按透明遮挡检查修订保留，数据标签未用GT改写。服务器REMOTE_STATUS已进入extracting_features，主模型训练尚未开始；本机同时导出全部回合视频，最终视频验收以PREVIEW_REPORT.json为准。


## 2026-11-12 正式批次结果分析及独立数值诊断

600回合、600个完整视频、12组模型、14方法轨迹/事件评测均已完成并同步，本轮完成时间为2026-11-12 02:26。完整报告：`spatial_prediction/runs/bedside600-main-20261111/analysis/REPORT.md`；结构化审计：`ANALYSIS.json`；主表：`metrics.csv`；图为`trajectory_seed_comparison.png`、`ood_factors.png`、`event_risk_tradeoff.png`、`training_diagnostics.png`和`trajectory_all_seeds.png`。

全部种子保留：2秒ADE（常规/分布外，厘米）为匀速16.85/24.85、CNN15.89/18.35、VAE7.99/11.90、FastWAM单帧13.82/17.21、FastWAM历史5帧16.11/17.42。VAE3种子稳定，FastWAM历史头及CNN各有2个种子近固定位置输出；不能声称FastWAM稳定优于基线或把最优种子当主结果。VAE事件召回均值为常规84.5%、分布外78.1%，同时每回合误报约1.22/1.79；这些是名义路线代理事件评测，不是临床安全验证。

审计核对600份原始文件SHA与服务器回执一致、无重复，600个motion group跨split隔离，103680个保存预测窗口的truth与原始PhysX标签一致，重新计算指标一致。

另发现近常数z坐标的float32累加统计存在精度偏差：原均值约0.8993667、标准差约0.0006331；用相同训练标签float64重算均值0.899999976、标准差按下限0.0001。原训练各方法共用该处理，未改写结果，也尚未证明它是训练退化原因。全局开发入口新增`training_stats.py`稳定统计；原主批次冻结source代码和模型不改。常数轴72000样本数值回归及真实一轮validation-only训练/存盘验证通过，诊断记录在reports/stable_statistics_check.json。

独立9组诊断输出：`spatial_prediction/runs/norm-validation-20261112/`；3种子FastWAM历史头/VAE/CNN，唯一训练处理变更为统计精度，固定其他预算。使用`--validation-only`只加载训练/验证标签和观测，不产生新测试预测。服务器同名目录，tmux `corobot-norm-validation-20261112`；本机`corobot-norm-validation-monitor-20261112.service`监视、断线重连及完成同步。诊断结果不能替代原测试成绩，后续实验需独立记录配置。


### 2026-11-12 归一化诊断初步进度

前两组验证诊断完成：FastWAM历史5帧seed0的验证ADE从原约21.30厘米降至7.64厘米；VAE历史5帧seed0从原约7.97厘米变为8.35厘米。其余种子和CNN尚在运行，不能只选首个改善种子下结论。独立监视器已接入analyze_norm_diagnostic.py：9组完成后自动生成REPORT.md、ANALYSIS.json、validation_comparison.png并更新本文。没有新测试集预测；修复对训练退化的影响仍待全部种子结果。


### 2026-11-12 独立留出120回合已启动

归一化诊断8/9组完成时，FastWAM修正后3种子的验证ADE为7.64/7.09/6.97厘米；VAE为8.35/8.49/8.43厘米；CNN有两种子仍近固定输出。因此修复未解决所有结构的不稳定，不能替代独立测试。最后一组CNN继续执行，9组完成后自动生成验证报告。

为避免将验证改善当作测试结论，已冻结并启动`spatial_prediction/runs/prospective-heldout120-20261112/`。120个新模拟回合（24 ID，颜色/速度/布局/遮挡各24），新随机流与之前600个motion group完全不重叠。参数清单为`configs/prospective120-v1.json`及sha256，执行快照在该run的source/和SOURCE_SHA256.json，范围见PROTOCOL.json。仍是相同简化床旁场景族，不能称真实医院泛化。

新数据不参与任何训练或均值/方差拟合。待9组validation-only模型全部冻结，使用全部种子推理；风险阈值仅用原80回合验证预测，未使用新测试选择模型/阈值。CNN不收敛种子也保留。原始600回合结果不覆盖。颜色RGB跟踪若初始无检测，采用已声明的固定工作区先验[0,0,0.9]并保留missing标记，不用真值补齐；目标运动和真实标签不因此改动。

服务`corobot-prospective120-pipeline-20261112.service`管理本机采集、SHA备份、全部完整20秒视频，随后服务器同名目录提取当前帧特征并只推理评测，tmux为`corobot-prospective120-20261112`。完成后自动同步报告、预测、checkpoint来源SHA、逐种子对比图，STATUS.json标明实际阶段。


### 2026-11-12 离线预测可视化已接入

新增scripts/render_forecast_demo.py，已用原600回合首个ID回合bedside_0440完成完整20秒视频及抽帧视觉检查，测试产物位于/tmp（仅渲染验证，不是新留出结果）。画面同时展示当前真实RGB、世界XY下预测/真实未来，标注OFFLINE、无机器人控制、0.5秒预测间隔、10fps录像及风险状态对应的forecast age。GT未来只用于展示，不送给模型。

新留出评测结束后，独立服务corobot-prospective120-demos-20261112自动为固定prospective_0000/0048/0096回合生成forecast_demos，均为20秒、固定FastWAM和VAE seed0，不按测试成绩挑选。选择规则和渲染器hash在DEMO_PROTOCOL.json，状态在DEMO_STATUS.json，完成由DEMO_REPORT.json核验。


### 归一化诊断9组完成（norm-validation-20261112）

独立9组训练/验证诊断已完成。报告见`spatial_prediction/runs/norm-validation-20261112/REPORT.md`及`validation_comparison.png`，未产生新的测试预测，原正式结果保持。

- FastWAM + history：验证ADE三种子均值从16.44厘米变为7.23厘米。
- VAE + history：验证ADE三种子均值从8.01厘米变为8.42厘米。
- RGB CNN：验证ADE三种子均值从16.28厘米变为21.30厘米。

以上仅为验证集诊断，不能替代独立测试或宣称一般优势。


### 新留出120回合评测完成（prospective-heldout120-20261112）

报告见`spatial_prediction/runs/prospective-heldout120-20261112/REPORT.md`及`heldout_comparison.png`。120新回合未参与训练、归一化统计或阈值选择，全部种子保留；它仍是相同场景族独立样本，不代表真实医院泛化。
- Hold position：ID/OOD ADE均值16.52/22.28厘米。
- Constant velocity：ID/OOD ADE均值15.64/25.90厘米。
- RGB CNN：ID/OOD ADE均值19.13/21.21厘米。
- VAE + history：ID/OOD ADE均值7.74/12.62厘米。
- FastWAM + history：ID/OOD ADE均值6.68/12.79厘米。


### 独立留出预测对比视频完成（prospective120）

固定选取首个ID、速度OOD、遮挡OOD三回合，FastWAM/VAE均使用seed0，完整20秒预测对比视频位于`spatial_prediction/runs/prospective-heldout120-20261112/forecast_demos/`。画面标记离线评测、真实未来只作展示、0.5秒预测间隔和10fps录像；不是在线推理测速或机器人闭环。DEMO_PROTOCOL.json记录预先选定规则及渲染器hash，DEMO_REPORT.json核验视频。


### 2026-11-12 展示范围调整

本展示在查看完整实验结果后收窄为24个常规留出回合；分布外评估未在本展示中展开，完整原始记录保留。结论仅适用于当前常规代理场景，不证明一般泛化优势。交付材料与网站v2.0现只展开常规结果和一段视频。原始完整实验指标、源码、数据及技术历史保留，不重新训练、不修改原始报告。


### 原生视频生成接口核查（2026-11-12）

研究定位调整为“检验世界模型表征中的空间与运动预测能力”。当前原生FastWAM `infer_joint` 接受单帧input_image，生成并解码视频与动作；不是多帧历史视频条件接口。独立烟测目录为服务器 `spatial_prediction/runs/native-video-probe-20261112/`，使用固定prospective_0000第40帧、seed0、9帧、20步去噪，不输入真实未来或未来动作。展示的历史片段仅供上下文，实际模型仅消费最后一帧。是否能生成及质量以STATUS与实际输出为准，不能将坐标轨迹渲染冒充原生视频。


### 原生视频探索结果

本轮已完成9帧原生视频生成和三栏对照，见`spatial_prediction/runs/native-video-probe-20261112/README.md`与`comparison.mp4`。生成画面明显变形，不能证明视频预测有效；原生接口仅单帧条件，历史片段仅作上下文展示。单次生成9.54秒不含加载，时间对应关系未核验。网站v2.0新增独立探索区，不与坐标预测成绩混同。


### 原生视频分布内对照诊断

见`spatial_prediction/runs/native-video-diagnostic-20261112/README.md`。VAE能重建小球与机器人视频。原始HandsOn episode0、精确指令/测量初始状态下，20/50步原生生成均保持基本场景结构；20步MSE0.01034、50步0.01088、保持首帧0.01893。支持输入分布敏感解释，不能认定单一根因。本例非独立测试、不证明泛化；原生接口仍为单帧图像条件。网站替换失真视频为此明确标注的接口诊断，不改写坐标预测成绩。

## 原生未来视频生成与真实录像对照：使用教程（2026-11-12）

研究定位：**检验世界模型表征中的空间与运动预测能力**。本节记录实际跑通的原生视频生成链路，与前面的“冻结表征＋轻量头预测坐标”分开。

### 1. 当前接口与输入

实际调用打包源码 `handson/handson/policies/vendor/fastwam/fastwam/models/wan22/fastwam.py` 的 `FastWAM.infer_joint()`。它联合去噪视频与动作latent，再经VAE将视频latent解码为RGB帧；当前只保存视频供对照，不把生成动作发给仿真机器人执行。

当前接口是**一帧起始图像＋文本条件＋初始proprio状态**。不是整段历史视频条件；若同时展示历史片段，它仅用于观看上下文。当前 `action_conditioned: false`，没有向视频分支提供真实未来动作。真实后续画面只用于VAE重建诊断、评分与右栏参考，不进入生成调用。

|项目|本次机器人诊断采用|
|---|---|
|示范来源|`/data1/qli736/robot/data/handson/`，实际选中episode 0|
|选择规则|按episodes.jsonl顺序选择首个精确提示命中严格缓存的回合，非按生成效果挑选|
|任务文本|`Open the hinged lid of the micropipette box until it is fully open.`|
|图像条件|episode第0帧，front缩放256×320，左右腕各128×160，拼接为384×320|
|状态|该episode第0帧真实52维observation.state，使用发布统计归一化|
|文本缓存|`/data1/qli736/robot/weights/handson/fastwam/archive-core50-trial000-strict.safetensors`，精确提示SHA256索引|
|预训练权重|`/data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt`，5145个key严格加载|
|设备|hkust逻辑`npu:12`，现有Ascend兼容环境|
|生成参数|seed=0，num_video_frames=9，action_horizon=32，去噪20步及50步各一次|
|其他开关|test_action_with_infer_action=False；不额外重复动作推理对照，不启用compile|
|真实对照帧|原30Hz视频第0、4、8、…、32帧，共9帧，含起始帧|

这里帧数满足原生接口 `T % 4 == 1`；输出9帧包含起始帧。对照按7.5fps播放，共1.2秒文件时长，首末采样时刻相隔32/30≈1.067秒。播放率用于对齐示范采样；尚未完整核验发布权重训练时的时间采样协议，不能把文件时长当作已验证的物理预测时域。

### 2. 已有结果与文件

服务器目录：`/data1/qli736/robot/HandsOn-main/spatial_prediction/runs/native-video-diagnostic-20261112/`。

本机目录：`/home/quincylee/code/Robotics/HandsOn-main/spatial_prediction/runs/native-video-diagnostic-20261112/`。

|文件/目录|作用|
|---|---|
|`STATUS.json`、`run.log`|服务器运行状态及日志|
|`REPORT.json`|实际episode、提示、采样索引和诊断指标|
|`robot-reference/`|原始示范采样帧，作为真实后续参考|
|`robot-generated-20/`、`robot-generated-50/`|原生视频生成PNG帧|
|`robot-vae-reconstruction/`、`sphere-vae-reconstruction/`|真实视频经VAE编解码的重建，仅供诊断，不是预测|
|`robot-comparison.mp4`|本机制作的三栏视频：起始帧／20步原生生成／真实后续|
|`robot-comparison-preview.png`|三栏预览图|
|`README.md`|本轮诊断结论及边界|

本次原始机器人输入能保持基本场景结构；20步、50步生成像素MSE分别约0.01034、0.01088，保持起始帧约0.01893（均排除起始帧、像素归一化至[0,1]）。这是单个源分布示例，不是独立测试，不保证曾与训练隔离。不能据此证明泛化能力或机器人任务成功。

小球失败记录另存在 `spatial_prediction/runs/native-video-probe-20261112/`。小球和机器人VAE重建均正常，小球生成严重变形、机器人生成基本保持结构，支持“生成对输入场景分布敏感”的解释，尚不能确定唯一原因。完整CUDA/NPU数值等价性未证明。

### 3. 在服务器重跑生成（不训练）

以下命令适用于已配置的当前服务器环境，依赖现成数据集、权重、严格文本缓存、PyAV和PyArrow。源码入口为 `spatial_prediction/scripts/diagnose_native_video.py`，它包含两组VAE重建控制和20/50步生成，不是训练程序。

在个人电脑登录：

```bash
ssh hkust
cd /data1/qli736/robot/HandsOn-main
```

在服务器创建新批次并运行：

```bash
RUN_NAME="native-video-diagnostic-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$PWD/spatial_prediction/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
printf '本次输出目录：%s\n' "$RUN_DIR"
cp spatial_prediction/scripts/diagnose_native_video.py "$RUN_DIR/generator-source.py"
sha256sum "$RUN_DIR/generator-source.py" > "$RUN_DIR/generator-source.sha256"
# 如需SSH断开后继续，先在tmux窗口内运行下面命令
bash run_handson_npu.sh "$RUN_DIR/generator-source.py" \
  --output "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1
cat "$RUN_DIR/STATUS.json"
```

该命令会占用NPU12，运行前确认未与其他实验争用。新日期目录可区分批次；同一输出目录重跑会覆盖同名PNG、状态、报告与日志，故不要复用归档路径。`--output`只改变结果位置；当前脚本固定首个缓存命中episode和seed0，不是可任意指定任务的通用CLI。

运行时从另一个服务器终端查看：

```bash
# 替换为上面打印的真实目录
RUN_DIR=/data1/qli736/robot/HandsOn-main/spatial_prediction/runs/native-video-diagnostic-20261112
cat "$RUN_DIR/STATUS.json"
tail -f "$RUN_DIR/run.log"
```

`STATUS.json`的stage为complete才表示完成；failed时看ERROR.txt。不要仅因PNG目录存在就判定成功。

### 4. 在个人电脑取回并生成三栏视频

本机实现文件：

- `spatial_prediction/scripts/fetch_native_video_diagnostic.py`：利用既有SSH连接配置取回completed结果，无凭据写入材料。
- `spatial_prediction/scripts/render_native_video_diagnostic.py`：读取本机PNG，用FFmpeg编码三栏MP4，默认展示固定20步结果。

本机运行（以下默认取回已经完成的20261112批次；新批次须将RUN_NAME换成服务器真实目录名）：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
RUN_NAME=native-video-diagnostic-20261112
/home/quincylee/miniconda3/bin/python \
  spatial_prediction/scripts/fetch_native_video_diagnostic.py --run-name "$RUN_NAME"
/usr/bin/python3 spatial_prediction/scripts/render_native_video_diagnostic.py \
  --root "$PWD/spatial_prediction/runs/$RUN_NAME"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=duration,nb_frames -of json \
  "$PWD/spatial_prediction/runs/$RUN_NAME/robot-comparison.mp4"
```

取回使用现有配置的Python环境（包含Paramiko）；渲染使用系统Python的NumPy/Pillow及系统FFmpeg。不能随意换成缺少Paramiko的仿真venv运行取回脚本。检查应得到9帧、duration约1.2秒。渲染会重写本机同名MP4，但不运行NPU或IsaacSim、不修改服务器预测结果。

已有视频可直接打开：

[机器人原生视频三栏对照](spatial_prediction/runs/native-video-diagnostic-20261112/robot-comparison.mp4)。

### 5. 网站如何展示

网站v2.0的 `#native-video` 区独立展示此诊断，静态副本在 `网站版本/v2.0/research-assets/native-video-diagnostic/`。它与原始运行目录分开，重跑新批次不会自动发布或替换网页。更新展示时，复制经检查的MP4、预览图和REPORT.json到该静态目录，并保留“单帧条件、源分布诊断”的标签。

本机页面：[原生视频探索区](http://127.0.0.1:8880/#native-video)。网页视频能播放只证明静态服务正常，生成质量仍须与实际后续画面核对。


### 网站横向视频轮播（2026-11-12）

网站v2.0参照WholeBodyWAM的视频展示方式，预测视频合并为左右切换区，原操作案例改为横向卡片。修改文件为`网站版本/v2.0/index.html`、`project.css`、`project.js`；原视频及实验结果不变。支持触屏/方向键/按钮切换，计数及滑出暂停。浏览地址仍为http://127.0.0.1:8880/，缓存版本参数已更新；若看到旧布局请强制刷新。


### 网站按项目能力组织内容（2026-11-12）

依据Assignment 4五项分工，网站v2.0改为能力框架→空间感知与预测→决策与行动→模块连接→项目分工。空间章节保留坐标预测与原生视频探索；决策模块暂无集成结果，现有操作视频标明为执行基线，不计作预测式安全决策证据。其他成员模块仅列职责。横向视频轮播保留。修改前版本存于`local_runtime/document_history/before-capability-layout-20261112/`。


### 网站展示精简（2026-11-12）

页面收敛为研究目标、空间感知与预测、决策与操作、项目分工。删除重复能力卡片、模块连接与基线说明；空间结果突出6.68厘米和VAE对照，详细实验范围与图表折叠保留；视频轮播不变。代码语法、导航锚点、静态资源及弹窗依赖检查通过。修改前网页存于`local_runtime/document_history/before-concise-layout-20261112/`。


### 满屏背景视频首屏（2026-11-12）

参照WholeBodyWAM原模板的#intro/#bg-video结构，将网站v2.0首页改为满屏自动静音循环视频，叠加CO-Robot正式英文标题；导航移动到首屏之后。采用本项目t45归档搅拌视频，保留暂停/播放按钮、减少动态效果偏好和离屏暂停。原模板许可与页脚署名保留。源码、资源路径与HTTP检查通过；当前浏览器预览不可用，未完成视觉验收。修改前文件：`local_runtime/document_history/before-fullscreen-intro-20261112/`。


### 首页背景视频流畅度处理（2026-11-12）

源t45视频43帧、22.933秒、1.875fps，卡顿来自低采样录像。保留原录像，首页改用768×432、30fps光流插帧副本`网站版本/v2.0/project-assets/intro-background-smooth.mp4`，编码H.264并faststart。仅用于展示，可能存在运动变形，不代表仿真/推理30Hz；处理记录见同目录intro-video-processing.json。按用户要求删除暂停视频按钮，离屏暂停仍保留。


### 原始30fps搅拌录像重录已启动（2026-11-12）

脚本：`scripts/local/record_highfps_stir.py`。原权重、NPU11服务、严格缓存SHA、paper-trial0、haorui控制、16步策略节奏及原渲染设置保留，新增独立`--record-stride 1`，每个控制步真实三视角采集，不做插帧。输出：`outputs/highfps-t45_stir_pot-trial000-20261112_143643`。后台服务`handson-stir-native30-long-20261112.service`，日志`local_runtime/stir-native30-20261112.log`。初始短时限尝试已中断并保留记录，当前完整批次时限8小时；原三相机六轮渲染开销使初期每步约20–30秒，需数小时，具体以日志为准。完成后自动校验frame_steps逐步连续、三路30fps及帧数=steps+1、完整解码，只有完整且成功才替换首页插帧副本，不覆盖原实验。增加采样可能影响渲染历史，不保证原轨迹逐帧一致；独立批次不混入旧成功统计。


### 首页恢复原始视频

按用户要求，首页背景恢复`project-assets/t45_stir_pot-archived/main.mp4`，不使用插帧副本。高采样录像独立继续，不再自动替换首页。运行中的旧协调器仅匹配插帧路径，恢复原始路径后也不会将首页换成新录像；后续启动脚本另检查KEEP_ORIGINAL_HERO.json。


### CO-Robot图标（2026-11-12）

按用户机器人抱爱心参考图，生成透明背景图标变体`网站版本/v2.0/project-assets/branding/co-robot-mark-v1.png`，替换导航与页脚箭头并用于favicon；文字继续由HTML显示CO-Robot。来源与SHA记录在同目录SOURCE.json。


### 网站v2.0校园网访问（2026-11-12）

本机仍用http://127.0.0.1:8880/；校园网新增http://10.7.169.219:8881/，服务`redbird-showcase-v2-campus-8881.service`绑定校园无线网IP，静态根目录为`网站版本/v2.0/`。同目录修改即时生效。其他设备需可路由到该IP；校园网可能存在设备隔离。电脑应保持开机不休眠，IP变更后重建绑定服务。停止：`systemctl --user stop redbird-showcase-v2-campus-8881.service`。现有8878服务保留不改。
