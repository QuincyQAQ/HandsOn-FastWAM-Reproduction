# FastWAM 单次闭环复现与输出规则（本机＋hkust）

记录日期：2026-11-06。本文记录这套实际工作区如何得到一次完整运行的所有结果，既包括视频，也包括 JSON、JSONL、图片和运行日志。不是训练教程，也不代表已复现论文全部任务或统计成功率。

## 1. 已成功的参考运行

任务：`t25_place_cup_on_plate`，seed：0。

本机结果目录：

```text
/home/quincylee/code/Robotics/HandsOn-main/outputs/fastwam-cup-full-20261106/
```

该目录的 `RESULT.json` 记录：

- `success=true`，原生环境 `reward=1.0`、`info.succeed=true`。
- `steps=146`，`stop_reason=native_terminated`。
- `complete_evaluation_episode=true`，本次预算为405步，提前成功终止。
- 回合耗时260.786秒，不含场景初始化；推理12.764秒，渲染232.373秒。
- 三路视频均正常编码，各11帧；主视角854×480。
- `counts_as_formal_trial=true`，同时 `full_alignment_accepted=false`。这是使用本地 Ascend 兼容扩展的一次成功闭环，不能等同于复现官方 CUDA 多种子评测分数。

视频按观测采样节奏输出，并非桌面屏幕的连续录像。该次主视频约5.87秒，不等于运行的墙钟时间。

## 2. 整套框架怎么连接

```text
本机 Isaac Sim 场景和物理仿真
  → 三路真实渲染的当前相机观测＋测量状态＋任务指令
  → 本机 127.0.0.1:8630 的 SSH 转发
  → hkust 127.0.0.1:8630 的 FastWAM 服务（Ascend npu:8）
  → 返回16个52维动作
  → 本机执行动作，重新采集观测，循环直到成功或预算结束
  → 本机输出完整回合文件
```

这不是播放数据集录像。服务器执行模型推理，本机RTX负责场景、相机和物理。已下载的示范数据不需要在每次推理时重新下载；纯评测本身不要求加载整套示范数据集。

## 3. 环境、资产与本地适配

| 部件 | 本次使用的配置或路径 |
|---|---|
| 本机项目 | `/home/quincylee/code/Robotics/HandsOn-main` |
| 本机GPU | RTX3070 Laptop，8GB |
| 仿真环境 | `/home/quincylee/code/Robotics/.venvs/handson-sim`，Python3.11 |
| 仿真软件 | Isaac Sim5.1，IsaacLab2.3.2 |
| IsaacLab源码 | `/home/quincylee/code/Robotics/IsaacLab-HandsOn` |
| IsaacLab提交 | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| 服务器项目 | `/data1/qli736/robot/HandsOn-main` |
| 推理环境 | `/data1/qli736/robot/HandsOn-main/.venvs/fastwam-npu` |
| 推理设备 | `npu:8` |
| 文本缓存 | `/data1/qli736/robot/weights/handson/fastwam/care-tasks.safetensors` |
| 本机服务元数据 | `/home/quincylee/code/Robotics/handson_setup_logs/fastwam-cup/policy-metadata.json` |

实际完成的配置步骤：

1. 本机建立独立仿真venv，复用已有IsaacLab环境的软件包；补齐IsaacLab源码的apps及扩展目录。
2. 使用项目自带机器人、物体、任务场景、控制和相机配置，先执行90步保持命令的诊断，验证物理和三路视频链路。
3. 服务器建立独立NPU推理venv，复用已有torch2.6/torch_npu；补齐sentencepiece、ftfy等依赖。此环境不等于官方CUDA依赖锁。
4. 通过 `handson/handson/policies/runtime_device.py` 和适配器启用NPU；位置编码相关运算采用兼容实现，NPU旋转位置编码使用float32。权重5145个键严格匹配。
5. 下载并校验冻结UMT5编码器和tokenizer，使用原生文本包装构建真实文本缓存，未使用替代文本特征。
6. 启动FastWAM HTTP服务，连接SSH转发，先验证真实相机输入能得到有限的 `[16,52]` 动作，再进行90步集成检查，最后运行完整回合。

以上是现有机器的实际配置记录，不是任意干净机器可直接执行的通用一键安装。详细安装历史见工作区 `handson_setup_logs/STATUS.md`；该文件按时间追加，后面的更正覆盖前面的状态。

权重文件：

```text
/data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt
```

权重SHA256：

```text
9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f
```

文本缓存SHA256：

```text
672b9f21df1a6f29e8aea86391516b7793a71ed8307b53207db5a0df2b11a3ba
```

缓存包含297条提示，覆盖t25、t26、t28的相关指令。已有缓存可以直接复用。只有缺失时才在服务器执行 `bash prepare_fastwam_cup.sh`；该脚本发现已有缓存会拒绝覆盖。冻结编码器在 `/data1/qli736/robot/weights/handson/wan-text/`，不是每次运行都重新编码或训练。

## 4. 在现有环境再生成一次完整结果

### 4.1 服务器推理服务

已有服务位于服务器tmux `handson-policy`，监听回环8630。保持运行时，无需为每次仿真重启或重新加载权重。

在服务器检查：

```bash
cd /data1/qli736/robot/HandsOn-main
tmux capture-pane -pt handson-policy -S -30
curl --noproxy '*' --fail http://127.0.0.1:8630
```

启动脚本是 `serve_fastwam_cup.sh`，内部调用 `run_handson_npu.sh`，绑定t25并加载上面的权重和文本缓存。只有确认原服务不在运行时才启动；元数据输出也有防覆盖检查，不能将重复启动报错理解为需要重下权重。

### 4.2 本机SSH转发和元数据

在本机检查已配置的转发服务：

```bash
systemctl --user status handson-fastwam-tunnel.service --no-pager
curl --noproxy '*' --fail http://127.0.0.1:8630
```

若服务单元仍存在但已停止，可执行 `systemctl --user start handson-fastwam-tunnel.service`。

若已有配置使用的临时单元已消失，可用原有转发程序重新创建：

```bash
systemd-run --user --unit=handson-fastwam-tunnel \
  --property=Restart=on-failure --property=RestartSec=5 \
  /home/quincylee/miniconda3/bin/python \
  /home/quincylee/code/Robotics/handson_setup_logs/fastwam_tunnel.py
```

这里复用本机已有SSH凭据配置，不在文档中写密码。如果你自己的OpenSSH已配置 `hkust` 别名，也可用 `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8630:127.0.0.1:8630 hkust` 替代现有转发；同一端口只运行一个转发。

本机 `handson_setup_logs/fastwam-cup/policy-metadata.json` 必须对应实际服务。这是模型身份、动作设置和权重校验记录；不是模型权重文件。

### 4.3 生成与参考运行同类型的完整输出（推荐）

在本机执行：

```bash
cd /home/quincylee/code/Robotics
bash HandsOn-main/run_fastwam_cup_eval.sh
```

默认不打开交互窗口，但仍进行真实相机渲染和模型闭环推理。默认seed0，执行完整任务预算，生成整套结果。

手动指定新目录和seed：

```bash
bash HandsOn-main/run_fastwam_cup_eval.sh outputs/fastwam-cup-repeat-001 0
```

上例实际生成到：

```text
/home/quincylee/code/Robotics/HandsOn-main/outputs/fastwam-cup-repeat-001/
```

若需要实时窗口，在本机桌面终端运行：

```bash
bash HandsOn-main/run_fastwam_cup_live.sh
```

GUI也生成整套回合文件。优先用eval脚本复现参考结果的执行方式。低采样或提前结束相机采集的实验模式没有验证为与参考运行等价，不要用其结果替代参考配置。相同seed不保证跨硬件、精度或代码版本逐帧相同，也不保证每次都成功。

## 5. 输出文件夹规则与覆盖行为

**普通第二次运行不会覆盖第一次。显式传入已经存在的目录会报错退出。**

规则直接来自 `run_fastwam_cup_eval.sh`、`run_handson_sim.sh` 和 `handson/handson/evaluation/cli.py`：

1. 不传第一个参数时，目录名为 `outputs/fastwam-t25_place_cup_on_plate-YYYYMMDD_HHMMSS_纳秒`。时间取启动命令所在本机的系统时间，包含纳秒字段，降低重复启动时重名的概率。
2. `run_handson_sim.sh` 先切换到项目根目录。所以通过这两个杯子启动脚本传入的相对路径，均相对于 `/home/quincylee/code/Robotics/HandsOn-main`，不相对于你打开终端的位置。
3. 第一个参数可使用绝对路径，直接保存到该位置；必须有写入权限，且目录尚不存在。
4. 第二个参数是seed，不是重复次数。当前脚本每次启动一次回合。
5. 输出目录只要已经存在，即使是空目录或上次失败留下的目录，也会拒绝运行：`--output must be a new directory`。不要提前 `mkdir` 最终运行目录。
6. 默认名称包含完整任务名和纳秒时间戳；若仍出现同名目录，会拒绝运行，不会覆盖或自动加后缀。
7. 每次新运行目录内都从 `policy_000` 开始。它不是所有运行共享的全局编号，第二次不会自动变成旧目录中的 `policy_001`。
8. 直接运行 `python -m handson.evaluation` 时，普通相对路径按该进程的当前目录解析；“固定相对于HandsOn-main”是上述shell启动脚本切换目录产生的行为。

示例：

```text
HandsOn-main/outputs/
├── fastwam-cup-full-20261106/       # 已保存的成功参考运行
│   ├── RESULT.json
│   └── policy_000/main.mp4
├── fastwam-cup-20261106_160001/     # 下一次自动命名的完整运行
│   ├── RESULT.json
│   └── policy_000/main.mp4
└── fastwam-cup-repeat-001/          # 手动指定名称的另一整次运行
    ├── RESULT.json
    └── policy_000/main.mp4
```

`fastwam-cup-full-20261106` 是当时手动指定的参考目录名，不是今后每次的固定输出路径。

需要更不容易碰撞的名字，可以使用纳秒时间戳（GNU date）：

```bash
cd /home/quincylee/code/Robotics
bash HandsOn-main/run_fastwam_cup_eval.sh "outputs/fastwam-cup-$(date +%Y%m%d_%H%M%S_%N)" 0
```

## 6. 单次结果里有哪些文件

以下来自已成功参考目录的实际清单。其他任务、诊断模式或后续版本可能增加文件；不同步数的末帧名称会变化。

| 文件或目录 | 用途 |
|---|---|
| `RESULT.json` | 整次运行最终结果：成功标记、步数、停止原因、耗时、视频信息、策略元数据及预算 |
| `policy_000/RESULT.json` | 本策略回合结果 |
| `policy_000/steps.jsonl` | 逐步执行记录，一行一个JSON；包含动作、请求编号和原生环境返回结果 |
| `policy_000/main.mp4` | 主相机视频 |
| `policy_000/wrist_left.mp4`、`wrist_right.mp4` | 左右腕相机视频 |
| `policy_000/ffmpeg_*.log` | 各路视频编码日志 |
| `policy_000/READY.json` | 策略准备就绪记录 |
| `policy_000/FIRST_COMMAND_RECEIVED.json` | 收到首个动作命令的记录 |
| `policy_000/PROTOCOL.json` | 本次策略执行协议 |
| `JOB.json`、`PROMPT.json`、`BUDGET.json` | 运行任务、实际语言指令、步数预算 |
| `CONTROL_CONTRACT.json`、`STATE_CONTRACT.json` | 控制与状态约定 |
| `CAMERA_CONTRACT.json`、`RECORDING_ORIGIN.json`、`RENDERING.json` | 相机、采集来源及渲染记录 |
| `ORIGIN_IMPLEMENTATION.json`、`ORIGIN_PARAMETERS.json`、`TRACE_IMPLEMENTATION.json` | 来源实现、参数及追踪记录 |
| `PHYSICS_BACKEND.json`、`CHECKER_LAYOUT_PATCH.json` | 物理后端与任务相关布局记录 |
| `FINAL_STATE.json` | 最终仿真状态 |
| `initial_000_*.png` | 初始三视角图片 |
| `frame_00146_000_*.png` | 本参考运行第146步的最终三视角图片；其他运行步数可能不同 |
| `camera_initial_000.json`、`camera_frame_00146_000.json` | 对应采集的相机位姿、内参等检查记录 |
| `scene-cache/` | 该次运行的场景缓存，保留以便追溯 |
| `runtime.log` | 仿真worker的完整终端日志 |

要保留一次完整结果，复制或归档整个运行目录，不要只保存 `main.mp4`。服务器常驻模型服务日志不在这个目录中：服务器日志为 `/data1/qli736/robot/HandsOn-main/outputs/fastwam-cup-server.log`。

启动及运行期间，父进程先将日志写到输出父目录下的隐藏文件 `.evaluation-随机字符.log`，worker退出后才移动成该次运行的 `runtime.log`。因此运行中还没看到 `runtime.log` 不一定是没有日志。

## 7. 如何判定一次运行真的完成

例如检查已有参考结果：

```bash
cd /home/quincylee/code/Robotics/HandsOn-main
python3 - <<'PY'
import json
from pathlib import Path
p = Path('outputs/fastwam-cup-full-20261106')
r = json.loads((p / 'RESULT.json').read_text())
for k in ('task', 'seed', 'success', 'steps', 'stop_reason',
          'complete_evaluation_episode', 'wall_seconds'):
    print(k, r.get(k))
print('ERROR.json exists:', (p / 'ERROR.json').exists())
for name in ('main', 'wrist_left', 'wrist_right'):
    print(name, (p / 'policy_000' / (name + '.mp4')).is_file())
PY
```

区分以下情况：

- `complete_evaluation_episode=true` 且 `success=true`：完成且任务成功。
- 完整执行到预算结束、`success=false`：完成了评测，但模型没有成功完成任务。
- 有 `ERROR.json` 或缺少最终结果：启动/执行未正常完成，需要看日志，不能直接当作模型任务失败。
- 人为设置 `--max-policy-steps 90` 的检查不是完整预算评测，不能拿它代替整次成功率结果。

参考运行使用原生任务判定，不是只凭最后一张图片或视频判断成功。再次运行会得到一套新结果，应检查它自己的JSON，而不能沿用参考目录的成功结论。


## 输出命名更新

默认杯子脚本及调用它的GUI脚本现在生成 `outputs/fastwam-t25_place_cup_on_plate-日期_时间_纳秒/`。旧结果目录不改名。显式传入第一个参数时仍完全使用指定目录名；任务当前仍固定为t25，不能仅靠改目录名切换任务。
