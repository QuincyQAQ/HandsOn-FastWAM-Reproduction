# HandsOn 双机 FastWAM 视频复现

在 HKUST 昇腾服务器上运行 FastWAM，在本机 Isaac Sim 中执行动作、采集主相机和双腕相机，再用 FFmpeg 写出视频。本仓库保存实验用的本地改动、调度脚本、配置，以及可与原始回合核对的 13 条主视角录像。

## 查看结果

[主视角视频](examples/videos)包括 12 个任务的首轮录像，以及一次成功的拉抽屉复测。[视频索引](examples/videos/index.json)给出每条录像的 SHA-256、原始目录和环境报告的成功状态。[逐回合记录](examples/results)保留 RESULT.json 或 ERROR.json。10 号抬纸箱在首轮执行异常，目录里保留了当时的短片。

## 还原工作目录

在包含本仓库的新父目录中执行以下命令，分别还原本机和服务器需要的源码。这些命令假设当前目录是本仓库的根目录，旁边的 HandsOn-restored 是新目录。

```bash
git clone https://github.com/TsinghuaC3I/HandsOn.git ../HandsOn-restored
git -C ../HandsOn-restored checkout --detach e357cb500709857bb92d86e22c4577af2118eb22
git clone https://github.com/isaac-sim/IsaacLab.git ../HandsOn-restored/third_party/IsaacLab-HandsOn
git -C ../HandsOn-restored/third_party/IsaacLab-HandsOn checkout --detach 37ddf626871758333d6ed89cf64ad702aef127d0
cp -a overlay/. ../HandsOn-restored/
```

构建此仓库时逐文件比较了 HandsOn 上游提交与实验目录。上游的 9,293 个文件中，9,283 个与本地文件内容相同，10 个修改文件均随 overlay 提供。IsaacLab 的本地 GUI 配置在同一个 overlay 中。附带的任务 YAML 也可直接浏览。

## 运行入口

| 实验 | 入口 | 历史服务配置 |
| --- | --- | --- |
| 12 项首轮，seed 0 | [run_care_batch.py](overlay/scripts/local/run_care_batch.py) | 远端 npu:9，SSH 转发 8631 |
| 首轮未成功项重试，seed 1 | [retry_care_tasks.py](overlay/scripts/local/retry_care_tasks.py) | 复用 8631 服务，读取首轮 SUMMARY.json |
| 单任务 | [run_fastwam_task.sh](overlay/run_fastwam_task.sh) | 默认本机回环 8631，可用 --policy-url 指定 |
| 归档条件下的 38 项试验 | [run_remaining_archived.py](overlay/归档实验/checks/run_remaining_archived.py) | 远端 npu:11，SSH 转发 8633，paper-trial 0 |

先按[双机配置记录](reference/RBMproject.MD)准备仿真环境和模型。记录中的本机路径、SSH 配置、远端用户目录、虚拟环境和逻辑设备号取自实验机。迁移机器时，先调整批处理中的远端目录及设备选择，同时为 SSH 配置提供当前用户的连接参数。模型文件和任务文本缓存由运行环境提供。FastWAM step-100000 权重的 SHA-256 为 `9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f`。

在还原目录中，如果原 8631 服务与 SSH 转发已就绪，可运行一个任务：

```bash
bash run_fastwam_task.sh --task 25 --seed 0
```

本机依次经过 [rollout.py](overlay/handson/handson/evaluation/rollout.py)、[cameras.py](overlay/handson/handson/evaluation/cameras.py) 和 [evidence.py](overlay/handson/handson/evaluation/evidence.py)，在新回合的 policy_000 目录中写出 main.mp4、wrist_left.mp4 和 wrist_right.mp4。推理服务入口为 [serve_paper_policy.py](overlay/scripts/serve_paper_policy.py)，FastWAM 适配器为 [fastwam.py](overlay/handson/handson/policies/adapters/fastwam.py)。早期单杯子实验使用端口 8630，步骤见[单回合实验记录](reference/docs/fastwam-reproduction-local-npu-zh.md)。

[LICENSE](LICENSE)、[NOTICE](NOTICE) 和[策略代码来源](reference/docs/policy-source-notices.md)记录上游及第三方组件的署名和许可。
