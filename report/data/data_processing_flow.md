# 数据来源和处理说明

## 1. 当前数据从哪里来

当前 Stage 1 训练输入不是作者私有完整数据集，而是本机可审计、可处理、可进入 `whole_body_tracking` 合同的数据：

```json
{
  "BeyondMimic Zenodo ablation reference CSV": 1,
  "HuB supplemental 29-DoF pkl": 8,
  "Unitree-retargeted LAFAN1": 40
}
```

总计：

- motion 数：`49`
- 总帧数：`448358`
- 总时长：`2.491` 小时

## 2. 为什么说“接近 2.5h”但不能说“就是论文 2.5h”

论文描述使用约 2.5 小时的多样动作训练 tracking teacher，并从中挑选代表性动作做真实机器人验证。但作者没有公开完整 curated motion list、teacher checkpoint、完整 DAgger rollout 或 diffusion training rollout。

本项目的 2.49h bundle 是公开/本地可用数据拼成的 trainable bundle，时长接近论文，但不保证动作选择、清洗规则、retargeting、joint order、质量筛选和作者完全一致。

## 3. Dataset_beyondmimic 的作用

`Dataset_beyondmimic/` 更适合做论文结果分析和 released-data 可视化，例如 GRF、IMU、ablation、rosbag/mcap replay。它不是完整 teacher/diffusion 主训练集。

可以用：

- `ablation/tkd_skill.csv`：36 列 generalized coordinate CSV，可作为 Stage 1 reference motion 或 replay reference。
- GRF/IMU/ablation CSV：用于复画论文曲线、做报告对照。
- MCAP/rosbag：用于 released real-robot state replay / visualization。

不能直接说：

- 所有 GRF CSV 都是可训练 reference motion；
- Zenodo 数据就是完整 2.5h training set；
- Zenodo 数据包含官方 teacher/VAE/diffusion checkpoint。

## 4. 当前处理流程

```text
原始/retargeted motion
  -> 检查列数、DoF、fps、joint order
  -> 生成/验证 G1 generalized coordinate motion
  -> 合并为 Stage 1 multi-source bundle
  -> 进入 whole_body_tracking PPO training
```

## 5. 证据路径

- `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`
- `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle_rows.tsv`
- `report/tables/dataset_inventory.csv`
- `report/tables/motion_duration_summary.csv`
