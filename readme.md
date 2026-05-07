# Predicting F1 Pit Stops

## 1. Tire compound changes during a race. Compounds determine the tire's "Personality".
轮胎配方（tire compound）在比赛过程中可以发生变化，并且会显著影响轮胎的“性格”，包括 **grip（抓地力）**、**durability（耐久性）**、**temperature resistance（温度适应性）** 和 **degradation（磨损/衰减）**。

轮胎配方采用从 **C0 到 C6** 的编码体系，其中 **C 表示 Compound**：

* **C0 / C1（hardest）**：durability 最高，适用于高磨损或高温赛道
* **C3（medium）**：作为“workhorse”，在 **speed** 和 **durability** 之间取得平衡
* **C5 / C6（softest）**：提供最高的 **grip** 和单圈速度，但 **degradation** 很高

轮胎颜色与类型对应关系如下：

* **Red = soft**
* **Yellow = medium**
* **White = hard**
* **Green = intermediate**
* **Blue = wet**

此外，在比赛规则方面：

* 在 **dry race** 中，每辆车必须使用至少两种不同的 **tire compounds**
* 在 **wet race** 中，则不需要满足这一要求

👉 从特征角度来看，**compound**、**compound_type（soft/medium/hard）**、以及 **degradation** 都是关键变量，并且会随着比赛过程动态变化。

Reference: https://www.kaggle.com/competitions/playground-series-s6e5/discussion/696012

## 2. Inconsistencies in the "original" dataset
`original` dataset 虽然声称基于 FastF1 数据，但其中一些 derived features 存在时间序列不一致问题。主要问题包括：

- `LapTime_Delta` 与前后圈速差不一致
- `Position_Change` 与前后名次变化不一致
- `PitNextLap` 与下一圈的 `PitStop` 不一致
- 部分数据甚至显示车辆几乎每圈都在 pit，明显不合理

建议不要完全信任 dataset 中已有的 derived features，尤其是 `PitStop` 和 `PitNextLap`。更稳妥的做法是从 FastF1 raw data 重新计算关键特征，并加入 consistency check：

- `LapTime_Delta = current LapTime - previous LapTime`
- `Position_Change = previous Position - current Position`
- `PitNextLap = next lap PitStop`

这样可以避免模型学习到错误的 pit stop pattern。

Reference: https://www.kaggle.com/competitions/playground-series-s6e5/discussion/696380


## TE 的分类

| TE 类型 | 组合方式 | 代表特征 | 特点 | 当前判断 |
|---|---|---|---|---|
| Baseline TE | 原 baseline 已包含的中粒度 TE | `Race × Compound`, `Race × Year` | 有明确业务含义，且已在 baseline 中验证有效 | 保留 |
| Coarse Stable TE | 单字段 TE，且 train/test 覆盖稳定 | `Compound`, `Race`, `Year`, `Stint`, `Position`, `PitStop` | 样本数充足，泄漏风险较低，但可能与原始特征重复 | 可单独测试 |
| Coarse High-card TE | 单字段 TE，但类别数量较多 | `Driver` | 虽是单字段，但 group 较稀疏，容易引入噪声 | 谨慎使用 |
| Mid Stable TE | 双字段 TE，不含 Driver，覆盖较稳定 | `Stint × Compound`, `Race × Stint`, `Year × Stint`, `Year × Compound` | 比 coarse 更有策略含义，且稀疏风险可控 | 下一阶段优先测试 |
| Fine Stable TE | 三字段 TE，但统计上仍较稳定 | `Race × Year × Compound` | 粒度更细，但覆盖率仍较好 | 后续单独 ablation |
| High-risk TE | Driver 相关或过细组合 | `Driver × Race`, `Driver × Compound`, `Driver × Race × Compound` | 组合稀疏，容易过拟合 | 暂不优先 |

## Version 管理

| Version | 修改内容 | OOF AUC | Public Score |
|---|---|---:|---:|
| baseline | 原始 baseline。包含原始特征、基础 categorical/count/bin 特征，以及 baseline 已验证有效的中粒度 TE：`Race × Compound` 和 `Race × Year`。 | 0.95368 | 0.95355 |
| ver6 | `orig` 真实数据历史统计 TE / 细粒度 target-rate 扩展。新增 `hist_pit_next_rate`、`driver_race_year_pit_next_rate`、`compound_pit_next_rate` 等历史进站率聚合特征，并使用层级 fallback 处理 NaN。该版本属于 external/orig-based historical TE，粒度偏细，结果 OOF 和 Public 均明显下降。 | 0.95254 | 0.95199 |
| ver12 | baseline TE + coarse TE 扩展。在 baseline 的 `Race × Compound`、`Race × Year` 基础上，新增粗粒度 TE：`Compound`、`Race`、`Year`、`Driver`、`Stint`。这些组合只作为 TargetEncoder 的类别输入列，实际编码在 K-Fold 训练过程中完成。结果低于 baseline，说明这组 coarse TE 整体没有带来额外收益，其中 `Driver` 可能引入 high-card 噪声。 | 0.95352 | 0.95343 |
