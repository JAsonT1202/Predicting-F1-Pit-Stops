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


## Version 管理

| Version | 修改内容 | 分数 |
|---|---|---:|
| baseline | 原始 baseline | 0.95355 |
| ver6 | 基于 `orig` 真实数据新增历史进站率聚合特征。理由是 visual 结果中 `hist_pit_next_rate`、`driver_race_year_pit_next_rate`、`compound_pit_next_rate` 等特征较强；guide 中也将 `hist_pit_next_rate` 作为最强预测器之一。但由于该类特征存在 NaN，因此加入了层级 fallback 填充。 | 0.95199 |
| ver7 | 在 baseline 基础上新增低泄漏风险的当前状态交互特征，不使用 `PitNextLap` 统计类特征，主要基于轮胎状态、比赛进度、衰减速度、位置变化和 Compound 交互构造特征。 | 0.95339 |
| ver8 | 在 baseline 基础上扩展类别交互编码组合，新增 `Driver × Compound` 和 `Driver × Race`。不新增 row-wise 数值特征，主要沿用 baseline 中的 categorical interaction + TargetEncoder 思路，验证驾驶员与轮胎、驾驶员与赛道组合是否能捕捉更稳定的策略差异。 | 0.95182|
| ver9 | 在 baseline 基础上新增少量核心数值变换特征，包括 `LapTime_Delta_abs`、`TyreLife_sq`、`Stint_x_TyreLife`、`PitStop_x_TyreLife`、`PitStop_x_Stint` 和简单的 `Compound_simple_enc`。由于 ver6/ver8 中历史统计和类别交互均明显变差，本版本不再扩展 Driver/Race 历史类特征，而是聚焦于讨论区和 visual 分析中更稳定的核心信号：Stint、TyreLife、LapTime_Delta、PitStop 和 Compound。 | 待提交 |
