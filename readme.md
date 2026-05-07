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


## version管理
baseline
ver6:基于 orig 真实数据的历史进站率聚合特征。理由是visual结果里 hist_pit_next_rate、driver_race_year_pit_next_rate、compound_pit_next_rate这一类最强；guide里也把hist_pit_next_rate排在最强预测器第一位，但提醒会有 NaN，所以这里做了层级 fallback 填充。
