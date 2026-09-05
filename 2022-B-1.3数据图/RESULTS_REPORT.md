# v2 无里程计纯夹角校正结果报告

本报告的数值均重新核对自 `outputs/` 与 `figures/figure_data/`。控制器只使用带标签无符号夹角、目标角签名和机体系脉冲 token；`L_pos`、真实坐标、极坐标与总飞行路程只用于离线记录和评价，不参与动作选择或停止判断。

## 1. 运行环境与复现

运行使用已有虚拟环境 `/Users/shenyuyang/Documents/Codex/2026-09-04/x-2/.venv/bin/python`。实测核心版本为：

| 组件 | 版本 |
| --- | ---: |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| Matplotlib | 3.11.1 |
| Pillow | 12.3.0 |

主运行命令为：

```bash
/Users/shenyuyang/Documents/Codex/2026-09-04/x-2/.venv/bin/python \
  code/uav_no_odometry_optimizer.py \
  --output-dir outputs \
  --stage-i-max-cycles 500 \
  --stage-ii-max-batches 500
```

主要运行参数如下，完整记录见 `outputs/run_summary.json`：

| 参数 | 值 |
| --- | ---: |
| `tau_rad` | 0.001 rad |
| `stage_i_level0` | 3 |
| `stage_ii_level0` | 0 |
| `level_min` | 24 |
| `base_gain_m` | 5.0 m |
| `stage_i_max_cycles` | 500 |
| `stage_ii_max_batches` | 500 |
| `actuator_seed` | 20260905 |

隐藏执行器使用确定性机体轴映射 `orientation=0.73+0.41*k` rad（`k` 为无人机编号），实际位移长度为 `5.0/2^level` m。该映射和米制长度只属于隐藏 `ActuatorModel`，控制器只看到 `F/B/L/R` 与整数 `level`。运行未使用 SciPy 或 FFmpeg；GIF 由 Pillow 生成。

## 2. 控制方法

两阶段流程如下：

1. **阶段 I**：固定 FY00 和 FY02，交替调整 FY05、FY08。每个单机目标分别为 FY05 对 `{FY00,FY02,FY08}` 的三维角签名和 FY08 对 `{FY00,FY02,FY05}` 的三维角签名。目标覆盖四个 30° 角和两个 60° 角。
2. **阶段 II**：冻结 `{FY00,FY02,FY05,FY08}`，对 FY01/FY03/FY04/FY06/FY07/FY09 分别匹配由四锚生成的六维角签名。接收机之间没有目标耦合，因此按同步批次试探和提交。

每一步候选动作由 `F/B/L/R` 的固定顺序产生。试探流程为 `pulse -> measure -> inverse -> rollback check`；只有局部角 RMS 严格改善的候选会正式提交。一轮无改善时命令级别加深一级。`Controller` 没有 World、Sensor、Actuator、Evaluator、回调或坐标数组接口。

## 3. 收敛结果

### 3.1 阶段 I

阶段 I 在第 4 个交替周期通过六个关键角逐项判据，随后冻结 FY00/FY02/FY05/FY08。阶段 I 的全局 `L_ang` 略有上升，符合建模报告中“局部锚框构造不保证全局损失单调”的判断。

| 阶段 I 指标 | 值 |
| --- | ---: |
| 完成周期 | 4 |
| 正式接受脉冲 | 4 |
| 试探候选 | 32 |
| 试探反向脉冲 | 32 |
| 撤回失败 | 0 |
| 最终六角最大绝对残差 | 6.4528867516e-4 rad |

六个关键角残差来自 `state_history.csv` 的 `state_step=5, event=stage_i_complete` 行：

| 角 | 残差 / rad |
| --- | ---: |
| `key_250_rad` | -3.1414344091e-4 |
| `key_850_rad` | -3.3114523424e-4 |
| `key_258_rad` | -6.4528867516e-4 |
| `key_280_rad` | 6.1086523820e-4 |
| `key_580_rad` | -2.9672247335e-4 |
| `key_285_rad` | 3.1414276485e-4 |

四个接受的脉冲为：FY05 `F@3`、FY08 `F@5`、FY05 `B@5`、FY08 `B@5`。逐项达标表明没有进入四个 30° 的共线伪解支。

### 3.2 阶段 II

阶段 II 在第 19 个同步批次后通过六机四锚局部角签名判据，最终严格冻结态全局角损失也达到停止条件。

| 阶段 II 指标 | 值 |
| --- | ---: |
| 完成同步批次 | 19 |
| 正式接受脉冲 | 62 |
| 试探候选 | 360 |
| 试探反向脉冲 | 360 |
| 撤回失败 | 0 |

各接收机正式接受数为：

| 接收机 | 接受脉冲 |
| ---: | ---: |
| FY01 | 9 |
| FY03 | 13 |
| FY04 | 17 |
| FY06 | 18 |
| FY07 | 14 |
| FY09 | 19 |

最终四锚局部 RMS 如下：

| 接收机 | 局部 RMS / rad |
| ---: | ---: |
| FY01 | 1.6356284190e-4 |
| FY03 | 6.9744305309e-4 |
| FY04 | 7.6317393308e-4 |
| FY06 | 7.5939174972e-4 |
| FY07 | 7.9069297341e-4 |
| FY09 | 2.6092145975e-4 |

最大值出现在 FY07，为 `7.9069e-4 rad < 1e-3 rad`。

## 4. 最终指标

最终冻结态指标来自 `run_summary.json` 与 `state_history.csv` 的 `event=final_success` 行。

### 4.1 角度与位置评价

| 指标 | 初态 | 最终态 |
| --- | ---: | ---: |
| 严格 `L_ang` / rad | 1.1245814644e-1 | 9.4988660239e-4 |
| 离线 `L_pos` | 2.887790951e-3 | 1.0634291916e-6 |
| 相似拟合尺度 `rho_fit` / m | 104.4440276760 | 98.0096229844 |
| FY00 相对拟合圆心偏移 / m | 0.4460362713 | 0.0449940400 |
| 最大位置误差 / m | 7.9847183859 | 0.1703540216 |
| 平均位置误差 / m | 4.8074970065 | 0.0941914601 |
| 最大相邻圆心角间隔误差 / ° | 0.4407590328 | 0.1734361082 |
| 外围半径均值 / m | 104.4444444444 | 98.0096892084 |
| 外围半径 RMS / m | 104.6125125297 | 98.0096923633 |

最终 `L_ang = 9.498866024e-4 rad < 1e-3 rad`。`L_pos` 及上表中的位置指标只作为离线诊断，没有作为停止条件。

### 4.2 固定点与标签顺序

FY00 全程位移为 `0 m`。阶段 I 完成后，FY02/FY05/FY08 在阶段 II 中均未再移动。阶段 I 完成态与最终态的四点坐标逐分量相同：

| 固定点 | 最终坐标 / m |
| --- | ---: |
| FY00 | `(0, 0)` |
| FY02 | `(74.9622972900, 63.1241157166)` |
| FY05 | `(-92.1735291833, 33.3044536935)` |
| FY08 | `(17.3038004604, -96.4602430519)` |

最终 `label_order_ok=true`，相似变换旋转矩阵行列式为 `1`，固定标签的逆时针顺序保持，未使用标签交换或镜像分支。

## 5. 控制预算与试探校验

总预算统计来自 `moves.csv`、`adjustments.csv` 和 `run_summary.json`：

| 项目 | 值 |
| --- | ---: |
| 候选试探次数 | 392 |
| 试探反向动作次数 | 392 |
| 正式提交脉冲 | 66 |
| 撤回失败 | 0 |
| `moves.csv` 动作记录总数 | 850 |

阶段划分如下：

| 阶段 | 候选试探 | 反向试探 | 正式提交 |
| --- | ---: | ---: | ---: |
| 阶段 I | 32 | 32 | 4 |
| 阶段 II | 360 | 360 | 62 |

接受脉冲的机体系方向分布为：

| 阶段 | F | B | L | R |
| --- | ---: | ---: | ---: | ---: |
| 阶段 I | 2 | 2 | 0 | 0 |
| 阶段 II | 17 | 15 | 22 | 8 |

全程隐藏实际路程为 `1465.9375 m`，包含试探、反向撤回和正式提交，仅作离线统计。

## 6. 全局扫描与成功判据

最终全局诊断按建模报告的固定扫描进行：三组外围发射机 `{2,5,8}`、`{3,6,9}`、`{1,4,7}` 各与 FY00 联合，三个接收组共产生 `3*6*6=108` 个无符号夹角残差。最终 108 项残差的 RMS 复核值为 `9.4988660239e-4 rad`，与 `run_summary.json` 一致。

成功状态不是由 `L_pos` 判定，而是同时满足：

1. FY00 净位移为 0；
2. 阶段 I 六个关键角逐项达到 `1e-3 rad`；
3. 阶段 II 六机四锚局部 RMS 均达到 `1e-3 rad`；
4. 冻结态 108 项 `L_ang < 1e-3 rad`；
5. 全部试探可撤回且无撤回失败。

`run_summary.json` 的 `success=true` 与上述五项判据一致。

## 7. 信息隔离与回归测试

完整测试使用：

```bash
/Users/shenyuyang/Documents/Codex/2026-09-04/x-2/.venv/bin/python \
  code/test_information_isolation_v2.py
```

本次复验结果：**15 tests OK**。覆盖项包括：

- `AngleBatch`、`TargetSignature`、`Pulse` 深度不可变；
- Controller 源码、对象图和预录角度 tape 均不暴露真值或坐标；
- 反向试探无累计漂移、更深级别实际位移更小、固定编号不动；
- 四个 30° 共线反例被拒绝；
- 全局扫描只读且产生 108 项残差；
- `L_pos` 满足相似不变性，端到端相似等变下控制选择一致。

## 8. 确定性复跑

为避免只比较同一路径的缓存文件，本次在 `/private/tmp/x-2-v2-determinism-check` 独立重跑一次主程序。下列文件与正式 `outputs/` 对应文件逐字节一致：

1. `state_history.csv`
2. `adjustments.csv`
3. `moves.csv`
4. `phase_summary.csv`
5. `run_summary.json`

因此核心轨迹、预算与最终指标在同参数运行中可复现。

## 9. 图表与动画

论文图由 `code/plots_v2.py` 从 CSV/JSON 生成，同时输出 PNG、SVG、PDF；原始图表数据保存在 `figures/figure_data/`。

| 文件 | 内容与数据来源 |
| --- | --- |
| `figures/fig_initial_final.*` | 初态与最终成功态位置图；数据来自 `state_history.csv` |
| `figures/fig_loss_curves.*` | `L_ang` 与离线 `L_pos` 曲线；数据来自 `state_history.csv` |
| `figures/fig_angle_diagnostics.*` | 六个关键角残差与六机四锚 RMS；数据来自 `state_history.csv` |
| `figures/fig_control_budget.*` | 脉冲级别、接受/拒绝和隐藏位移；数据来自 `moves.csv` |
| `figures/figure_data/*.csv` | 上述图的原始绘图数据 |
| `figures/anim_first_round_slow.gif` | 第一阶段 36 帧慢速动画，尺寸 `740x760 px`，逐帧间隔 `1800 ms` |
| `figures/representative_frames/*.png` | 初态、FY05 首次接受、FY08 首次接受、一次联合交替后、阶段 I 完成 |

动画帧索引和帧级指标保存在 `figures/figure_data/anim_first_round_frames.csv`，共 36 帧。它由 `moves.csv` 中阶段 I 的真实试探/提交状态重建，不使用插值伪造状态。

## 10. 视觉检查

对五张代表帧 `01_initial.png`、`02_fy05_first_accepted.png`、`03_fy08_first_accepted.png`、`04_after_joint_alternation.png`、`05_stage_i_complete.png` 做了人工检查。五张图中的无人机编号、状态标题和指标框均可读，没有标签相互遮挡；底部图例和图题贴近页边但未遮挡图形元素。

GIF 的首、中、末状态与阶段 I 事件对应，阶段标题、极坐标和关键角信息可读。

## 11. 结论与边界

在无噪声基础仿真中，纯夹角控制成功完成两阶段编队校正。阶段 I 排除共线伪解并冻结正确锚框，阶段 II 只用四锚角签名校正其余六机；最终冻结态严格角 RMS 为 `9.4989e-4 rad`，满足设定阈值。控制过程未使用全局航向、实际位移、真实坐标或 `L_pos`。

本结果依赖已冻结的基础仿真假设：无角测噪声、可重复机体系动作、可反向撤回试探、固定标签顺序作为连续分支。未做噪声鲁棒性实验，不做 v1/v2 对比图。由于最终 `state_history.csv` 的 `final_success` 行不再保留阶段 I 六角残差，最终六角值应从 `state_step=5, event=stage_i_complete` 行读取，而不是从最终行读取。
