# 2022-B 1.3 无里程计纯夹角校正实验

本目录收录 2022-B 问题 1.3 的 v2 实验结果，包括优化脚本、论文图、代表帧、慢速动画、图表数据和结果报告。

## 结果摘要

- 阶段 I 在第 4 个交替周期完成，全程包含 32 次可逆试探和 4 次正式接受脉冲。
- 阶段 II 在第 19 个同步批次后通过四锚局部角签名判据。
- 最终严格角损失 `L_ang=9.498866024e-4 rad`，低于阈值 `1e-3 rad`。
- 离线评价的位置损失 `L_pos=1.063429192e-6`；该值未参与控制或停止判定。
- 全程共 392 次候选试探，正式接受 66 个脉冲，0 次撤回失败。
- 同参数重复运行后核心结果逐字节一致。

## 目录结构

```text
scripts/
  uav_no_odometry_optimizer.py     两阶段仿真、试探、控制和结果输出
  plots_v2.py                      由 outputs/ 数据生成图、代表帧和 GIF
  test_information_isolation_v2.py 信息隔离与几何退化测试
figures/
  fig_*.png|.svg|.pdf              四张论文主图
  representative_frames/           五张关键状态代表帧
  figure_data/                     图表来源数据
  anim_first_round_slow.gif        第一阶段慢速动画
RESULTS_REPORT.md                  完整运行结果和校验记录
```

## 运行方式

实验依赖 Python、NumPy、Pandas、Matplotlib 和 Pillow，不需要 SciPy 或 FFmpeg。先运行优化器生成 `outputs/` 数据，再运行绘图脚本：

```bash
python scripts/uav_no_odometry_optimizer.py --output-dir outputs --stage-i-max-cycles 500 --stage-ii-max-batches 500
python scripts/plots_v2.py
python scripts/test_information_isolation_v2.py
```

注意：仓库中当前只提交了实验结果和图表数据，未提交优化器重新生成的完整 `outputs/` 目录。若要重新生成论文图，需要先运行优化器。
