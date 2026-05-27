# 多参考帧共视定位与 Local BA

本文档说明 `covis_absolute_pose` 管道中**多参考帧**策略：如何从若干已知位姿的共视邻帧恢复目标帧绝对位姿，以及各步骤与公开实现的对应关系。

相关设计笔记见仓库根目录 [LocalBA.md](../../LocalBA.md)、[Multicamera.md](../../Multicamera.md)。

---

## 1. 问题设定

- **输入**：目标帧 `T`、若干共视邻帧 `{R_i}`（车体平面位姿 + 相机外参已知，带噪）。
- **输出**：目标帧相机/车体在世界系下的位姿。
- **约束**：地面机器人场景下，车体主运动为平面 `(x, y, θ)`；相机相对车体有小高度与姿态安装误差，可在 BA 中用小范围 `dz / roll / pitch` 吸收。

整体流程：

```
共视邻帧检索 → 图像对匹配 + 极线 RANSAC → 选参考对 → 建 track → PnP 初值 → Local BA → 评估
```

入口脚本：`scripts/eval/covis_absolute_pose.py`  
核心几何：`lib/geometry/local_ba_planar.py`

---

## 2. 多参考帧策略概览

### 2.1 参考对选择

在共视邻帧中枚举 `(C1, C2)`，按**基线长度**与**三对匹配质量**（C1–C2、C1–T、C2–T 极线 inlier 数）打分，选最优一对作为主参考。

- 基线过小（`min_baseline_m`）→ 三角化退化，跳过（LocalBA 情况 A）。
- 实现：`select_ref_pair` / `_sorted_ref_pairs` in `covis_absolute_pose.py`。

### 2.2 三视图轨迹 `build_triview_tracks`（主路径）

对应 **triview_c23** 模式：

1. **C1–C2 匹配** → DLT 两视图三角化 → 深度过滤（`depth_min_m` ~ `depth_max_m`）。
2. **C1–T 桥接**：三角化点在 C1 上的像素，在 C1–T 匹配中最近邻关联（阈值 `associate_px`）。
3. **C2–T 桥接**：同理在 C2–T 匹配中关联。
4. **交叉验证**：C1 桥接与 C2 桥接给出的 target 像素须一致（`c23_agree_px`），通过则取平均作为 target 观测。

漏斗统计（写入 `track_stats`，HTML 报告可见）：

| 字段 | 含义 |
|------|------|
| `n_match_c12/c13/c23` | 极线过滤后三对匹配数 |
| `n_triangulated` | C1–C2 三角化点数 |
| `n_depth_ok` | 深度合法点数 |
| `n_c13_assoc` / `n_c23_assoc` | 桥接关联成功数 |
| `mean_c13_assoc_dist_px` 等 | **实际**最近邻像素偏差均值（非阈值） |
| `n_c23_agree` | 双桥接一致通过数 |
| `mean_c23_agree_dist_px` | 双桥接 target 像素差均值 |
| `n_tracks` | 最终轨迹数 |

### 2.3 Fallback：`fallback_c1_bridge`

当三视图轨迹不足（`< min_pnp_points`）时，仅用 **C1 桥接** target，不做 C2–T 一致性校验。统计字段同上（无 C23 相关项）。

### 2.4 追加参考帧 `augment_tracks_with_extra_refs`

当 `multiview=true` 且 `max_ref_cameras > 2` 时：

1. 按 C1–T 匹配 inlier 数排序，选取额外共视帧 `R_extra`。
2. 对每条已有 track：C1 像素 → C1–R_extra 匹配 → R_extra 上期望像素 → R_extra–T 匹配最近邻（`associate_px`）。
3. 将 R_extra 观测并入 track，不重新跨视图全局匹配。

统计：`n_extra_obs_added`、`n_tracks_augmented`、`mean_associate_dist_px`。

### 2.5 多视图三角化精炼 `refine_tracks_multiview`

对每条 track **已有**各相机观测，用 **多视图 SVD 三角化**（`triangulate_multiview_svd`）重算 `X_world`，再按重投影门控剔除坏点。不新增跨视图匹配。

对应 Multicamera.md §1「多视图三角化 + 分布式 PnP」中的三角化步骤。

### 2.6 PnP 初值 + Local BA

1. **PnP**：`solve_pnp_from_tracks`（RANSAC + EPnP）。
2. **Local BA**：`local_ba_planar_robust` — Huber 重投影 + 参考帧先验 + 可选外点迭代剔除。

**优化自由度（方案 B，默认开启）**：

| 相机 | 变量 | 边界 / 先验 |
|------|------|-------------|
| 参考帧 | `x, y, θ, dz, roll, pitch` | xy/yaw 强先验 + 小边界；dz/roll/pitch ±小范围 |
| 目标帧 | 同上 6 DOF | 以 PnP 初值为锚；dz/roll/pitch 小范围优化 |

关闭目标 6DOF：`--no-ba-target-extra-dofs`。  
关闭参考 6DOF：`--no-ba-ref-extra-dofs`。

---

## 3. 匹配与极线缓存

- 匹配器：`silk`（mutual NN）或 `lightglue`。
- 极线：`filter_matches_epipolar`（Essential + RANSAC，见 `scripts/eval/epipolar_safe.py`）。
- 序列级缓存：`outputs/.../match_pairs/covis_edges.npz`（全序列边一次落盘，避免重复 LightGlue）。

报告中的匹配列格式：`raw→epi N→M→used K`（极线前 → 极线 inlier → 实际使用数）。

---

## 4. 失败模式速查

| 现象 | 可能原因 |
|------|----------|
| `triangulation_or_tracks_failed` | 参考对基线小、匹配少、或 C23 交叉验证过严 |
| `n_c13_assoc ≈ 0` | C1–T 匹配与 C1–C2 三角化点无法关联（`associate_px` 过小或匹配质量差） |
| `n_c23_agree` 低 | C1/C2 桥接 target 不一致（位姿噪声或误匹配） |
| BA 重投影 `—` | 该相机无 track 观测（非 bug，HTML 已格式化为空） |

---

## 5. 命令行与配置

```bash
python scripts/eval/covis_absolute_pose.py \
  --seq-id SEQ \
  --data-dir /path/to/org/SEQ \
  --camera-json .../fishEyeIn.conf \
  --extrinsic-json .../fishEyeWholeEx.conf \
  --out-dir ./outputs/covis_absolute_pose/SEQ \
  --matcher lightglue \
  --max-ref-cameras 4 \
  --associate-px 3.0 \
  --c23-agree-px 4.0
```

Hydra 模板：`etc/mode/run-covis-absolute-pose.yaml`

---

## 6. 公开参考仓库

下列开源项目与本管道各阶段有对应关系，便于对照实现与调参。

### 6.1 特征匹配与极线几何

| 项目 | 链接 | 与本管道关系 |
|------|------|--------------|
| **SiLK** | https://github.com/facebookresearch/silk | 本仓库特征提取与 mutual-NN 匹配基线 |
| **LightGlue** | https://github.com/cvg/LightGlue | 可选学习型匹配器（默认 disk/superpoint 头） |
| **SuperGlue** | https://github.com/magicleap/SuperGluePretrainedNetwork | LightGlue 的前代；极线过滤思路相同 |
| **OpenCV calib3d** | https://github.com/opencv/opencv/tree/4.x/modules/calib3d | `triangulatePoints`、`findEssentialMat`、`recoverPose` |

### 6.2 多视图三角化与 SfM / BA

| 项目 | 链接 | 与本管道关系 |
|------|------|--------------|
| **COLMAP** | https://github.com/colmap/colmap | 工业标准 SfM：两视图/多视图三角化、BA、鲁棒核 |
| **pycolmap** | https://github.com/colmap/pycolmap | COLMAP Python 绑定；hloc 三角化/定位后端 |
| **hloc** | https://github.com/cvg/Hierarchical-Localization | 检索 + 匹配 + COLMAP 三角化 + PnP 定位全流程 |
| **OpenGV** | https://github.com/laurentkneip/opengv | 相对/绝对位姿、RANSAC 几何库 |

hloc 中共视聚类与多参考帧 PnP 见  
[`hloc/localize_sfm.py`](https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/localize_sfm.py)（`do_covisibility_clustering`、`pose_from_cluster`）。

### 6.3 图优化与 SLAM 式 Local BA

| 项目 | 链接 | 与本管道关系 |
|------|------|--------------|
| **g2o** | https://github.com/RainerKuemmerle/g2o | 位姿图 / BA 通用优化框架 |
| **GTSAM** | https://github.com/borglab/gtsam | 因子图 SLAM；边缘化与鲁棒核 |
| **Ceres Solver** | https://github.com/ceres-solver/ceres-solver | 非线性最小二乘（本管道用 scipy `least_squares` 同类思路） |
| **ORB-SLAM3** | https://github.com/UZ-SLAMLab/ORB_SLAM3 | 多关键帧局部 BA 的工程参考 |
| **Basalt** | https://gitlab.com/VladyslavUsenko/basalt | 视觉惯性 BA 与三角化实现 |

### 6.4 共视与数据集工具

| 项目 | 链接 | 与本管道关系 |
|------|------|--------------|
| **LaMAR benchmark** | https://github.com/microsoft/lamar-benchmark | 深度/位姿驱动的视觉重叠估计（[`overlap.py`](https://github.com/microsoft/lamar-benchmark/blob/main/scantools/proc/overlap.py)） |
| **MegaDepth / ScanNet 评测** | 见 hloc pipelines | 共视对 + 匹配 + 位姿 AUC 评测范式 |

### 6.5 相关论文（便于追溯算法来源）

- **Hierarchical Localization**：Sarlin et al., CVPR 2019 — 检索 + 局部匹配 + SfM 定位框架（hloc）。
- **LightGlue**：Lindenberger et al., ICCV 2023 — 快速特征匹配。
- **SiLK**：Gleize et al., ICCV 2023 — 子像素级特征。

---

## 7. 代码索引

| 模块 | 路径 |
|------|------|
| 主评估脚本 | `scripts/eval/covis_absolute_pose.py` |
| HTML 报告 | `scripts/eval/covis_absolute_pose_viz.py` |
| Local BA / track 构建 | `lib/geometry/local_ba_planar.py` |
| 极线安全封装 | `scripts/eval/epipolar_safe.py` |
| 序列匹配缓存 | `scripts/eval/sequence_pair_cache.py` |
| 批量脚本 | `batch_covis_absolute_pose.sh` |

---

## 8. 与 COLMAP / hloc 的差异（简要）

| 维度 | 本管道 | COLMAP / hloc |
|------|--------|---------------|
| 位姿参数化 | 平面车体 + 小 6DOF 扰动 | 完整 SE(3) 相机位姿 |
| 地图 | 每帧临时 track，不持久化全局地图 | 全局 SfM 模型 + 3D 点云 |
| 参考帧 | 共视邻帧 + 参考对枚举 + 可选追加 | 检索 / 共视聚类 + 多 DB 图像 |
| 优化器 | scipy TRF + Huber | Ceres / pycolmap BA |

本方案面向**已有里程计位姿的序列内重定位**，而非从零建图。
