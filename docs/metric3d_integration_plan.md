# Metric3D 接入 ORB-SLAM3 以辅助视觉惯性初始化：设计方案

> 阅读说明：本文公式均使用普通文本公式块，避免依赖 Markdown/LaTeX 数学渲染扩展。

## 0. 目的、范围与结论

本文讨论在当前工程中接入 **Metric3D/Metric3Dv2** 产生的单目米制深度图，以改善 `IMU_MONOCULAR` 的尺度初始化。目标是缩短获得**可信米制尺度**所需的运动时间，或提高初始尺度的稳定性；目标不是用网络深度取代 IMU。

必须区分两件事：

1. 单目视觉地图的 Sim(3) 尺度 `s`；
2. 惯性状态中的重力方向、每帧速度、陀螺仪 bias `b_g`、加速度计 bias `b_a`。

Metric3D 只对第 1 项提供观测。即使网络深度完全正确，静止、匀速或缺乏旋转激励时，`g`、`b_g`、`b_a` 和速度仍不能仅靠深度图可靠恢复。因此既不能跳过惯性初始化，也不应删除 `VigInit()` 的 IMU 激励检查。

推荐先实施**方案 A：尺度软先验**。它不改变视觉地图的建图语义，允许 IMU 和视觉惯性优化拒绝有偏的网络深度。方案 B（伪 RGB-D）可以让视觉地图更早进入米制，但会把预测误差直接写入 MapPoint，风险显著更高，适合作为后续实验分支而非默认路径。

本文只描述当前仓库可核实的接口和建议新增的接口；所有新增类、成员和函数都会标明为“新增”。

---

## 1. 当前代码基础

### 1.1 原生惯性初始化中的尺度变量

原生路径为 `LocalMapping::InitializeIMU()`（`src/LocalMapping.cc`）。它调用：

```cpp
Optimizer::InertialOptimization(
    map, mRwg, mScale, mbg, mba,
    mbMonocular, infoInertial,
    false, false, priorG, priorA);
```

`Optimizer::InertialOptimization()`（`src/Optimizer.cc`）创建了 `VertexScale`。单目中该顶点可优化；双目/RGB-D 中固定：

```cpp
VertexScale* VS = new VertexScale(scale);
VS->setFixed(!bMono);
```

`VertexScale` 的增量并非加法，而是：

```cpp
setEstimate(estimate() * exp(*update_));
```

即优化增量天然位于 ξ = log(s) 空间。这一点使 log-scale 先验可以直接使用常量雅可比。

每个相邻关键帧对使用 `EdgeInertialGS`。其速度、位置残差的核心形式是：

```text
r_v = R_bw,i * [s * (v_j - v_i) - g * dt] - Delta_v,ij(b_g, b_a)
```

```text
r_p = R_bw,i * [s * (p_j - p_i - v_i * dt) - 0.5 * g * dt^2]
      - Delta_p,ij(b_g, b_a)
```

实现见 `src/G2oTypes.cc::EdgeInertialGS::computeError()`。原生 IMU 边以预积分协方差的逆作为信息矩阵，因此不是等权约束。

### 1.2 当前 RGB-D 深度接口

`Frame` 已有用于立体/RGB-D 的字段：

```cpp
std::vector<float> mvDepth;
std::vector<float> mvuRight;
```

`Frame::ComputeStereoFromRGBD()` 会在每个 ORB keypoint 处读取深度并填充这两个数组。`Frame::UnprojectStereo()` 将 `mvDepth[i]` 解释为 pinhole 相机坐标的 **Z-depth**：

```text
x = (u - c_x) * z / f_x;  y = (v - c_y) * z / f_y;  z = mvDepth[i]
```

因此方案 B 输入给 `mvDepth` 的必须是同一相机模型、同一图像坐标系下的 Z-depth，不能不加确认地将“相机中心欧氏距离”当作 Z-depth。

### 1.3 Metric3D 输出的必要校正

Metric3D 官方推理流程先在 canonical camera space 中预测深度，再依据真实相机内参做反变换。官方示例给出：

```text
D_metric = D_canonical * f_x / 1000
```

其中 `1000` 是示例中 canonical focal length；实际必须跟随所使用模型/config 的 canonical focal length，不能硬编码。参考：<https://github.com/YvanYin/Metric3D/blob/main/hubconf.py>。

同时应记录并验证：输入图像 resize/crop/pad 的映射、预测图到原图的反采样、使用的是原图内参还是变换后的内参。若图像先去畸变，Metric3D 也应看到同一张去畸变图，或必须把深度图严格重投影回原图坐标。

---

## 2. 共同的数据准备与质量控制

两种方案均应先实现一个独立的 Metric3D 推理与缓存层；不要在 Tracking 主线程中同步执行 Python/PyTorch 推理。

### 2.1 建议新增的数据对象

以下是建议接口，不是现有代码：

```cpp
struct MetricDepthResult {
    cv::Mat depth_z_m;       // CV_32FC1, pinhole Z-depth, metre
    cv::Mat confidence;      // CV_32FC1, optional
    cv::Mat valid_mask;      // CV_8UC1
    Eigen::Vector4f K_used;  // fx, fy, cx, cy in depth_z_m coordinates
    uint64_t frame_id;
};
```

推理任务应在 KeyFrame 创建后由独立 worker 排队处理，缓存结果按 `frame_id` 或 KeyFrame id 绑定。Local Mapping 读取“已完成的不可变结果”；结果未完成时跳过该帧，而不是等待。

### 2.2 坐标与深度定义验证

对某个 ORB 特征点，不能默认 `mvKeys[i].pt` 就可直接索引预测深度图。应显式维护从特征图像坐标到深度图坐标的映射：

```text
[u_D; v_D; 1]
= H_D<-I
[u_I; v_I; 1]
```

对于仅有 resize + pad 的情况，`H` 是缩放加平移；若存在畸变/去畸变，则应使用相机模型投影/反投影，而非假定单一 3x3 单应。

必须完成以下离线检查后再接入初始化：

1. 在已知深度数据上验证输出单位为米；
2. 用已知平面或标尺检查深度是不是 Z-depth；
3. 检查 resize/crop 后采样像素的重投影误差；
4. 分场景统计尺度比、深度 RMSE 和失效区域；
5. 检查相机焦距与训练域差异带来的尺度偏差。

### 2.3 通用有效像素筛选

建议仅接受同时满足以下条件的像素/观测：

- 深度有限且落于经验证的工作范围 `[d_min, d_max]`；
- confidence（若模型输出）高于阈值；
- 距离图像边缘足够远；
- 不在语义掩码的天空、动态物体、反光/透明区域内（若系统具备此类掩码）；
- 对应 MapPoint 位于相机前方；
- 跨帧尺度比没有被鲁棒统计判为离群值。

阈值不应写死在算法推导中；应配置化，并以目标传感器数据集的验证统计选取。

---

## 3. 方案 A：Metric3D 作为单目尺度软先验（推荐）

### 3.1 思路

保持现有单目 ORB-SLAM3 的视觉建图和关键帧位姿不变。Metric3D 不直接创建或移动 MapPoint，而是估计当前视觉地图到米制世界的一个全局尺度观测 `s_D`，随后以**软约束**加入原生 `InertialOptimization()`。

优点：

- 网络深度错误不会直接修改地图；
- 原生 IMU 预积分、协方差、bias 先验、Full Inertial BA 均保留；
- 先验可随置信度自动变弱或被关闭；
- 不需要把 `IMU_MONOCULAR` 改为 `IMU_RGBD`。

### 3.1.1 相比原生单目初始化，究竟增强在哪里

原生单目 VIO 中，视觉重建只确定相似变换意义下的轨迹；地图位置、速度都带有未知全局尺度。`InitializeIMU()` 必须从 IMU 预积分和视觉运动的一致性中同时恢复：

```text
scale + all keyframe velocities + gravity direction + gyro bias + accel bias
```

当平移小或加速度变化弱时，尺度、速度和加速度计 bias 会互相耦合：优化器可以用某种速度变化或 `ba` 变化，部分解释本应由尺度解释的残差。因此尺度相关的 Hessian 信息较弱，初始化会更依赖较长时间窗口和充分运动。

Metric3D 的有效深度观测提供：

```text
D_metric(u, v) approx s * Z_slam(u, v)
```

经跨关键帧鲁棒聚合后，得到的不是“又一张普通深度图”，而是对全局 `log(scale)` 的附加信息。若原生问题中 `xi = log(scale)` 的尺度信息为 `H_xi_xi`，尺度先验的信息为：

```text
H_metric = 1 / sigma_log_scale^2
H_xi_xi_new = H_xi_xi + H_metric
```

在网络尺度近似无偏、`sigma_log_scale` 合理且先验与 IMU 误差不强相关的条件下，尺度的后验方差会减小。若其他待估变量记为 `y`，消去尺度后的边缘信息可写为：

```text
H_y_marginal = H_yy - H_y_xi * inverse(H_xi_xi + H_metric) * H_xi_y
```

这说明先验首先固定尺度，再间接改善与尺度强耦合的速度和 `ba` 的数值条件；它并非直接测得了 bias。

工程上可预期的收益是：

- 初始化尺度初值更接近米制，`ApplyScaledRotation()` 后的尺度跳变通常更小；
- 在平移较小、远景较多或视觉三角化深度弱的窗口中，尺度更不易漂移；
- 原生 g2o 优化对尺度的搜索空间变小，更可能较早进入正确收敛盆地；
- 在尺度已经可靠时，启动动作可从“大平移/大加速度”转为“多方向缓慢旋转 + 少量自然平移”；
- 后续 VIBA 1/2 可从更合理的尺度和速度初值继续校正，而不是承担首次大尺度修正。

这些收益说的是**初始化质量和成功窗口**，不等同于推理后 CPU 时间一定更短。Metric3D 本身有 GPU/CPU 推理延迟；若同步运行，端到端初始化时间反而可能增加。因此必须异步运行，并以“首次可信初始化所需的用户运动量和失败率”而非单个函数耗时评价收益。

### 3.1.2 它不能带来的收益，以及可能变差的地方

Metric3D 没有观测以下量：

- 重力与 `ba` 在固定姿态下的分离；
- gyro bias 对旋转预积分的影响；
- IMU-相机时间偏移与外参错误；
- 静止或噪声量级微小运动时的惯性可观测性。

所以它不能使“完全静止”自动变成可完成的完整 VIO 初始化；相关推导见第 7 节。

更重要的风险是，学习深度误差通常存在空间相关性和系统偏差，而不是独立同分布的高斯噪声：

```text
D_metric = alpha * D_true + beta + scene_dependent_error
```

例如焦距处理错误、室内外域偏移、天空/玻璃/反射、动态物体、极近或极远区域，都可能让 `alpha`、`beta` 或误差模式发生变化。若把大量像素当作独立测量，会错误地把先验权重叠加得过强；若强行固定尺度，优化器可能将网络的尺度偏差错误吸收到速度、`ba` 或重力方向中。

这也是方案 A 采用“每关键帧聚合 -> 跨帧鲁棒聚合 -> 一个带方差的软 scale edge”的原因：它主动避免把稠密网络输出误当作数千个独立的高精度传感器测量。预测深度只能在验证通过时提供帮助；深度不稳定时，正确行为是拒绝先验并回退原生初始化。

### 3.2 从 MapPoint 与深度图估计尺度

设当前单目地图单位下的 MapPoint 为 `X_k`，关键帧 `i` 的视觉位姿为 `T_{cw,i}`。其在相机坐标的深度为：

```text
z^{slam}_{ik} = [T_{cw,i}X_k]_z>0
```

从 Metric3D 深度图在该点投影像素采样得到：

```text
d_{ik}=D_i(pi(T_{cw,i}X_k))
```

视觉地图与物理世界仅差一个尺度时：

```text
d_{ik}=s_Dz^{slam}_{ik}+epsilon_{ik}
```

直接在线性深度域估计会让远点支配代价。采用 log-depth 更适合尺度估计：

```text
y_{ik}=log d_{ik}-log z^{slam}_{ik}=log s_D+eta_{ik}
```

一个鲁棒、实现简单的估计器是加权中位数：

```text
xi_D = weightedMedian(y_ik; w_ik)
```

```text
s_D = exp(xi_D)
```

其中权重可由模型 confidence 与 MapPoint 观测质量构成。为了避免一帧有大量点而支配结果，推荐两级聚合：

1. 每个关键帧先对其有效点求 `xi_i`；
2. 对最近若干关键帧的 `xi_i` 求加权中位数得到 `xi_D`。

使用中位绝对偏差（MAD）估计不确定性：

```text
sigma_MAD = 1.4826 * median_i |xi_i - xi_D|
```

只有在有效关键帧数、有效点数、`sigma_MAD` 都满足要求时才发布尺度先验。若尺度比在帧间不一致，应**不添加先验**，而不是勉强输出平均值。

### 3.3 加入原生尺度顶点的推导

当前 `VertexScale` 用：

```text
s <- s * exp(delta_xi)
```

更新，故其局部变量正是：

```text
xi=log s
```

将网络给出的尺度观测写作 `xi_D = log(s_D)`，定义一维残差：

```text
r_D(xi) = xi - xi_D = log(s) - log(s_D)
```

高斯先验代价为：

```text
E_D=(r_D^2)/(sigma_D^2)
```

由于优化增量就是 `delta_xi`，雅可比是：

```text
dr_D / d(delta_xi) = 1
```

信息矩阵为一维标量：

```text
Omega_D = 1 / sigma_D^2
```

完整目标从原来的：

```text
E_{IMU}+E_{prior}(b_g,b_a)
```

变成：

```text
E_{IMU}+E_{prior}(b_g,b_a)+E_D
```

必要时可对 `r_D` 使用 Huber 核，防止少数仍未筛掉的错误深度图让尺度先验过强。

### 3.4 建议的代码改动

#### 新增 1：尺度观测对象

建议新增一个只表达统计结果的类型：

```cpp
struct MetricScalePrior {
    bool valid = false;
    double log_scale = 0.0;
    double sigma_log_scale = 0.0;
    int num_keyframes = 0;
    int num_observations = 0;
};
```

它应由 Local Mapping 的只读地图状态计算得到，不应由 Tracking 直接修改地图。

#### 新增 2：g2o 一维边

在 `include/G2oTypes.h` / `src/G2oTypes.cc` 中新增一个连接 `VertexScale` 的一维 unary edge，例如 `EdgePriorScaleLog`。概念性实现为：

```cpp
// 新增：接口示意，不是现有代码
class EdgePriorScaleLog : public g2o::BaseUnaryEdge<1, double, VertexScale> {
public:
    void computeError() override {
        const auto* v = static_cast<const VertexScale*>(_vertices[0]);
        _error[0] = std::log(v->estimate()) - _measurement;
    }
    void linearizeOplus() override {
        _jacobianOplusXi[0] = 1.0; // VertexScale 的 oplus 是 log-scale 增量
    }
};
```

边创建时：

```cpp
if (metricPrior.valid) {
    auto* edge = new EdgePriorScaleLog;
    edge->setVertex(0, VS);
    edge->setMeasurement(metricPrior.log_scale);
    edge->setInformation(Eigen::Matrix<double,1,1>::Constant(
        1.0 / (metricPrior.sigma_log_scale * metricPrior.sigma_log_scale)));
    optimizer.addEdge(edge);
}
```

`log(s)` 要求 `s>0`。当前 `VertexScale` 的乘法更新可保持正性；但外部初值 `mScale` 和先验估计仍应在进入图优化前验证为有限正数。

#### 新增 3：扩展优化器参数

推荐为 `InertialOptimization()` 增加可选参数，而不是全局变量：

```cpp
const MetricScalePrior* metricScalePrior = nullptr
```

在创建 `VertexScale` 后、`optimizer.initializeOptimization()` 前，若指针非空且 `valid`，加入上述边。原有调用传 `nullptr`，从而保持回归行为不变。

#### 新增 4：在初始化前估计 prior

在 `LocalMapping::InitializeIMU()` 调用优化器之前，基于最近已完成 Metric3D 的关键帧计算 `MetricScalePrior`。不要在 `Optimizer` 内部运行深度网络，也不要在其中读取/写入缓存。

代码流：

```text
新关键帧创建
  -> 异步 Metric3D 推理并缓存 depth/confidence
  -> Local Mapping 收到后续关键帧
  -> 收集已完成深度的最近 KFs + 可见 MapPoints
  -> 计算 MetricScalePrior（或 invalid）
  -> InitializeIMU(..., metricPrior)
  -> EdgeInertialGS + bias priors + EdgePriorScaleLog
  -> g2o 联合优化
  -> 原有 ApplyScaledRotation / UpdateFrameIMU / FIBA
```

### 3.5 初值与先验强度策略

可以将 `mScale` 初始化为 `s_D`，但仍应保留软边。仅设置初值而没有先验，在 LM 迭代后可能完全丢失网络尺度信息；仅设置极强先验，则退化为硬固定尺度。

建议策略：

- 低有效点数或高 MAD：不使用 prior；
- 中等稳定性：以较大的 `sigma_D` 添加弱先验；
- 多关键帧一致、在目标相机上已验证：缩小 `sigma_D`，但不建议设为零；
- 在原生 VIBA 1/2 阶段：可继续用弱 prior，也可通过配置关闭，以评估长期 IMU 是否自行收敛。

### 3.6 对 VigInit 的使用方式

最保守的做法是只把方案 A 接入原生 `InitializeIMU()`。`VigInit()` 仍用自身的 `ImuInitializer` 求尺度，然后在初始化后由原生 VIBA 1/2 使用尺度先验继续校正。

若要向 `VigInit` 注入先验，建议只作为诊断或拒绝条件，例如比较：

```text
|log s_{VIG}-log s_D|<tau
```

不建议一开始就把 `ImuInitializer::scale` 直接替换为 `s_D`，因为其随后的地图旋转、bias 回写和 Tracker 更新假设 `scale` 与 IMU 约束一致。

### 3.7 验证指标

对带真值或可测量尺度的数据，记录：

- 首次 IMU 初始化成功所需时间与关键帧数；
- 初始与最终尺度相对误差；
- 重力方向角误差；
- `|b_g|,|b_a|` 及其后续漂移；
- Metric prior 的有效率、`sigma_{MAD}`、被拒绝比例；
- 初始化后 30 秒内 tracking loss、reset、重定位次数；
- 与不使用 prior 的原生路径相比的运行时间。

成功标准不能只看“更早通过初始化”：若更早初始化导致 bias 变大、尺度在 VIBA 后大幅跳变或跟踪更容易丢失，则先验过强或深度域不匹配。

---

## 4. 方案 B：把 Metric3D 当作“伪 RGB-D”

### 4.1 思路

将 Metric3D 提供的每像素深度作为 RGB-D 深度输入，让单目帧获得 `mvDepth`，从而可像 RGB-D 一样在单帧反投影 ORB 特征并建立米制 MapPoint。

它可能显著缩短**视觉地图初始化**所需的视差/基线；但预测深度将直接影响 MapPoint 初值和近远点判断，因此网络误差会比方案 A 更早、更深地进入系统。

这不是当前仓库已有模式：现有 `Frame` 的 RGB-D 构造函数接收真实 `imDepth`，而 `IMU_MONOCULAR` 路径并不会自动调用 Metric3D。需要显式改造 Frame 创建及传感器处理流程。

### 4.2 几何模型

对像素 `(u,v)`，Metric3D 给出 Z-depth `z_D` 时，使用相机内参反投影：

```text
X_c=z_DK^{-1}[u; v; 1]
=[
(u-c_x)z_D/f_x; 
(v-c_y)z_D/f_y; 
z_D
]
```

若相机位姿为 `T_{wc}`，则：

```text
X_w=T_{wc}X_c
```

这正对应当前 `Frame::UnprojectStereo()` 的实现。

如果网络输出的是相机中心的欧氏距离 `r=|X_c|`，不能令 `z_D=r`。对 pinhole 光线：

```text
x_n=(u-c_x)/f_x,         y_n=(v-c_y)/f_y
```

```text
z_D = r / sqrt(1 + x_n^2 + y_n^2)
```

必须通过模型文档、已知平面试验或源码确认网络输出定义。

### 4.3 与当前 `Frame` 语义对齐

当前真实 RGB-D 路径为：

```cpp
Frame::ComputeStereoFromRGBD(const cv::Mat& imDepth)
```

其行为是：

```cpp
d = imDepth.at<float>(v, u);
mvDepth[i] = d;
mvuRight[i] = kpU.pt.x - mbf / d;
```

因此伪 RGB-D 的最小对齐目标是：

1. 在 Frame 构造期间已完成 ORB 特征提取后，给每个有效特征采样 Metric3D 深度；
2. 填写 `mvDepth[i]`；
3. 按现有 RGB-D 约定同步填写 `mvuRight[i]`，保证后续深度可用判断与数据结构一致；
4. 确保 `mvKeys`、`mvKeysUn`、深度图和 `K` 所处坐标系一致。

`mbf/d` 在这里并非真实左右相机测得的 disparity；它只是复用现有 RGB-D 数据通路所需的右像素形式。绝不能将它解释为 Metric3D 产生了双目测量。

### 4.4 推荐实现路径

不建议先修改所有 `Frame` 构造函数。建议新增一个显式方法，避免把网络预测伪装成真实传感器数据而丢失来源信息：

```cpp
// 新增：接口示意
void Frame::AttachPredictedDepth(
    const MetricDepthResult& result,
    float min_depth,
    float max_depth,
    float min_confidence);
```

职责：

1. 验证 `result.frame_id == mnId`；
2. 对每个 ORB keypoint 做坐标转换和双线性采样；
3. 依据 confidence、范围和 mask 筛选；
4. 填充 `mvDepth`/`mvuRight`；
5. 额外记录每个深度的来源和置信度，例如 `mvDepthSource`、`mvDepthConfidence`（新增）；
6. 不对低置信度深度填默认值，保持 `-1`，使现有代码自然忽略。

随后需要为传感器模式增加清晰、独立的配置，例如 `IMU_MONOCULAR_PREDICTED_DEPTH`，而不是静默把 `IMU_MONOCULAR` 改成 `IMU_RGBD`。Tracking/Local Mapping 可针对该模式选择不同的深度筛选与 MapPoint 创建策略。

建议代码流程：

```text
输入图像 + 真实 K / 畸变参数
  -> 与 ORB 特征所用图像一致的预处理
  -> 异步或低频 Metric3D 推理
  -> 输出反变换为真实相机的 Z-depth（m）
  -> Frame::AttachPredictedDepth
  -> 仅用高置信度 keypoints 反投影初始 MapPoints
  -> 普通视觉匹配 / 局部 BA 验证与修正这些 MapPoints
  -> 原生 InitializeIMU 仍估计 g, v, bg, ba；单目 scale 可设为强先验或保留优化
```

### 4.5 初始化策略：不要把全部预测深度当硬事实

建议采用“有限可信深度种子”而不是“所有像素都是真 RGB-D”。具体地：

- 只在首批若干关键帧、深度高置信区域创建深度种子 MapPoint；
- 其他特征仍走标准单目三角化；
- 每个伪深度 MapPoint 必须通过多视图重投影、一致视差和局部 BA 验证；
- 对预测深度创建的点设置更保守的观测门槛与更低初始权重；
- 若后续多视图几何显著反驳网络深度，应允许剔除该点，而不是锁死深度。

若直接把深度当真实 RGB-D 并将尺度固定为 1，隐含假设为：

```text
D_{Metric3D}(u,v)=D_{true}(u,v)
```

这对零样本模型通常不成立。更合理的模型是：

```text
D_{Metric3D}(u,v)=alpha D_{true}(u,v)+beta+epsilon(u,v)
```

其中 `alpha` 可能是场景/焦距相关的尺度偏差，`beta` 和 `epsilon` 则反映近距离、边缘和域偏移误差。故即使采用伪 RGB-D，也应允许原生 `VertexScale` 继续优化，并可同时加入方案 A 的尺度软先验。

### 4.6 可选：在 BA 中作为深度残差，而非改造伪双目

若后续需要更严谨的融合，应新增“预测深度观测边”，连接 MapPoint、KeyFrame pose 与可选的全局 scale，而不是复用 `mvuRight`。形式可写为：

```text
r_{D,ik}=log(s [T_{cw,i}X_k]_z)-log D_i(u_{ik},v_{ik})
```

使用 confidence 和尺度不确定性构造信息：

```text
E_D=sum_{ik}w_{ik}rho(r_{D,ik}^2)
```

其中 `rho` 是 Huber/Cauchy 等鲁棒核。这比伪 RGB-D 更符合“学习深度是不确定观测”的事实，但改动范围更大：需要新的 g2o 边、观测缓存生命周期和 BA 中的像素投影管理。它可以作为方案 B 的第二阶段，而非首版目标。

### 4.7 风险与缓解

| 风险 | 后果 | 缓解方式 |
|---|---|---|
| 内参/canonical 反变换错误 | 全局尺度系统性错误 | 单元测试焦距缩放；标尺/平面验证 |
| resize、crop、畸变坐标不一致 | 深度采样到错误像素 | 显式坐标映射；可视化 keypoint-depth overlay |
| 边缘、反射、天空、动态物体 | 错误 MapPoint 与错误初始结构 | confidence/mask；严格门限；多视图验证 |
| 网络尺度随场景变化 | 地图尺度漂移 | 只作软约束；按 KF 鲁棒聚合；保留 IMU scale 优化 |
| 推理延迟/吞吐不足 | Tracking 阻塞或深度与帧错配 | 异步 worker、frame id 校验、超时即跳过 |
| 把预测值视作硬 RGB-D | 错误结构难以被后端修复 | 标记来源、降低权重、保留剔除机制 |

---

## 5. 方案选择与实验顺序

### 阶段 0：不改 SLAM，仅评估深度

离线保存关键帧、内参和 Metric3D 结果；实现尺度比统计。先回答：在目标环境中 `s_D` 是否跨关键帧稳定？如果不稳定，方案 A/B 均不应进入在线初始化。

### 阶段 1：方案 A 的日志模式

在线计算 `MetricScalePrior`，但不添加 g2o 边；记录它与原生 `mScale`、VIG `initializer.scale` 的 log-scale 差异。该阶段可发现内参、时间戳和坐标映射错误。

### 阶段 2：方案 A 的弱先验

添加 `EdgePriorScaleLog`，从保守的大 `sigma_D` 开始。比较初始化时间、尺度、bias、重力和跟踪稳定性。确认无退化后再调整先验强度。

### 阶段 3：方案 B 的有限深度种子

仅对高置信深度、少量关键帧创建伪 RGB-D 点；不固定 scale；验证局部 BA 与 MapPoint 剔除表现。此阶段失败时可完整回退到方案 A，不应影响原生单目路径。

### 阶段 4：必要时实现预测深度 BA 边

仅当伪 RGB-D 已显示出明确收益、但硬深度注入仍造成局部结构错误时，再实现 4.6 的软深度观测边。

---

## 6. 最终建议

1. 首选方案 A：Metric3D 只产生 log-scale 软先验，保留原生 `InitializeIMU()` 与 Full Inertial BA 的联合估计能力。
2. 不要因具有 Metric3D 尺度就降低 IMU 的重力/bias 可观测性要求；它们来自不同传感器信息。
3. 把 Metric3D 的 canonical 相机反变换、深度定义和像素坐标映射作为上线前硬性验收项。
4. 方案 B 仅在目标场景的绝对深度精度、置信度质量和实时吞吐均已验证时采用，并应将预测深度视作可被多视图几何否决的软信息。

---

## 7. Metric3D 能否降低初始化运动要求？

### 7.1 简短结论

**可以降低对“大平移/大加速度”的依赖，但不能把惯性可观测性要求降到“微小且单一的运动也必然成功”。**

当 Metric3D 的绝对尺度在目标相机和场景中可靠时，它提供了原单目系统缺失的一项信息：

```text
p^{metric}=s_Dp^{mono}
```

这会显著削弱 `scale` 与速度、加速度计 bias 之间的耦合。因此不再一定需要靠大幅度的平移（例如空中划“8”字）来让单目尺度变得可观。

但重力与加速度计 bias 的可观测性来自 IMU 与**姿态/加速度变化**，而不是来自单帧深度。如果运动小到视觉/IMU 噪声量级，或始终保持同一个姿态，任何深度网络都不能凭数据分辨真实重力和 accelerometer bias。

因此正确的目标应是：

```text
“由大幅平移 + 大加速度激励”
        降为
“尺度先验可靠 + 有足够多方向的缓慢旋转/轻微运动 + 通过不确定度验收”。
```

而不是：

```text
“有 Metric3D，所以静止或任意微小运动都可初始化”。
```

### 7.2 为什么已知尺度仍不能在静止时解出 ba 和重力

IMU 加速度计测量的是 specific force。忽略噪声，使用 body-to-world 旋转 `R_{wb,i}` 时可写为：

```text
a^m_i=R_{bw,i}(a^w_i-g)+b_a
```

若设备静止，`a^w_i=0`，则：

```text
a^m_i=-R_{bw,i}g+b_a
(1)
```

若始终没有转动，所有 `R_{bw,i}` 相同。式 (1) 只能观测到组合量：

```text
b_a-R_{bw}g
```

即使 `|g|=9.81` 已知，也存在多组 `(g,b_a)` 能产生同样的静止加速度测量；Metric3D 的尺度不出现在式 (1) 中，因而无法解决该退化。

反之，若设备做了多方向的缓慢旋转，即使平移很小，堆叠多个静止/低动态观测也会形成：

```text

[
-R_{bw,1} & I; 
-R_{bw,2} & I; 
... & ...; 
-R_{bw,N} & I
]  # A
[g; b_a]
=
[a^m_1; a^m_2; ...; a^m_N]
```

配合 `|g|=9.81` 的约束，重力方向与 `ba` 的条件会明显改善。这里的关键是**旋转覆盖多个方向**，不是必须快速或大范围平移。

gyro bias 也需要旋转信息来与真实角速度区分；完全静止时可估计常值 gyro 输出，但这依赖“确实静止”的额外假设，不等价于一般视觉惯性初始化。

### 7.3 对原生 InitializeIMU 的影响

原生 `EdgeInertialGS` 位置、速度残差中的尺度项为：

```text
s * (v_j-v_i),     s * (p_j-p_i-v_i*dt)
```

若方案 A 提供稳定的 `log s` 软先验，或方案 B 已使地图近似米制，优化器无需再主要依靠惯性数据去确定 `s`。这通常会带来：

- 更好的 scale 初值；
- `s` 与速度/`ba` 耦合减弱；
- 小平移时位置残差的数值条件改善；
- 更可能在较早的时间窗口获得稳定的优化解。

但图中仍有每帧速度、重力方向、`bg`、`ba`。它们的误差与协方差没有因为尺度先验而自动消失。因此应继续保留原生流程的时间窗口与后续 VIBA 1/2，只能在实测显示协方差充分下降后再逐步调整启动策略。

### 7.4 对当前 VigInit 的影响：不能只删除门限

当前 `VigInit()` 在调用 `ImuInitializer` 前检查：

```cpp
average ||dP|| >= 0.02 m
stddev(dV / dT) >= 0.4 m/s^2
```

而 `ImuInitializer::SolveGravityScale()` 的三帧方程为：

```text
s*Q - 0.5*t_1*t_2*(t_1+t_2)*g = D
```

其中：

```text
Q = t_1*(hat_p_3-hat_p_2) - t_2*(hat_p_2-hat_p_1)
```

若平移很小，`Q` 接近零，当前方程对 `s` 的信息很弱；若加速度变化很小，重力与 bias 的条件也会变差。这正是当前门限存在的原因。

即使外部给定 `s_D`，也不能只把这两个 `return false` 删除：当前三帧消元模型仍假设从 IMU/视觉运动中恢复重力和 bias，病态的正规方程依旧可能产生数值有限但物理错误的解。

若希望让 VIG-Init 利用 Metric3D，应进行模型级改造，而不是门限绕过：

1. 将尺度固定或以 `log s_D` 软先验加入其最小二乘问题；
2. 保留 `ba`、重力方向的可观测性测试；
3. 使用旋转覆盖度、IMU 信息矩阵条件数或预测协方差作为接受条件；
4. 对低激励结果只发布“尺度已知、IMU 未初始化”的中间状态，待惯性条件满足后再写入 bias、旋转地图并置 `isImuInitialized()`。

从实现成本和稳健性看，更建议先将 Metric3D 尺度先验加到原生 `InitializeIMU()`，而不是先修改 `ImuInitializer`。

### 7.5 建议用“可观测性/不确定度”替代固定的大动作要求

固定的关键帧数、位移和加速度阈值简单但保守。引入 Metric3D 后，可以在保留安全下限的前提下，增加基于实际数据质量的验收：

1. 运行带尺度先验的原生惯性优化；
2. 从优化后的正规方程/Hessian 取与 `(b_g,b_a,gravity direction,s)` 对应的边缘信息；
3. 检查其最小特征值、条件数，或逆矩阵对角线给出的预测标准差；
4. 只有尺度、重力方向、bias 的不确定度均低于配置阈值时，才提交初始化结果。

若记边缘状态为 `x=[b_g^T,b_a^T,theta_g^T,log s]^T`，其近似协方差为：

```text
Sigma_x approx inverse(H_x)
```

可使用以下量作为拒绝条件：

```text
lambda_min(H_x) 过小
```

```text
kappa(H_x) = lambda_max(H_x) / lambda_min(H_x) 过大
```

```text
sqrt(Sigma_x,kk) 超过对应状态的允许不确定度
```

阈值必须以真实传感器噪声、目标场景和离线实验标定，不应照搬本文或其他设备的数值。此方法的意义不是让不可观问题“变得可观”，而是在出现足够旋转/轻微运动时，比“必须大位移”更准确地识别可接受窗口。

### 7.6 现实可行的低运动启动动作

若希望避免空中大幅划 8 字，推荐的动作不是完全不动，而是几秒内缓慢改变相机朝向：俯仰、横滚、偏航中至少覆盖两个以上方向，并夹带自然的小平移。这样的动作：

- 给 gyro bias 和视觉-IMU 旋转一致性提供信息；
- 通过不同 `R_{bw}` 改善重力与 `ba` 的分离；
- 在 Metric3D 已提供尺度时，不再必须依赖大位移来恢复 scale；
- 对视觉特征匹配通常也比纯原地剧烈旋转更友好。

如果使用场景是固定安装、启动时几乎静止，则应考虑引入额外先验而非期望 SLAM 自动估计所有量，例如：出厂标定的 `ba/bg`、已知安装朝向、或一个明确的静止检测 + bias 初始化阶段。这些做法本质上是增加先验；应在系统配置和状态机中明确表达，不能伪装为“由 Metric3D 数据自动求出”。

---

## 8. VigInit 不做 VIBA 1/2 时：在 ImuInitializer 内加入 Metric3D 尺度先验

### 8.1 适用背景与总原则

当前 `VigInit()` 的流程是：先通过 bias-only 的 g2o `InertialOptimization()` 取得 `bg`，然后由 `ImuInitializer` 依次执行：

```text
SolveGravityScale()
SolveScaleAccelBias()
Refine()
```

最后立即缩放/旋转地图并设置 `isImuInitialized()`。若不再执行后续 VIBA 1/2，则这次解就是系统的主要惯性初值，错误尺度和 bias 缺少后续全局优化来纠正。

因此 Metric3D 应直接成为 `ImuInitializer` 三个阶段都使用的**软尺度先验**，而不是仅在 `VigInit()` 成功后进行日志比较。与此同时，网络 prior 与惯性解显著冲突时必须拒绝初始化，而不是强行接受任一方。

### 8.2 由关键帧深度估计 MetricScalePrior

Metric3D 的输入应是已保存的关键帧图像；其深度需要完成 canonical camera 到真实相机内参的反变换，并与 ORB keypoint 所在的像素坐标系一致。

对每个有 Metric3D 缓存结果的关键帧：

1. 通过 `KeyFrame::GetMapPointMatches()` 取得每个特征索引关联的 MapPoint；
2. 取 MapPoint 世界坐标 `X_w`，由关键帧相机位姿计算视觉地图深度；
3. 在该特征像素处对 Metric3D 的米制 Z-depth 进行双线性采样；
4. 对有效观测计算 log-scale 样本；
5. 先按关键帧聚合，再跨关键帧鲁棒聚合。

计算式为：

```text
z_slam_ik   = (T_cw_i * X_w_k).z
z_metric_ik = D_metric_i(u_ik, v_ik)
xi_ik       = log(z_metric_ik) - log(z_slam_ik)
```

每帧先获得 `xi_i = weighted_median(xi_ik)`；再获得：

```text
xi_D = weighted_median(xi_i)
s_D  = exp(xi_D)
sigma_log_s = 1.4826 * median(|xi_i - xi_D|)
```

建议新增如下数据对象；这不是当前仓库已有接口：

```cpp
struct MetricScalePrior {
    bool valid = false;
    float scale = 1.f;          // s_D, metric / current visual-map scale
    float sigma_log_scale = 0.f;
    int num_keyframes = 0;
    int num_observations = 0;
};
```

仅在多帧尺度一致、有效点数足够、深度 confidence 合格、`scale > 0` 且有限时设置 `valid=true`。深度不可靠时必须发布 `valid=false`，使系统回退到当前 VigInit 行为。

### 8.3 将 prior 传递给 ImuInitializer

建议扩展构造函数，而不是使用全局变量：

```cpp
// 建议新增的签名
ImuInitializer(const std::vector<KeyFrame*>& keyframes,
               const Eigen::Vector3f& gyro_bias,
               const MetricScalePrior& scale_prior);
```

在 `VigInit()` 内，`Optimizer::InertialOptimization()` 求得 `gyroBias` 后、创建 `ImuInitializer` 前，计算当前初始化窗口的 prior：

```cpp
const MetricScalePrior scalePrior =
    EstimateMetricScalePrior(initializationKeyframes);

ImuInitializer initializer(initializationKeyframes,
                           gyroBias.cast<float>(),
                           scalePrior);
```

`EstimateMetricScalePrior()` 应位于 Local Mapping 或独立的 MetricDepth 模块，仅读取 KeyFrame、MapPoint 和不可变的深度缓存；不要在 `ImuInitializer` 或 g2o 优化器中运行深度网络。

### 8.4 三个求解阶段的正规方程增量

当前每个阶段都累计普通最小二乘：

```text
minimize: sum_k || C_k * x - d_k ||^2
```

Metric3D 给出的是 log-scale 先验：

```text
r_metric = log(s) - log(s_D)
```

在 `s_D` 附近一阶线性化：

```text
r_metric approx (s - s_D) / s_D
```

将其表示为线性 pseudo-measurement：

```text
sqrt(w_s) * s = sqrt(w_s) * s_D
```

它对正规方程的贡献是：

```text
A(scale_index, scale_index) += w_s
b(scale_index)            += w_s * s_D
```

因此可新增一个私有辅助函数，例如：

```cpp
template<int N>
void ImuInitializer::AddMetricScalePrior(
    Eigen::Matrix<float, N, N>& A,
    Eigen::Matrix<float, N, 1>& b,
    int scale_index) const;
```

当 `scale_prior.valid == false` 时该函数为空操作，保持现有解完全不变。

三个调用点分别是：

| 函数 | 当前未知量 x | scale_index |
|---|---|---:|
| `SolveGravityScale()` | `[g_x, g_y, g_z, s]` | 3 |
| `SolveScaleAccelBias()` | `[s, ba_x, ba_y, ba_z]` | 0 |
| `Refine()` | `[s, delta_ba_x, delta_ba_y, delta_ba_z, delta_g_1, delta_g_2]` | 0 |

尤其是 `Refine()`：当前实现中 `scale = x[0]`，即该分量是绝对尺度而不是尺度增量，所以 prior 仍应直接约束 `s` 到 `s_D`。每轮 `Reintegrate()` 后都要重新加入 prior，避免尺度在 refinement 中再次漂移。

### 8.5 prior 权重不能机械等于 1 / sigma_log_scale^2

`ImuInitializer` 当前的三帧惯性约束没有按协方差白化，所有约束直接以 `C.transpose() * C` 累积。因此其残差尺度与 `log(scale)` prior 并不天然同量纲。

不建议直接写：

```text
w_s = 1 / sigma_log_scale^2
```

更适合首版的工程形式是：

```text
w_s = lambda_metric / sigma_log_scale^2
```

其中 `lambda_metric` 是配置项，必须以目标相机、IMU 噪声和场景数据进行标定。建议把它从较弱的值开始做消融实验，并记录 prior 是否让最终 `scale`、`ba`、重力方向更稳定。

更严谨但改动更大的方案，是推导每个三帧消元约束的协方差并白化 IMU 约束，再按统计意义直接使用 `1 / sigma_log_scale^2`。在没有完成该白化前，`lambda_metric` 必须明确为经验正则化权重，不应被解释为严格概率信息。

### 8.6 无 VIBA 时增加 prior-IMU 冲突拒绝

当前 `VigInit()` 主要检查 `scale` 范围、`ba.norm()`、`bg.norm()` 和有限性。无 VIBA 1/2 时还应检查 Metric3D prior 和 VIG 解的一致性：

```text
z_scale = |log(scale_vig) - log(s_D)| / sigma_log_s
```

若 `z_scale` 超过经验证的阈值，应返回失败或延后初始化；不能直接执行：

```cpp
ApplyScaledRotation(...)
UpdateFrameIMU(...)
SetImuInitialized()
```

冲突可能意味着：Metric3D 尺度不适用于当前场景、深度图/内参/裁剪映射错误、IMU 时间同步或外参错误、视觉地图已经漂移，或当前 IMU 激励不足。

无 VIBA 时，保守拒绝比接受一个不一致的初值更安全，因为后续缺少全局惯性 BA 来修正错误的尺度与 bias。

### 8.7 运动门限的处理

即使有 Metric3D scale prior，也不能简单删除 `ImuInitializer::Initialize()` 中的：

```text
average preintegrated displacement threshold
stddev(dV / dT) threshold
```

Metric3D 只弱化尺度未知带来的退化；当前 `ImuInitializer` 的三帧位置/速度消元方程仍依赖运动信息估计重力和 `ba`。特别是 `dV / dT` 的变化不足时，重力与加速度计 bias 仍可能病态。

建议分阶段推进：

1. 首版保留两个现有门限，仅验证 prior 是否改善尺度、bias 和初始化成功率；
2. 仅在 prior 多帧稳定时，实验性地下调与尺度相关的位移门限；
3. 保留加速度变化门限，除非另行实现利用多姿态静止观测的 `gravity/ba` 求解器；
4. 用正规方程条件数、最小特征值或边缘协方差替代“固定大动作”的最终接受条件。

### 8.8 不做 VIBA 的最小安全实现顺序

```text
1. 异步 Metric3D，缓存每个关键帧的 metric Z-depth 与 confidence
2. 从初始化窗口估计 MetricScalePrior；不稳定则 prior invalid
3. 将 prior 注入 ImuInitializer 的三个阶段
4. 保持现有 IMU 激励门限
5. 增加 prior-IMU 尺度冲突拒绝
6. 只在通过所有检查时执行现有地图缩放、重力对齐与 bias 回写
7. 记录 scale、ba、bg、gravity、prior 方差和冲突统计，完成消融评估后再调权重/门限
```

不建议首版将 Metric3D 深度直接作为伪 RGB-D，也不建议将 `scale` 硬固定为 Metric3D 输出。没有 VIBA 收尾时，“软 prior + 冲突拒绝 + 保守运动门限”是更可控的组合。
