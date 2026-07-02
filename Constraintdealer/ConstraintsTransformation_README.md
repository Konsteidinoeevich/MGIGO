# 约束转化方法论：从约束优化到黑箱标量代价

## 1. 自相似 σ 嵌套

$$\mathcal{L}(x) = \sqrt{2}\cdot\sigma_1\!\left(\cdots \sqrt{2}\cdot\sigma_1\!\left(\sigma_{k_{\text{in}}}\!\left(\frac{T_{\text{obj}}(f(x))}{\sqrt{2}^{\,n+1}}\right) + \Phi_1\right) + \cdots + \Phi_n\right)$$

$$\Phi_i = \max(0, T(g_i)) + \text{baseline}_i \quad (\text{if } g_i > \text{resolution}_i)$$

### 关键属性

1. **自相似**：$\sqrt{2}\cdot\sigma_1(\sigma_1(x/\sqrt{2})\cdot\sqrt{2}) = \sigma_1(x)$，任意层数
2. **$\Phi=0$ 透明**：所有约束满足时，多层 = 单层 = $\sigma(T_{\text{obj}}(f))$
3. **baseline 编码硬度**：0.5(SOFT), 1.3(TUNABLE), 2.0(HARD)，连续可调
4. **内层放大**：小 priority = 内层，被后续 $\sigma\cdot\sqrt{2}$ 放大 → 影响大
5. **输出有界**：$(-\sqrt{2}, \sqrt{2}) \approx (-1.41, 1.41)$

## 2. 数学基础

### 2.1 饱和函数

$$\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}$$

全链使用 $\sigma_1$。$k$ 只在最内层控制目标函数饱和速度。

### 2.2 T_alpha 变换

$$T_\alpha(g) = \text{sign}(g) \cdot T_{\text{target}}(|g|)$$

三段式：地板（$g<$resolution，$T$ 恒定）→ log 增长 → 缓坡天花板。

| 模式 | resolution | T_floor | T_ceil |
|------|-----------|---------|--------|
| SOFT | $10^{-2}$ | 0.003 | 4.5 |
| TUNABLE | $10^{-4}$ | 0.02 | 6.0 |
| HARD | $10^{-6}$ | 0.08 | 6.5 |

### 2.3 baseline — 硬度旋钮

$$\Phi = \begin{cases} 0 & g \le \text{resolution} \\ \max(0, T(g)) + \text{baseline} & g > \text{resolution} \end{cases}$$

| mode | baseline | 违规 Δ | 语义 |
|------|----------|--------|------|
| SOFT | 0.5 | ~0.23 | 柔和 |
| TUNABLE | 1.3 | ~0.40 | 清晰 |
| HARD | 2.0 | ~0.46 | 严格 |

## 3. 数值验证

- $\Phi=0$ 透明：任意层数 = $\sigma_1$，误差 < 1e-15
- 内层违规 Δ > 0.29（远超求解器 1e-3 精度）
- $k_{\text{in}}=1.0$，N=50 无精度损失

## 4. 代码索引

| 函数/变量 | Constran.py | 说明 |
|----------|------------|------|
| `T_alpha()` | L122 | 多段 log-like 变换 |
| `sigma_k()` | L259 | 饱和函数 |
| `_assemble_nest()` | L415 | 自相似嵌套 |
| `build()` | L446 | 公共 API |
| `ConstraintSpec.baseline` | L274 | 硬度旋钮 (0.5/1.3/2.0) |
| `TRANSFORM_SOFT` | L45 | SOFT T 表 |
| `TRANSFORM_TUNABLE` | L50 | TUNABLE T 表 |
| `TRANSFORM_HARD` | L55 | HARD T 表 |
| `Deterministic` | L330 | 确定性约束 |
| `Chance` | L335 | 机会约束 |
| `Robust` | L348 | 鲁棒约束 |
| `DRO` | L355 | 分布鲁棒约束 |

## 5. B-spline 时域约束

推荐 `aggregate='q95'`（95% 分位数），比 `max` 更鲁棒，比 `mean` 更严格。
