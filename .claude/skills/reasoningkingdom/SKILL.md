---
name: reasoningkingdom
description: 学习的地形编辑部技能 —— 兔狲教授的几何动力学实验室（《学习的地形：推理几何与智能动力学导论》）
---

# 学习的地形编辑部技能

## 本书概览

**《学习的地形：推理几何与智能动力学导论》**

| 属性 | 内容 |
|------|------|
| 定位 | 《推理王国》的几何学 |
| 读者 | 有深度学习基础，追求几何直觉与动力系统理解 |
| 兔狲教授风格 | 犀利、几何化思维，用空间语言描述一切 |
| 核心命题 | 学习有地形，推理有路径，智能有力学 |

**理论主线**：Space → Landscape → Dynamics → Fixed Point → Reasoning

---

## 角色设定

### 兔狲教授（Pallas's Cat Professor）

行事独立、目光锐利，用几何语言思考一切：

- 说话直接，偶尔说"你把这当优化，我把这当地形运动"
- 对"梯度下降就是调参"这类说法不客气
- 对真正好奇内部结构的人无限耐心
- 喜欢用地形图、相图、吸引子打碎直觉
- 在关键处停顿，让读者自己感受空间

### 作者

李籽溪（中山大学，广州，2026年）

---

## 四卷结构（共13章）

### 卷一：学习的地形（3章）

建立几何直觉。从"模型在哪里学习"出发，讲欧几里得空间、参数空间、表示空间、损失地形、梯度场、欧拉步、山谷鞍点极小值。

| 章 | 标题 | 内容 |
|----|------|------|
| ch1 | 为什么学习需要几何 | 几何必要性 + 空间、点、向量与距离 |
| ch2 | 身体与视野：参数空间与表示空间 | 参数空间（模型的身体）+ 表示空间（模型的视野） |
| ch3 | 损失地形与梯度运动 | 损失函数作为地形 + 梯度方向 + 欧拉步 + 山谷/鞍点/极小值 |

### 卷二：行走与信念（4章）

优化器作为行走方式，非欧距离，推理轨迹，动力系统与不动点。核心公式：$S_{t+1} = S_t + \eta F_\theta(S_t, x)$

| ch4 | 行走的方式：优化器与正则化 | 学习率 + 优化器 + SGD→Adam + 正则化 |
| ch5 | 非欧距离：Bregman与KL | Bregman散度 + KL散度：信念空间里的偏差 |
| ch6 | 推理轨迹与动力系统 | 推理作为轨迹 + 从地形到地貌 + 学习作为动力系统 |
| ch7 | 不动点：从欧拉法到信念收敛 | ResNet残差加性更新=显式欧拉；GPT因果自回归（输出→输入）=隐状态空间的欧拉法迭代；DEQ=内部迭代至不动点 $h^* = f_\theta(h^*, x)$；信念不动点 $D_{\mathrm{KL}}(p_{t+1}(y|x) \| p_t(y|x)) < \epsilon$ |

### 卷三：推理的几何（3章）

核心命题：**思维链不是推理本身，而是推理轨迹的可见投影。** CoT训练改变的不是输出格式，而是内部表示空间的地形结构。

| ch8 | 思维链：推理轨迹的投影 | 推理为何不是答案 + 显式/隐式链 + 读题时已运动 + CoT作为轨迹投影 |
| ch9 | 推理场：吸引子与验证器 | 答案概率 + 信念更新KL + 吸引子 + 坏吸引子 + 验证器改变地形 + RLHF几何 |
| ch10 | 长推理与地貌重塑 | 长推理=长轨迹 + 推理模型的地貌重塑 |

### 卷四：算法的地貌学（3章）

经典算法几何化。每章：一张大图 + 一张公式 + 一张直觉图 + 一段代码实验。

| ch11 | 经典算法的几何 | 线性回归（二次碗形）+ PCA/SVD（数据云主轴）+ SVM（最大间隔） |
| ch12 | 现代深度架构的几何 | Attention（token间软几何连接）+ LoRA（低秩子空间参数运动） |
| ch13 | 扩散与收敛 | Diffusion（噪声空间→数据流形的反向动力学） |

---

## 写作策略：夹心饼干（Sandwich）

全书统一采用三层夹心结构，ch03 为范式模板。

### 第一层：叙事开头（~1–2 节）

用一个具象的、身体化的比喻打开核心问题。不让读者看到公式，而是看到一个场景。

- ch03 开头：蒙眼空降到荒野中，脚下只有坡度信号，任务是找到最低点
- 引出核心张力：地形一样，走路方式不同，终点就不同
- 揭示本章要回答的问题链

### 第二层：形式化定义（~5–8 节）

严格的形式化内容，每节一个核心数学对象。**关键：节与节之间必须有叙事粘合层**——一段 2–4 句的过渡文字，用同一个比喻（如荒野徒步）串联"为什么从上一节走到这一节"。

形式化层的节奏：
- 定义 → 几何直觉 → 图 → **粘合过渡** → 下一个定义
- 粘合过渡不是"接下来我们讲 X"，而是"你有了 A 之后，自然会问 B——这就是我们要讲的"

### 第三层：叙事收尾（~1–2 节）

用完整的比喻叙事把形式化层中的所有概念串联成一个故事。不是重复定义，而是让定义在故事中活起来。

- ch03 收尾：从随机初始化到最终盆地的完整徒步叙事
- 最后是：小结 → 12 个悬而未决的问题 → :::info 核心追问

### 篇幅指南

| 类型 | 节数 | 适用 |
|------|------|------|
| 长章 | ~15–20 节 | 合并 3–4 个原始短章的重型章节（如 ch3, ch6） |
| 标准章 | ~10 节 | 合并 2 个原始短章的标准章节（如 ch1, ch2, ch4） |

### :::info 块的三种用途

1. **叙事解释**：在形式化跳跃处插入具体例子、数值演示、生活比喻，不超 15 行，不引入新定义
2. **跨章桥接**：显式连接前面章节的平行结构（如 ch5 的"熵地形↔损失地形↔牛顿→哈密顿"三层桥接）
3. **兔狲教授的立场评论**：每章至少一处。用独立、锐利的几何语言，对本章的核心命题发表立场。语气直接，偶尔不客气。禁止"显然""众所周知"，要求从具体结构出发而非空泛假设。典型句式："你把这当优化，我把这当地形运动。"

### 每节内部结构

1. **直觉锚点**：一句话点明本节要回答什么
2. **核心公式**：嵌在叙事中，不单独陈列
3. **几何解释**：每个符号出现前先用空间/地形语言描述
4. **过渡句**：最后一句话自然引出下一节的问题

### 语言原则

- **禁止**："梯度下降就是调参""接下来我们讨论 X""显然""众所周知"
- **要求**：用身体化的比喻（荒野徒步、身体移动、视野清晰化），每个概念出现前先有空间直觉
- **风格**：介于 3Blue1Brown 几何直觉与数学物理教材之间——公式精确，叙事驱动

---

## 项目结构

```
the-terrain-of-learning/
├── docs/
│   ├── 01_the_terrain_of_learning/    # 卷一 ch01-ch03
│   ├── 02_walking_and_belief/         # 卷二 ch04-ch07
│   ├── 03_geometry_of_reasoning/      # 卷三 ch08-ch10
│   ├── 04_algorithmic_landscapes/     # 卷四 ch11-ch13
│   ├── figures/                       # landscapes/ vector_fields/ manifolds/ attractors/
│   ├── notebooks/                     # Jupyter实验原型
│   └── index.md
├── scripts/                           # plot_loss_landscape.py 等可视化脚本
└── .vitepress/config.mts
```

---

## 技术规范

### 数学公式
- 行内：`$ ... $`
- 块级：`$$ ... $$`，单独成行
- 中文：`\text{中文}`
- 不用 `\[ \]` 或 `\( \)`

### 可视化规范（Nature 出图标准）

所有图均按 Nature 印刷规格制作。每图必须导出 SVG（矢量无损）。

**尺寸**
- 单栏：88 mm = 3.46 in
- 双栏：180 mm = 7.09 in
- 高度上限：247 mm

**分辨率**
- SVG/PDF 导出首选（无限分辨率）
- 位图备用：线条图 1200 dpi，组合图 600 dpi

**字体**：Helvetica / Arial，6–8 pt，全 sans-serif，禁止 serif

**线宽**：数据线 0.5–1 pt，坐标轴 0.5 pt

**配色**：Okabe-Ito 色盲友好 8 色
`#E69F00` `#56B4E9` `#009E73` `#F0E442` `#0072B2` `#D55E00` `#CC79A7` `#000000`
禁止纯红绿对比。

**全局样式文件** `scripts/style.py`（所有绘图脚本第一行 import）：

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE = 3.46   # 88mm，Nature 单栏
DOUBLE = 7.09   # 180mm，Nature 双栏

COLORS = ['#E69F00','#56B4E9','#009E73','#F0E442',
          '#0072B2','#D55E00','#CC79A7','#000000']

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial'],
    'font.size':         7,
    'axes.linewidth':    0.5,
    'axes.labelsize':    7,
    'xtick.labelsize':   6,
    'ytick.labelsize':   6,
    'lines.linewidth':   1.0,
    'figure.dpi':        300,
    'savefig.dpi':       600,
    'savefig.format':    'svg',
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.01,
})
```

用法：
```python
from style import SINGLE, DOUBLE, COLORS
fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.75))
ax.set_prop_cycle(color=COLORS)
fig.savefig('figures/loss_landscape.svg')
```

**各图类型工具**
| 图类型 | 工具 | 导出 |
|--------|------|------|
| 损失地形（3D曲面）| matplotlib Axes3D | SVG/PDF |
| 梯度/向量场 | matplotlib quiver | SVG |
| 流形可视化 | matplotlib 3D 参数曲面 | PDF |
| 动力系统相图 | matplotlib | SVG |
| 吸引子轨迹 | matplotlib | SVG |
| 计算图 | Graphviz Digraph，`splines='ortho', rankdir='LR'`，方形节点，Courier 字体 | SVG |

### 动手出片：跑起来看看

遇到任何几何概念，先跑个图出来再说。以下是几个快速原型，拿到 notebook 或脚本里直接跑。

**损失地形（二维碗）**
```python
from style import SINGLE, COLORS
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, x)
Z = X**2 + 2*Y**2  # 椭圆碗，模拟各向异性 loss

fig, ax = plt.subplots(figsize=(SINGLE, SINGLE), subplot_kw={'projection': '3d'})
ax.plot_surface(X, Y, Z, cmap='Blues', alpha=0.85, linewidth=0)
ax.set_xlabel('w₁'); ax.set_ylabel('w₂'); ax.set_zlabel('Loss')
fig.savefig('figures/loss_bowl.svg')
```

**梯度场 + 欧拉步轨迹**
```python
from style import SINGLE, COLORS
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-2.5, 2.5, 20)
X, Y = np.meshgrid(x, x)
GX, GY = 2*X, 4*Y  # ∇Loss of X²+2Y²

# 欧拉步轨迹
eta = 0.1
pos = np.array([2.0, 1.5])
traj = [pos.copy()]
for _ in range(30):
    pos -= eta * np.array([2*pos[0], 4*pos[1]])
    traj.append(pos.copy())
traj = np.array(traj)

fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
ax.quiver(X, Y, -GX, -GY, alpha=0.35, color=COLORS[1], width=0.003)
ax.plot(traj[:,0], traj[:,1], color=COLORS[0], lw=1.2, zorder=5)
ax.scatter(*traj[0], color=COLORS[5], s=20, zorder=6)
ax.scatter(*traj[-1], color=COLORS[2], s=20, zorder=6)
ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
fig.savefig('figures/euler_steps.svg')
```

**不动点迭代（DEQ 直觉）**
```python
from style import SINGLE, COLORS
import numpy as np, matplotlib.pyplot as plt

f = lambda h: np.tanh(0.8 * h + 0.5)  # 模拟一层 DEQ
h_vals = np.linspace(-3, 3, 300)

fig, ax = plt.subplots(figsize=(SINGLE, SINGLE))
ax.plot(h_vals, f(h_vals), color=COLORS[0], lw=1.2, label='f(h)')
ax.plot(h_vals, h_vals,    color=COLORS[7], lw=0.7, ls='--', label='h = h')

# 蜘蛛网迭代
h = 2.5
for _ in range(12):
    fh = f(h)
    ax.plot([h, h], [h, fh], color=COLORS[1], lw=0.8, alpha=0.7)
    ax.plot([h, fh], [fh, fh], color=COLORS[1], lw=0.8, alpha=0.7)
    h = fh

ax.set_xlabel('h'); ax.legend(fontsize=6)
fig.savefig('figures/fixed_point_cobweb.svg')
```

**KL散度信念流**
```python
from style import SINGLE, COLORS
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 500)
mu_target = 1.0

fig, ax = plt.subplots(figsize=(SINGLE, SINGLE * 0.7))
for i, mu in enumerate(np.linspace(-2, 1, 6)):
    alpha = 0.3 + 0.14 * i
    ax.plot(x, norm.pdf(x, mu, 1), color=COLORS[0], lw=0.8, alpha=alpha)
ax.plot(x, norm.pdf(x, mu_target, 1), color=COLORS[2], lw=1.5, label='target p')
ax.set_xlabel('y'); ax.legend(fontsize=6)
fig.savefig('figures/kl_belief_flow.svg')
```

所有脚本放 `scripts/`，输出自动落到 `docs/figures/`。出图后直接在 `.md` 里 `![](../figures/xxx.svg)` 引用。

## pgfplots 3D 绘图规范（主力方案）

本书核心是几何景观——损失盆地、参数空间、能量曲面、动力学轨迹。**pgfplots 3D** 是主力方案。

### 统一 preamble

```latex
\documentclass[tikz,border=12pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
```

### 3D 曲面模板

```latex
\begin{tikzpicture}
\begin{axis}[
  view={55}{35},
  xlabel={$\theta_1$},
  ylabel={$\theta_2$},
  zlabel={$L(\theta)$},
  domain=-2:2, y domain=-2:2,
  samples=35, samples y=35,
  colormap/viridis,
]
\addplot3[surf, opacity=0.55] {x^2 + y^2};

% Optional: trajectory on surface
\addplot3[very thick, red!60!black, mark=*, mark size=1.5pt]
coordinates { (-1.8, 1.5, 4.365) (-0.7, 0.55, 0.64) (0, 0, 0) };
\end{axis}
\end{tikzpicture}
```

### 编译管线

```bash
cd docs/public/figures/tikz
pdflatex -interaction=nonstopmode NAME.tex
pdftocairo -svg NAME.pdf NAME.svg
cp NAME.svg ../chXX_description_tikz.svg
```

### 已有机群（11 张，子代理批量生成）

| 文件 | 章 | 函数/内容 |
|------|-----|-----------|
| `ch01_anisotropic_landscape_tikz.svg` | ch01 | $L=\theta_1^2+3\theta_2^2$ 各向异性 |
| `ch02_manifold_projection_tikz.svg` | ch02 | 流形投影 + 测地线 |
| `ch03_loss_bowl_tikz.svg` | ch03 | $L=\theta_1^2+\theta_2^2$ 损失碗 |
| `ch03_loss_bowl_trajectory_tikz.svg` | ch03 | 同上 + 下降轨迹 |
| `ch04_optimizer_paths_tikz.svg` | ch04 | SGD/Momentum/Adam 三轨迹对比 |
| `ch05_kl_surface_tikz.svg` | ch05 | KL 散度非对称曲面 |
| `ch06_phase_portrait_tikz.svg` | ch06 | 双阱势 $(\theta_1^2-1)^2+\theta_2^2$ |
| `ch07_fixed_point_basin_tikz.svg` | ch07 | 环状不动点盆地 $(\theta_1^2+\theta_2^2-1)^2$ |
| `ch08_attractor_basins_tikz.svg` | ch08 | 三吸引子盆地（正确/错误/混淆） |
| `ch09_rough_vs_smooth_tikz.svg` | ch09 | 崎岖 vs 平滑地形 |
| `ch11_attention_geometry_tikz.svg` | ch11 | softmax 注意力景观 |

### 风格规则

- 坐标轴：黑色
- 曲面：viridis 色图，opacity 0.55–0.65
- 轨迹/标记：高饱和色（红、橙、蓝），无灰色
- 标注：英文数学符号
- 视角：view={50-60}{30-40}

### 与 matplotlib 分工

| 需求 | 工具 |
|------|------|
| 3D 损失景观/能量曲面/盆地 | **pgfplots**（子代理批量） |
| 动力学示意（向量场/轨迹/不动点/相图） | **TikZ 2D**（视觉语法见下） |
| 2D 数据图（loss 曲线/散点/直方图） | **matplotlib**（现有流程） |

## TikZ 2D 动力学视觉语法（十类）

描述动力系统需要一套固定视觉语法。核心问题：状态怎么动？为什么这样动？最后往哪里去？

### 第 1 类：状态轨迹图

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw[->] (-0.5,0) -- (5,0) node[right] {$x_1$};
  \draw[->] (0,-0.5) -- (0,3.5) node[above] {$x_2$};
  \coordinate (x0) at (0.7,3.0);
  \coordinate (x1) at (1.5,2.2);
  \coordinate (x2) at (2.2,1.6);
  \coordinate (x3) at (2.8,1.1);
  \coordinate (x4) at (3.2,0.8);
  \draw[->, thick] (x0) -- (x1);
  \draw[->, thick] (x1) -- (x2);
  \draw[->, thick] (x2) -- (x3);
  \draw[->, thick] (x3) -- (x4);
  \fill (x0) circle (2pt) node[left] {$x_0$};
  \fill (x1) circle (2pt) node[above] {$x_1$};
  \fill (x2) circle (2pt);
  \fill (x3) circle (2pt);
  \fill (x4) circle (2pt) node[right] {$x_4$};
\end{tikzpicture}
```

表达 $x_{t+1} = F(x_t)$。动力学不是一次计算，是一串状态更新。

### 第 2 类：向量场图

```latex
\begin{tikzpicture}[scale=1.0, >=Stealth]
  \draw[->] (-3,0) -- (3.3,0) node[right] {$x$};
  \draw[->] (0,-3) -- (0,3.3) node[above] {$y$};
  \foreach \x in {-2,-1,0,1,2} {
    \foreach \y in {-2,-1,0,1,2} {
      \draw[->] (\x,\y) -- ({\x - 0.25*\x}, {\y - 0.25*\y});
    }
  }
  \fill (0,0) circle (2pt) node[below right] {$x^\ast$};
\end{tikzpicture}
```

空间里每个点倾向于怎么走。固定点、收敛、吸引域的核心图。

### 第 3 类：能量景观 + 下降轨迹

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw (0,0) ellipse (2.6 and 1.3);
  \draw (0,0) ellipse (1.8 and 0.85);
  \draw (0,0) ellipse (1.0 and 0.45);
  \fill (0,0) circle (2pt) node[below] {$x^\ast$};
  \coordinate (a) at (-2.2,1.0);
  \coordinate (b) at (-1.5,0.7);
  \coordinate (c) at (-0.9,0.4);
  \coordinate (d) at (-0.35,0.18);
  \draw[->, thick] (a) -- (b);
  \draw[->, thick] (b) -- (c);
  \draw[->, thick] (c) -- (d);
  \draw[->, thick] (d) -- (0,0);
  \fill (a) circle (2pt) node[left] {$x_0$};
  \node at (2.8,1.2) {$E(x)$};
\end{tikzpicture}
```

表达 $x_{t+1} = x_t - \eta\nabla E(x_t)$。梯度下降、KL Lyapunov、Bregman 收缩。

### 第 4 类：Euler step 图

```latex
\begin{tikzpicture}[scale=1.2, >=Stealth]
  \draw[->] (-0.5,0) -- (4,0) node[right] {$x$};
  \draw[->] (0,-0.5) -- (0,3) node[above] {$y$};
  \coordinate (xt) at (1,1);
  \coordinate (v) at (2.2,1.7);
  \coordinate (next) at (3.2,2.7);
  \fill (xt) circle (2pt) node[below left] {$x_t$};
  \draw[->, thick] (xt) -- (v) node[midway, below] {$v(x_t)$};
  \draw[->, very thick] (xt) -- (next) node[midway, above] {$\eta v(x_t)$};
  \fill (next) circle (2pt) node[above right] {$x_{t+1}$};
  \node at (2.4,0.5) {$x_{t+1}=x_t+\eta v(x_t)$};
\end{tikzpicture}
```

当前位置 + 步长 × 方向场 = 下一步。

### 第 5 类：固定点 / 吸引子图

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw[->] (-3,0) -- (3,0) node[right] {$x_1$};
  \draw[->] (0,-2.5) -- (0,2.5) node[above] {$x_2$};
  \fill (0,0) circle (3pt) node[below right] {$x^\ast$};
  \draw[->, thick] (-2,1.6) .. controls (-1.2,1.0) and (-0.5,0.4) .. (0,0);
  \draw[->, thick] (2,1.4) .. controls (1.1,0.8) and (0.4,0.3) .. (0,0);
  \draw[->, thick] (-2,-1.4) .. controls (-1.1,-0.7) and (-0.4,-0.2) .. (0,0);
  \draw[->, thick] (2,-1.5) .. controls (1.2,-0.8) and (0.5,-0.3) .. (0,0);
  \node at (0,-2.8) {$F(x^\ast)=x^\ast$};
\end{tikzpicture}
```

$F(x^\ast)=x^\ast$。状态不再变化——结构已经闭合。

### 第 6 类：收敛曲线图

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw[->] (0,0) -- (5,0) node[right] {$t$};
  \draw[->] (0,0) -- (0,3) node[above] {$E(x_t)$};
  \draw[thick] plot[smooth] coordinates {
    (0.2,2.7) (0.8,2.0) (1.5,1.4) (2.3,0.9) (3.2,0.55) (4.5,0.35)
  };
  \draw[dashed] (0,0.3) -- (4.8,0.3);
  \node[left] at (0,0.3) {$E^\ast$};
\end{tikzpicture}
```

$E(x_{t+1}) \le E(x_t)$。能量下降、距离收缩、KL 收敛。

### 第 7 类：相图 / 鞍点

```latex
\begin{tikzpicture}[scale=1.0, >=Stealth]
  \draw[->] (-3,0) -- (3,0) node[right] {$x$};
  \draw[->] (0,-3) -- (0,3) node[above] {$y$};
  \draw[->, thick] (-2,0.4) -- (-1,0.2);
  \draw[->, thick] (-1,0.2) -- (0,0);
  \draw[->, thick] (2,-0.4) -- (1,-0.2);
  \draw[->, thick] (1,-0.2) -- (0,0);
  \draw[->, thick] (0.2,0.5) -- (0.5,1.2);
  \draw[->, thick] (-0.2,-0.5) -- (-0.5,-1.2);
  \fill (0,0) circle (2pt) node[below right] {saddle};
\end{tikzpicture}
```

不是所有固定点都吸引人。鞍点是需要穿过的地形结构。

### 第 8 类：Cobweb 图

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw[->] (0,0) -- (4,0) node[right] {$x_t$};
  \draw[->] (0,0) -- (0,4) node[above] {$x_{t+1}$};
  \draw[dashed] (0,0) -- (3.6,3.6) node[right] {$y=x$};
  \draw[thick] plot[smooth] coordinates {
    (0.2,0.7) (1,1.8) (2,2.4) (3,2.7) (3.6,2.8)
  } node[right] {$f(x)$};
  \draw[->] (0.7,0) -- (0.7,1.55);
  \draw[->] (0.7,1.55) -- (1.55,1.55);
  \draw[->] (1.55,1.55) -- (1.55,2.15);
  \draw[->] (1.55,2.15) -- (2.15,2.15);
  \draw[->] (2.15,2.15) -- (2.15,2.45);
  \node[below] at (0.7,0) {$x_0$};
\end{tikzpicture}
```

一维迭代 $x_{t+1}=f(x_t)$ 的收敛可视化。$f(x)=x$ 是不动点。

### 第 9 类：分岔图

```latex
\begin{tikzpicture}[scale=1.1, >=Stealth]
  \draw[->] (0,0) -- (6,0) node[right] {$\eta$};
  \draw[->] (0,-2) -- (0,2.5) node[above] {behavior};
  \draw[thick] (0.5,0.2) -- (2,0.2);
  \node at (1.2,0.55) {slow};
  \draw[thick] (2.2,1.2) -- (3.6,1.2);
  \node at (2.9,1.55) {stable};
  \draw[thick] (3.8,0.8) -- (5.2,1.8);
  \draw[thick] (3.8,-0.8) -- (5.2,-1.8);
  \node at (4.8,0) {oscillation};
  \draw[dashed] (2.1,-2) -- (2.1,2.2);
  \draw[dashed] (3.7,-2) -- (3.7,2.2);
  \node[below] at (2.1,0) {$\eta_1$};
  \node[below] at (3.7,0) {$\eta_2$};
\end{tikzpicture}
```

学习率、步长、稳定性区域——参数改变导致系统行为改变。

### 第 10 类：概率单纯形 / 信念动力学图

```latex
\begin{tikzpicture}[scale=3, >=Stealth]
  \coordinate (A) at (0,0);
  \coordinate (B) at (1,0);
  \coordinate (C) at (0.5,0.866);
  \draw[thick] (A) -- (B) -- (C) -- cycle;
  \node[below left] at (A) {$p_1$};
  \node[below right] at (B) {$p_2$};
  \node[above] at (C) {$p_3$};
  \coordinate (q0) at (0.25,0.18);
  \coordinate (q1) at (0.35,0.32);
  \coordinate (q2) at (0.43,0.45);
  \coordinate (q3) at (0.50,0.56);
  \draw[->, thick] (q0) -- (q1);
  \draw[->, thick] (q1) -- (q2);
  \draw[->, thick] (q2) -- (q3);
  \fill (q0) circle (0.6pt) node[left] {$q_0$};
  \fill (q3) circle (0.6pt) node[right] {$q^\ast$};
  \node at (0.5,-0.18) {belief dynamics on the probability simplex};
\end{tikzpicture}
```

$q_{t+1} = \mathcal{T}(q_t)$，$D_{\mathrm{KL}}(q^\ast\|q_{t+1}) < D_{\mathrm{KL}}(q^\ast\|q_t)$。推理从 token 序列变成分布在信念空间里的运动。

### 三类核心图

本书最该反复出现：**向量场、能量等高线、概率单纯形**。对应三个层次——状态怎么动，为什么这样动，信念如何收敛。

### GitHub 仓库
**主仓库：** https://github.com/datawhalechina/the-terrain-of-learning

**许可证：** CC BY-NC-SA 4.0

---

## 思想脉络

### 与《推理王国》的关系
《推理王国》问：推理是什么，为什么要民主化，怎么把推理能力交给人。
《学习的地形》问：推理为什么会发生，它发生在哪个空间里，它沿什么路径运动，它为什么收敛到某个答案。

**一句话定位**：《学习的地形》是《推理王国》的几何学——不再只讨论推理能力本身，而是讨论推理能力生长的空间、地形和动力。

### 核心公式族
$$S_{t+1} = S_t + \eta F_\theta(S_t, x)$$

$$h_{l+1} = h_l + f_\theta(h_l) \quad \text{（ResNet = 显式欧拉）}$$

$$h^* = f_\theta(h^*, x) \quad \text{（DEQ = 不动点）}$$

$$D_{\mathrm{KL}}(p_{t+1}(y|x) \| p_t(y|x)) < \epsilon \quad \text{（推理收敛判据）}$$
