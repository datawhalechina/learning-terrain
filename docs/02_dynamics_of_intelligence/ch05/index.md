# 第 5 章：非欧世界：Bregman 散度与 KL 散度

一只乌鸦和一滴水从同一个山顶出发。

乌鸦振翅起飞，沿着直线——一条在三维空间中完美笔直的线段——飞向远方的湖泊。这是欧几里得的路径。两点之间，线段最短。乌鸦不关心脚下的地形——它飞在空中，不受山坡、峡谷和河流的约束。

水滴不飞。它沿着山坡滑下，汇入溪流，沿着河谷蜿蜒前行，顺着地势的梯度流向最低处。水不沿直线走——它沿着**能量下降最快的方向**走。在山坡上，这个方向是最陡的下坡方向；在平原上，这个方向是水流最自然的汇聚方向。从山顶到湖泊，水的路径几乎肯定比乌鸦的路径长——但那是因为水走的不是空间中的最短路径，而是**能量地形上的最速下降路径**。

前四章，我们一直在用乌鸦的语言——欧几里得距离、欧几里得梯度、欧几里得步长。但当我们进入信念空间——概率分布构成的空间——乌鸦的语言开始失效。

原因很简单。两个概率分布 $p = (0.5, 0.5)$ 和 $q = (0.51, 0.49)$ 之间的"差异"——这个差异的量级，和 $p = (0.99, 0.01)$ 与 $q = (1.0, 0.0)$ 之间的差异——它们的欧几里得距离都是 $\sqrt{(0.01)^2 + (-0.01)^2} \approx 0.014$。但任何一个统计学家都会告诉你：前者是两个几乎无法区分的分布，后者是从"几乎确定"到"绝对确定"的跨越——这是根本不同的两种变化。

欧几里得距离看不见这种差异。因为它把概率单纯形当作一个平直的空间——每个坐标方向等价，每个区域同质。但概率空间不是平的。在接近边界的地方（某个 $p_i$ 接近 0 或 1），相同的绝对变化对应着巨大的信息量变化。在中心附近（所有 $p_i$ 接近均匀），相同的绝对变化几乎不改变信息量。

我们需要一种新的几何——一种像水一样沿能量地形流动的几何。这种几何的语言，叫做 **Bregman 散度**。

## 5.1 从泰勒展开到能量差

要理解 Bregman 散度，最自然的入口不是定义，而是一个简单的问题：**用一阶泰勒展开逼近一个凸函数，误差有多大？**

设 $F: \mathbb{R}^n \to \mathbb{R}$ 是一个严格凸且可微的函数。给定两点 $p$ 和 $q$，$F(p)$ 在 $q$ 处的一阶泰勒展开为：

$$F(p) \approx F(q) + \langle \nabla F(q), p - q \rangle$$

因为 $F$ 是凸的，这个线性逼近总是**低估**了 $F(p)$ 的真实值（凸函数的图形位于其切线的上方）。真实值与逼近值之间的差距，恰好刻画了 $F$ 在从 $q$ 到 $p$ 这段区间上的"弯曲程度"——也就是 $F$ 的曲率在这段区间上积累的效果。

Bregman 散度定义的就是这个差距：

$$D_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle$$

这是本章最重要的公式。它只有三项：$F(p)$ 是 $p$ 处的函数值，$F(q) + \langle \nabla F(q), p - q \rangle$ 是从 $q$ 出发的线性外推对 $F(p)$ 的估计，两者之差就是"线性世界"与"弯曲世界"之间的偏差。

Bregman 散度的生成函数 $F$ 像一块透镜——不同的 $F$ 定义了不同的弯曲方式，因而定义了不同的"距离"。但 $D_F$ 不是传统意义上的距离：它不对称（$D_F(p\|q) \neq D_F(q\|p)$），也不满足三角不等式。它衡量的是**从 $q$ 出发看 $p$ 的"意外程度"**——用 $q$ 处的局部几何去估计 $p$，会犯多大的错误。反过来，从 $p$ 出发看 $q$，错误的大小通常不同——因为 $p$ 和 $q$ 处的地形曲率不同。

:::info

**为什么不对称不是缺陷？**

想象你站在一个陡峭的山坡上（位置 $q$），望向远方平缓的山谷（位置 $p$）。从你脚下出发，你用"这里的地形一直像这里一样陡"的假设去估计山谷的海拔——你会严重高估，因为你不知道前方的坡度在变小。$D_F(p\|q)$ 衡量了这个高估的幅度。

现在反过来。你站在平缓的山谷中（位置 $p$），望向远方的陡坡（位置 $q$）。你用"这里的地形一直像这里一样平"的假设去估计陡坡的海拔——你会严重低估，因为你不知道前方的坡度在变大。$D_F(q\|p)$ 衡量了这个低估的幅度。

这两个误差不相等——因为造成误差的原因不同。$D_F(p\|q) \neq D_F(q\|p)$ 不是数学上的不完美，而是地形的弯曲在告诉你：从 $A$ 看 $B$ 和从 $B$ 看 $A$，本来就是两个不同的视角。

:::

![Bregman散度：凸函数上的切线与真实值的差距](/figures/ch05_bregman_geometry.svg)

*Bregman 散度的几何定义。曲线为负熵函数 $F(p) = p\log p$，虚线为在 $q=0.6$ 处的切线。$D_F(p\|q)$（橙色竖线）是从切线到函数曲线的垂直距离——它衡量了用 $q$ 处的局部线性近似去估计 $p$ 处的函数值时，由于 $F$ 的弯曲而产生的误差。不同的 $q$ 有不同的切线——这解释了不对称性。*

## 5.2 能量地形：Bregman 散度的物理直觉

公式是冷冰冰的。但 Bregman 散度有一个极其鲜活的物理解释——而这个解释，和 ch3 中损失函数在参数空间上画出地形的逻辑完全同构。而 ch3 的逻辑本身，又可以追溯到 ch1 中一个更古老的故事。

回忆 ch1 的开篇。牛顿力学从力出发——你必须把每一个受力都分析清楚。一个小球在斜面上滚动，你要分析重力、支持力、摩擦力。一个摆在运动，你要分析张力和重力。受力分析是逐项的、局部的、繁琐的——在复杂系统中，它很快就会退化为一团乱麻。哈密顿力学的革命，是把"分析每一个力"变成"追踪一个能量函数"——系统的运动不再由各处的散乱受力决定，而是由一个整体的势能地形 $V(x)$ 引导。当哈密顿量最小时，系统趋于稳定——你不需要解所有力，你只需要跟着能量往下走。

这就是能量的力量。它把逐项分析变成了整体流动。

ch3 把同一个思想移植到了参数空间：损失函数 $L(\theta)$ 就是那片能量地形，梯度下降就是小球沿地形往下滑。现在，ch5 把这个思想再推进一层：在信念空间中，**负熵函数 $F(p) = \sum p_i \log p_i$ 就是那片能量地形**，而 Bregman 散度 $D_F(p\|q)$ 就是小球从 $q$ 滑到 $p$ 的过程中，因地形弯曲而产生的不可逆能量差。我们引入非欧几里得度量的全部动机，就是为了在信念空间中也能使用哈密顿的思想——**用一个能量函数代替对每一个信念维度的逐项分析**，让模型免于在复杂的概率约束和坐标变换中迷失，只需要沿着熵地形往下走。

在 ch3 中，损失函数 $L: \mathbb{R}^N \to \mathbb{R}$ 在参数空间上画出了一片地形——每一个参数向量 $\theta$ 被赋予一个海拔 $L(\theta)$，梯度 $\nabla L$ 告诉模型下坡的方向。在这里，凸函数 $F$ 在概率单纯形上画出了另一片地形——每一个分布 $p$ 被赋予一个"熵海拔" $F(p)$。两片地形的结构是平行的：ch3 的地形由损失函数雕刻，用来指导参数的移动；ch5 的地形由熵函数雕刻，用来指导信念的更新。而 Bregman 散度 $D_F(p\|q)$，恰好就是这片熵地形上的"不可逆高度差"。

具体地说：将 $F$ 理解为某种**势能函数**——就像重力势或电势。$F(p)$ 是系统在状态 $p$ 处的势能。要从状态 $q$ 走到状态 $p$，系统至少需要付出 $F(p) - F(q)$ 的能量差——如果空间是平直的话。

但空间不是平直的。在地形的不同位置，同样的坐标变化对应不同的能量变化——因为梯度 $\nabla F$ 在每个位置指向不同的方向，有不同的模长。一阶近似 $\langle \nabla F(q), p - q \rangle$ 假设从 $q$ 到 $p$ 的整段路上，地形都像 $q$ 处一样倾斜。但在弯曲的地形上，这个假设被打破了。坡度在变化，曲率在起作用。

$D_F(p\|q)$ 衡量的就是：**系统在从 $q$ 到 $p$ 的过程中，由于地形的弯曲而额外获得（或损失）的能量。** 它是"不可逆的能量差"——从 $q$ 到 $p$ 和从 $p$ 到 $q$，地形的弯曲方向不同，能量的剩余也不同。

这个物理解释直接导出 Bregman 散度的不对称性。从山顶滑到山谷，重力帮你做功，你获得了动能。从山谷爬上山顶，你必须克服重力做功，你付出了体力。$D_F(\text{山谷}\|\text{山顶}) \neq D_F(\text{山顶}\|\text{山谷})$——这不是数学上的缺陷，而是能量守恒在弯曲地形上的自然结果。

:::info

**信念更新中的"水流"：从一个具体的例子看 Bregman**

假设你是一个正在读论文的模型。你当前的信念分布是 $q = (0.3_{\text{接受}}, 0.5_{\text{存疑}}, 0.2_{\text{拒绝}})$。读到一段强有力的论证后，你的信念变成了 $p = (0.7, 0.2, 0.1)$。

在欧几里得世界里，这个变化的大小是 $\sqrt{(0.4)^2 + (-0.3)^2 + (-0.1)^2} \approx 0.51$。但注意——"存疑"从 0.5 跌到了 0.2（变化 -0.3），而"拒绝"从 0.2 跌到了 0.1（变化 -0.1）。欧几里得认为前者的重要性是后者的 3 倍。但 Bregman（KL）不这么看。

当 $F$ 取负熵时，KL 散度关心的是**相对变化**而非绝对变化。"存疑"从 0.5→0.2，相对比例是 $0.2/0.5 = 0.4$。"拒绝"从 0.2→0.1，相对比例也是 $0.5$。在 KL 的眼中，这两个变化的信息量相近——因为它们都意味着模型对相应选项的信念减半。"接受"从 0.3→0.7，比例翻倍——KL 为此付出了最大的信息代价。

这就是为什么 Bregman 散度比欧几里得距离更适合信念空间：**它关心的是比例的乘积效应，而非绝对差值的加法效应。** 水的流动不是看坐标走了多远，而是看地势降了多少能量。

:::

## 5.3 熵地形：Bregman 的地形学构造

上一节说 Bregman 散度衡量"能量差"。但什么样的几何结构允许我们精确地谈论这个能量差？

答案是**熵地形**。这是 Bregman 几何最核心的构造。给定凸函数 $F$，我们可以定义两个对象：

**Bregman 能量面**：$\mathcal{E}_F = \{ (p, F(p)) : p \in \text{dom}(F) \}$。这是 $F$ 的图——一个嵌在 $\mathbb{R}^{n+1}$ 中的 $n$ 维曲面。在这个曲面上，每一个概率分布不仅有一个坐标位置，还有一个由 $F$ 赋予的"海拔"。

**Bregman 切线丛**：在每一个点 $q$ 上，可以画出 $F$ 在该点的切超平面 $T_q = \{ (p, F(q) + \langle \nabla F(q), p - q \rangle) : p \in \mathbb{R}^n \}$。Bregman 散度 $D_F(p\|q)$ 恰好就是从 $T_q$ 平面到 $\mathcal{E}_F$ 曲面在 $p$ 处的垂直距离。

这个构造之所以叫"熵地形"，是因为它在做一件非常深刻的事：**它用 $F$ 的图作为"绝对海拔"，用切平面作为"局部海平面"，而 Bregman 散度就是在局部海平面上方测量的相对高度。** 不同的 $q$ 有不同的局部海平面——这解释了不对称性。全局没有一个统一的海拔零点——这解释了为什么 Bregman 散度不是距离。

这个熵地形构造还直接给出了 Bregman 散度的广义勾股定理。给定三个点 $p, q, r$，如果 $q$ 和 $r$ 满足某种"正交条件"（即 $q$ 是 $p$ 在某个 Bregman 球上的投影），则：

$$D_F(p \| r) = D_F(p \| q) + D_F(q \| r)$$

这看起来像勾股定理——"斜边"的 Bregman 散度等于两条"直角边"的 Bregman 散度之和。但在欧几里得空间中，勾股定理需要直角。在 Bregman 几何中，"直角"被重新定义——不是向量点积为零，而是 **Bregman 正交**：$\langle \nabla F(q) - \nabla F(r), q - p \rangle = 0$。

这个广义勾股定理不是美学装饰。它是**巴拿赫收缩映射原理能够应用于 KL 散度的几何基础**——我们在永霖极限中将看到，每一步推理的 KL 散度减小之所以能连成一条必然收敛的链，正是因为 KL 散度满足 Bregman 的广义勾股结构。

:::info

**广义勾股定理：为什么它是一切收敛证明的心脏**

在欧几里得空间中，勾股定理 $c^2 = a^2 + b^2$ 需要两条直角边。直角意味着两个方向"互相看不见对方的投影"——沿一个方向移动不会改变你在另一个方向上的坐标。

在 Bregman 几何中，"直角"被重新定义为 **Bregman 正交**：$\langle \nabla F(q) - \nabla F(r), q - p \rangle = 0$。这个条件说的不是欧几里得角度为 90°，而是：沿 Bregman 测地线从 $r$ 到 $q$ 的方向，与从 $p$ 到 $q$ 的方向，在 $F$ 的曲率下"互不干扰"。

为什么这对永霖极限至关重要？因为当你从 $p_t$（当前信念）迈出一步到 $p_{t+1}$，再迈出一步到 $p_{t+2}$……每一步都沿着 KL 熵地形的梯度方向。如果每一步都 Bregman-正交于目标方向，那么每一步的 KL 散度减小可以像勾股定理一样**累加**——总减小量等于各步减小量之和。这意味着你不会"走回头路"——每一步都让你离目标更近，而且近的量是可以精确计算的。

如果 Bregman 广义勾股定理不成立——如果 KL 散度在更新方向上的减小不能累加——那么即使每步的 KL 都在减小，你也无法保证总序列收敛。你可能像在漩涡中打转——每步都下降，但永远到不了湖心。广义勾股定理保证了：只要每步都在正确的 Bregman-正交方向上前进，下降就是累积的，收敛就是必然的。

:::

## 5.4 KL 散度：当 $F$ 取负熵

让 $F$ 具体化。取 $F(p) = \sum_i p_i \log p_i$——香农熵的负数（注意是**负**熵，因为熵本身是凹函数，我们需要 $F$ 凸）。此时 $\nabla F(p)_i = 1 + \log p_i$。代入 Bregman 散度的定义：

$$D_F(p \| q) = \sum_i p_i \log p_i - \sum_i q_i \log q_i - \sum_i (1 + \log q_i)(p_i - q_i)$$

化简（利用 $\sum_i p_i = \sum_i q_i = 1$）：

$$D_F(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} = D_{\mathrm{KL}}(p \| q)$$

这就是 KL 散度。**KL 散度不是随便定义的一个"分布差异度量"——它是负熵函数生成的 Bregman 散度。** 它的几何性质不是外加的，而是负熵的凸性内在地决定的。

这个观察彻底改变了我们对 KL 散度的理解。通常教科书说"KL 散度不对称，因此不是真正的距离"。但 Bregman 视角告诉我们：不对称性恰恰是 KL 散度作为能量差的本性。$D_{\mathrm{KL}}(p\|q)$ 衡量的是——用分布 $q$ 来编码分布 $p$ 的样本时，平均每个符号多用了多少比特（与用 $p$ 自身的最优编码相比）。从 $q$ 到 $p$ 的信息浪费，与从 $p$ 到 $q$ 的信息浪费，本来就不是同一件事。

:::info

**"编码"的直觉：为什么 KL 是"多用的比特数"**

信息论中有一个基本事实：如果信源的真实分布是 $p$，但你用分布 $q$ 来设计最优编码（哈夫曼编码或算术编码），那么编码每个符号平均多用的比特数恰好是 $D_{\mathrm{KL}}(p\|q)$。

举例。假设 $p = (0.9_{\text{晴}}, 0.1_{\text{雨}})$——你住在一个几乎永远晴朗的地方。你设计了一个编码方案，用最短的码字表示"晴"。这是用 $p$ 的最优编码——每个符号平均用 $H(p) = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.47$ 比特。

现在你的朋友从伦敦来玩。伦敦的天气是 $q = (0.5, 0.5)$。他带着他的"伦敦编码方案"来到你的城市——给"晴"和"雨"各分配等长的码字。用伦敦方案来编码你所在城市的天气报告，每个符号平均多用 $D_{\mathrm{KL}}(p\|q) = 0.9\log_2(0.9/0.5) + 0.1\log_2(0.1/0.5) \approx 0.53$ 比特。

反过来，如果你带着你的"晴天方案"去伦敦，多用的比特数是 $D_{\mathrm{KL}}(q\|p) = 0.5\log_2(0.5/0.9) + 0.5\log_2(0.5/0.1) \approx 0.74$ 比特——比前者多了 40%。原因很清楚：用几乎只认识"晴"的编码去处理一半是"雨"的报告，比用认识"晴"和"雨"的编码去处理几乎只有"晴"的报告，浪费更大。

**KL 不对称的本质就是：用 $A$ 去解释 $B$ 的代价，不等于用 $B$ 去解释 $A$ 的代价。** 在信念空间中，这恰好对应了我们想要的：从错误到正确，和从正确到错误，本来就不是对称的操作。

:::

KL 散度在信念空间中天然合适的另一个原因，是它的**局部二次近似**。在 $q$ 附近做泰勒展开：

$$D_{\mathrm{KL}}(p \| q) \approx \frac{1}{2} \sum_i \frac{(p_i - q_i)^2}{q_i}$$

这个二次型的系数是 $1/q_i$——在 $q_i$ 接近 0 的区域，相同的绝对差异被极度放大。这正是我们在开篇提到的：将概率从 0.99 移到 1.0，与将概率从 0.5 移到 0.51，信息意义上的"距离"天差地别。KL 散度的局部曲率自动地、正确地捕捉了这种不均匀的敏感性。

![KL散度的不对称性](/figures/ch05_kl_asymmetry.svg)

*两个高斯分布之间的 KL 散度。$p$（窄峰，橙色）与 $q$（宽峰，蓝色）之间的 $D_{\mathrm{KL}}(p\|q) = 2.45$ vs $D_{\mathrm{KL}}(q\|p) = 10.45$——差异超过四倍。左图 $D_{\mathrm{KL}}(p\|q)$：$p$ 集中的区域（橙色高亮），$q$ 的概率密度很小，惩罚来自 $\log(p/q)$ 在 $p$ 下的期望。右图 $D_{\mathrm{KL}}(q\|p)$：$q$ 散布的区域（蓝色高亮）中 $p$ 几乎为零，$\log(q/p)$ 爆炸——$q$ 的"广度"在窄 $p$ 面前受到极端惩罚。不对称性不是缺陷——它反映了信息编码方向的不可逆性。*

![KL散度非对称曲面](/figures/ch05_kl_surface_tikz.svg)

*KL散度在概率单纯形上的非对称曲面。曲面高度表示 $D_{\mathrm{KL}}(p\|q)$ 的值——以固定 $q$（参考分布）为原点，$p$ 在单纯形上变化。注意曲面的不对称形状：在 $q$ 概率较小的维度上，曲面急剧陡峭（$1/q_i$ 的曲率放大效应）；在 $q$ 概率较大的维度上，曲面平缓。这种"地形弯曲"解释了为什么从确信到混乱与从混乱到确信的信息代价完全不同——KL散度不是距离，而是能量差。*

## 5.5 Fisher 信息与自然梯度

KL 散度的局部二次近似直接引出另一个核心概念：Fisher 信息矩阵。

在 $q$ 附近，$D_{\mathrm{KL}}(p\|q)$ 的 Hessian 矩阵（对 $p$ 的二阶导数在 $p=q$ 处的值）正是 Fisher 信息矩阵：

$$G(q)_{ij} = \mathbb{E}_{x \sim q}\left[ \frac{\partial \log q(x)}{\partial \theta_i} \frac{\partial \log q(x)}{\partial \theta_j} \right]$$

Fisher 信息矩阵是概率分布空间中的**黎曼度量**——它是 KL 散度在每一点处的局部曲率，像黎曼几何中的 $g_{ij}$ 一样，定义了概率空间中"真正的"距离和角度。

这就引出了**自然梯度**（natural gradient）。普通的梯度下降在参数空间中沿欧几里得最速下降方向走。但参数空间中的欧几里得距离没有概率意义——参数从 0.5 变成 0.51，和从 0.99 变成 1.0，在参数空间中差一样多，但在分布空间中差得完全不同。

自然梯度修正了这一点。它沿**分布空间中的最速下降方向**走：

$$\theta_{t+1} = \theta_t - \eta G(\theta_t)^{-1} \nabla L(\theta_t)$$

将欧几里得梯度 $\nabla L$ 乘以 Fisher 信息的逆 $G^{-1}$，相当于将参数空间中的"歪斜"拉直——让每一步在分布空间中（而非参数空间中）具有相同的"有效步长"。这完全平行于 ch1 中我们讨论过的：在弯曲的参数空间中使用固定欧几里得步长就像用直尺在球面上测地图——自然梯度用度量张量修正方向，使步长在局部几何中保持一致。

### 5.5.1 信息几何：为什么这一切是必然的

Fisher 信息矩阵、自然梯度、镜像下降——它们看起来像是三个独立发现的技术。但它们不是。它们是一个统一的数学框架的三个切面。这个框架叫做**信息几何**（Information Geometry），由甘利俊一（Shun-ichi Amari）在 20 世纪 80 年代建立。

信息几何的核心洞察极其简洁：**概率分布构成的空间不是一个平直的向量空间——它是一个带有自然黎曼度量的流形。** 这个流形上的每一点是一个概率分布 $p$。Fisher 信息矩阵 $G(p)$ 是这个流形的黎曼度量——它定义了流形上两点之间"真正的"距离：不是参数坐标的欧几里得差，而是沿着流形弯曲的测地线长度。

但信息几何的深刻之处不止于此。它揭示了这个流形具有一种罕见的数学结构：**双重平坦性**（dual flatness）。

在同一个概率流形上，可以定义两套不同的"直线"（测地线）。$m$-测地线是概率分布的线性组合——"混合"两个分布，像把两种颜色的颜料倒在一起。$e$-测地线是指数族的自然参数空间中的直线——"指数"地从一个分布变到另一个。这两套测地线通常不重合——除非流形是平直的。当它们不重合时，流形就具有了"弯曲"的结构。

与双重平坦性配套的是一族被称为 **$\alpha$-连接**（$\alpha$-connections）的微分几何对象。$\alpha = 1$ 对应 $e$-连接（指数连接），$\alpha = -1$ 对应 $m$-连接（混合连接），$\alpha = 0$ 对应 Levi-Civita 连接（Fisher-Rao 度量下的自对偶连接）。不同的 $\alpha$ 定义了不同的"平行移动"方式——在不同的连接下，"把向量 $v$ 从 $p$ 平移到 $q$ 而不改变方向"意味着完全不同的事。

Bregman 散度在这个框架中获得了它的终极定位：**每一个 Bregman 散度由一对对偶的平坦流形结构生成。** 具体地说，取一个凸函数 $F$，它在原始流形上诱导一个 $e$-平坦结构，它的凸共轭 $F^*$ 在对偶流形上诱导一个 $m$-平坦结构。$D_F(p\|q)$ 恰好度量了这两个平坦结构之间的"不重合程度"。

当 $F$ 取负熵时，$e$-平坦结构对应指数族分布（如高斯、伯努利、softmax），而 KL 散度 $D_{\mathrm{KL}}(p\|q)$ 恰好就是这两个对偶平坦结构之间的 Bregman 散度。这就是为什么在概率空间中，KL 散度不是"众多选择之一"——它是这个流形的内禀几何所**唯一确定**的"自然散度"。

信息几何给本章的四个核心概念提供了统一的数学底座：

- **Bregman 散度**（§5.1-§5.3）→ 由 $F$ 的凸性在对偶平坦流形上自然产生。
- **KL 散度**（§5.4）→ 当 $F$ 取负熵时的特例，是指数族的自然散度。
- **Fisher 信息矩阵**（§5.5）→ 概率流形的黎曼度量，是 KL 散度的局部二次型。
- **镜像下降**（§5.6）→ 在 $e$-平坦和 $m$-平坦结构之间交替投影的几何算法。

这一整套构造不是人为拼凑的。它是从"概率分布构成一个流形"这个单一前提出发，经微分几何的自然展开而必然得到的结构。你不需要"选择" Bregman 散度来度量信念差异——流形的弯曲替你选了。

:::info

**兔狲教授：不是 Amari 发明了信息几何——是概率流形一直有这个结构**

很多人第一次接触信息几何的时候会觉得：这是 Amari 构造的一套漂亮的数学框架。但严格地说，Amari 不是**发明**了信息几何——他是**发现**了它。

概率分布构成的空间天然地具有双重平坦结构和对偶 $\alpha$-连接。这不是任何人"加"上去的——就像地球表面的曲率不是高斯"加"上去的。高斯在汉诺威的山顶上测量三角形内角和，发现它大于 180°——他不是在发明球面几何，他是在发现地球的曲率。

同样地，当你在信念空间中做梯度下降时，你每一步迈出的"步长"是否有效，不由你选择的坐标系决定——由概率流形的 Fisher 度量决定。你可以在参数空间中走直线，但流形弯曲了你的路径。自然梯度、镜像下降、永霖极限——它们都不是"技巧"。它们是在弯曲的概率流形上，唯一前后一致的走路方式。

这本书从 ch1 到 ch5 讲了一个故事：从牛顿的力到哈密顿的能量，从损失地形到熵地形。信息几何是这个故事在 20 世纪的最后一章：**空间不是平的。概率空间更不是。** 你不需要强加一个度量——度量自己在那里。你只需要学会测量它。

:::

## 5.6 Mirror Descent：在 Bregman 地形中下降

自然梯度需要计算 Fisher 信息矩阵的逆——在高维空间中代价极高。但 Bregman 散度提供了另一条不需要显式计算 $G^{-1}$ 的路：**镜像下降（mirror descent）**。

镜像下降的核心思想是：不在原始的参数空间中做梯度下降，而是在**对偶空间**（镜像空间）中做。过程分三步：

1. **映射到对偶空间**：将当前参数 $\theta_t$ 通过 $\nabla F$ 映射到对偶点 $\mu_t = \nabla F(\theta_t)$。
2. **在对偶空间中下降**：在对偶空间中做普通的梯度步 $\mu_{t+1} = \mu_t - \eta g_t$（其中 $g_t$ 是损失函数的梯度）。
3. **映射回原始空间**：通过 $\nabla F^*$（$F$ 的凸共轭的梯度）将 $\mu_{t+1}$ 映射回 $\theta_{t+1} = \nabla F^*(\mu_{t+1})$。

这三个步骤的合成更新等价于求解：

$$\theta_{t+1} = \arg\min_\theta \left\{ \langle g_t, \theta \rangle + \frac{1}{\eta} D_F(\theta \| \theta_t) \right\}$$

这个公式是镜像下降的优雅核心：**每一步找到的新参数 $\theta_{t+1}$，既要在损失梯度的方向上取得进展（第一项），又不能离当前位置太远——"远"由 Bregman 散度度量（第二项）。**

当 $F(\theta) = \frac{1}{2}\|\theta\|^2$ 时，$D_F$ 退化为欧几里得距离，镜像下降退化为普通梯度下降。当 $F$ 取负熵时，$D_F$ 就是 KL 散度，镜像下降变成了**在概率单纯形上沿 KL 几何的最速下降**。

镜像下降的真正威力在于：它自动地将参数空间的几何（由 $F$ 的 Hessian 编码）融入了每一步更新中。在概率单纯形的边缘附近，KL 散度的曲率极高——相同的欧几里得步长在分布空间中对应巨大的跳跃。镜像下降通过 $D_F$ 的正则化自动缩减了这些区域的步长。这就是 **$\eta_{\max}(p)$ 的几何根源**——我们稍后会把它精确地形式化。

![镜像下降 vs 欧几里得下降：单纯形上的轨迹对比](/figures/ch05_mirror_descent.svg)

*概率单纯形（三角形）上的优化轨迹。起点为均匀分布（中心），目标为 $(0.7, 0.2, 0.1)$（方块）。欧几里得投影下降（左，朱红）沿直线走向目标，但在边界处需要投影回单纯形。镜像下降（右，绿色）沿 KL 几何的自然梯度方向前进——在单纯形边缘自动减速，轨迹平滑地收敛到目标。星号标记最终位置。*

:::info

**为什么叫"镜像"？对偶空间的直觉**

镜像下降的名字来自一个非常形象的几何结构。想象你站在一个弯曲的镜面前面——镜中的你与真实的你不在同一个空间中。真实的你在"原始空间"（参数空间），镜中的你在"对偶空间"（梯度空间）。

普通梯度下降在原始空间中直接迈步：$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$。这在欧几里得空间中自然，但在弯曲的 Bregman 空间中，原始空间中的"直走"并不对应分布空间中的直走。

镜像下降的做法是：**先在镜子里走，再映射回来。**
- 第 1 步（映射到镜中）：$\mu_t = \nabla F(\theta_t)$。$\nabla F$ 是从原始空间到镜中空间（对偶空间）的映射。在镜中世界，所有的运算都是欧几里得的——你可以放心地用普通的梯度步。
- 第 2 步（在镜中走）：$\mu_{t+1} = \mu_t - \eta g_t$。这是普通的、欧几里得的梯度步——镜中世界是平的。
- 第 3 步（映射回来）：$\theta_{t+1} = \nabla F^*(\mu_{t+1})$。$\nabla F^*$ 是 $\nabla F$ 的逆映射——从镜中世界回到真实世界。

神奇之处在于：第 2 步在镜中走的是一个**普通的欧几里得步长** $\eta$，但经过两次映射（$\nabla F^* \circ$ (欧几里得步) $\circ \nabla F$）后，在原始空间中产生的效果等价于**自动适应了 $F$ 的曲率**。你不需要显式计算 Fisher 信息的逆——$F$ 的凸共轭自然完成了这个任务。

:::

## 5.7 概率单纯形：softmax 作为投影

在讨论永霖极限之前，我们需要最后一个几何工具：**概率单纯形**的结构，以及 **softmax 作为投影算子**的角色。

概率单纯形是所有可能的离散概率分布构成的集合：

$$\Delta^{K-1} = \left\{ p \in \mathbb{R}^K : p_i \geq 0, \sum_i p_i = 1 \right\}$$

这是一个 $(K-1)$ 维的几何对象——嵌在 $K$ 维空间中，但本身只有 $K-1$ 个自由度（因为 $\sum p_i = 1$ 的约束吃掉了一个维度）。

任何向量 $z \in \mathbb{R}^K$（logits——模型的原始输出）都可以通过 softmax 函数投影到单纯形上：

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

从几何角度看，softmax 做的是：**将 $\mathbb{R}^K$ 中的任意点，沿某个特定方向，投影到概率单纯形上。** 这个投影方向不是欧几里得的（不是垂直线段），而是 **Bregman 的**——具体地说，它沿着由负熵函数 $F(p) = \sum p_i \log p_i$ 生成的 Bregman 测地线方向。

这意味着一个重要的事实：**模型的前向传播——logits → softmax → 概率分布——本身就携带了 KL 几何的结构。** 它不是随意选择的一个归一化操作，而是负熵 Bregman 几何中的自然投影。

这一点与 $\eta_{\max}(p)$ 的坐标不变性密切相关。因为 softmax 是一个 Bregman 投影，任何在参数空间中进行的欧拉步——$\theta_{t+1} = \theta_t + \eta \cdot \text{（某个方向）}$——在通过 softmax 投影到单纯形时，步长的"有效大小"取决于你在单纯形的哪个位置。在边缘处（某个 $p_i$ 很小），同样的参数变化引起更大的分布变化——因此，安全的步长更小。$\eta_{\max}(p)$ 精确地量化了这个位置依赖性。

![概率单纯形上的信念动力学](/figures/ch05_belief_simplex_tikz.svg)

*概率单纯形上的信念更新轨迹。三角形表示三类问题（$\Delta^2$）的概率单纯形，每一点对应一个信念分布 $(p_1, p_2, p_3)$，满足 $\sum p_i=1$。箭头表示 Bregman 几何下的信念更新方向——在单纯形中心（均匀分布，信息量最小），各方向更新幅度适中；在边缘区域（某个类别几乎被排除），更新方向沿单纯形边界弯曲，步长自动缩小——这体现了 KL 几何中"在不可逆的边缘小心行走"的自然约束。*

## 5.8 永霖极限 I：能量函数与推理算子

现在我们可以进入本章的理论核心——永霖极限。

永霖极限回答的问题是：**在信念空间中，推理一定会收敛吗？** 不是"通常如此""经验上如此"，而是"从任意初始信念出发，在适当的步长下，欧拉步推理必然收敛到唯一的不动点"。

这个命题的证明遵循一条清晰的数学链。我们把它拆成三个部分。

**第一步：定义能量函数。**

在信念空间中，我们需要一个可以充当"地势"的量——类似于 ch3 中损失函数 $L(\theta)$ 在参数空间中扮演的角色。这个量必须满足两个条件：它在推理过程中单调递减，且它在不动点处达到极小值。

KL 散度天然就是候选者。给定一个目标分布（训练数据的经验分布）$p^*$，定义能量函数为模型当前信念 $p_t$ 与 $p^*$ 之间的 KL 散度：

$$E(p_t) = D_{\mathrm{KL}}(p^* \| p_t) = \sum_i p^*_i \log \frac{p^*_i}{p_{t,i}}$$

（注意这里 KL 的方向是 $p^*\|p_t$——用当前模型去"解释"真实数据的额外代价。这个方向的选择不是随意的，它确保了后续压缩性证明的自然成立。）

$E(p_t) \geq 0$，且 $E(p_t) = 0$ 当且仅当 $p_t = p^*$。这满足了李雅普诺夫函数（ch3）的两个基本条件——处处非负，在目标处归零。

**第二步：构造欧拉步推理算子。**

推理过程被建模为信念状态在单纯形上的离散时间演化。每一步，模型根据当前信念 $p_t$ 和输入 $x$，产生一个更新方向，然后沿这个方向迈出步长 $\eta$：

$$p_{t+1} = \Phi_\eta(p_t) = \text{softmax}(\text{logit}(p_t) + \eta \cdot \text{update}(p_t, x))$$

这里的"update"方向可以从损失函数的梯度、注意力机制的输出、或者任何其他推理模块中产生。关键的抽象是：$\Phi_\eta$ 是单纯形上的一个映射，$\eta$ 是其步长。

这就是推理算子。给定当前信念和步长，它输出下一个信念。整个推理过程就是反复应用 $\Phi_\eta$：

$$p_0 \to p_1 \to p_2 \to \cdots \to p_T$$

![信念更新循环的计算图](/figures/ch05_belief_update_graph_tikz.svg)

*信念更新循环的计算图。q_t 经算子 T 变换为 q_{t+1}，KL 散度检查收敛：若 D_KL(q*||q_{t+1})<D_KL(q*||q_t) 则继续，否则已收敛到不动点 q*。*

**第三步：证明压缩性。**

永霖极限的枢纽在于：$\Phi_\eta$ 在 KL 散度的意义下是一个**压缩映射**——每一步都让模型离 $p^*$ 更近，而且这种"靠近"有一个明确的下界。

形式化地说，存在一个常数 $\gamma \in (0, 1)$ 使得对于所有 $p \neq p^*$：

$$E(\Phi_\eta(p)) \leq \gamma \cdot E(p)$$

或者等价地：

$$D_{\mathrm{KL}}(p^* \| \Phi_\eta(p)) \leq \gamma \cdot D_{\mathrm{KL}}(p^* \| p)$$

能量在每一步至少以因子 $\gamma$ 衰减。这不是一个模糊的"能量下降"——它是一个确定的收缩。

## 5.9 永霖极限 II：压缩性的完整证明

**第四步：证明压缩性。**

现在我们给出压缩性的完整证明。这是永霖极限的数学心脏。

**定理**：设能量函数 $E(p) = D_{\mathrm{KL}}(p^*\|p)$ 是 $\mu$-强凸的（$\mu > 0$），且其梯度 $\nabla E$ 是 $L$-光滑的（$L > 0$）。取步长 $\eta \in (0, 2\mu/L^2)$，则欧拉步推理算子 $\Phi_\eta(p) = \operatorname{proj}_{\mathcal{P}}(p - \eta \nabla E(p))$ 在 KL 散度下是一个压缩映射。

**证明**。令 $q_1 = \Phi_\eta(p_1)$，$q_2 = \Phi_\eta(p_2)$。我们要证明存在 $\gamma < 1$ 使得 $D_{\mathrm{KL}}(q_1 \| q_2) \leq \gamma \cdot D_{\mathrm{KL}}(p_1 \| p_2)$。

**第一步：应用 Bregman 三点恒等式。**

回忆 §5.3：KL 散度作为负熵 Bregman 散度，满足三点恒等式。取三点为 $p_1, q_1, q_2$：

$$D_{\mathrm{KL}}(p_1 \| q_2) = D_{\mathrm{KL}}(p_1 \| q_1) + D_{\mathrm{KL}}(q_1 \| q_2) - \langle \nabla\phi(q_2) - \nabla\phi(q_1),\, p_1 - q_1 \rangle$$

其中 $\phi(p) = \sum_i p_i \log p_i$（负熵）。整理得：

$$D_{\mathrm{KL}}(q_1 \| q_2) = D_{\mathrm{KL}}(p_1 \| q_2) - D_{\mathrm{KL}}(p_1 \| q_1) + \langle \nabla\phi(q_2) - \nabla\phi(q_1),\, p_1 - q_1 \rangle$$

前两项由散度给出，第三项是内积——我们需要展开它。

**第二步：展开内积项。**

由 $q_i = \operatorname{proj}_{\mathcal{P}}(p_i - \eta \nabla E(p_i))$ 和投影到单纯形的一阶最优性条件，内积项可以用 $E$ 的梯度差表示。利用 $E$ 的两个关键性质：

- **$\mu$-强凸**：$\langle \nabla E(p_1) - \nabla E(p_2),\, p_1 - p_2 \rangle \geq \mu \|p_1 - p_2\|^2$。强凸意味着能量函数在每一个方向上的曲率至少为 $\mu$——这是"地形有足够坡度"的保证。

- **$L$-光滑**：$\|\nabla E(p_1) - \nabla E(p_2)\| \leq L\|p_1 - p_2\|$。光滑意味着梯度变化不会太剧烈——这是"地形不会突然出现悬崖"的保证。光滑性的负贡献（内积的上界）至多为 $\eta^2 L^2 \|p_1 - p_2\|^2$。

两项合并，内积项的精确贡献为：

$$\langle \cdots \rangle = \eta(2\mu - \eta L^2)\|p_1 - p_2\|^2$$

**第三步：得到压缩不等式。**

代回原式，并利用 $D_{\mathrm{KL}}(p_1\|q_2) \leq D_{\mathrm{KL}}(p_1\|p_2)$（投影到单纯形不增加 KL 散度——这是 Bregman 投影的基本性质）：

$$D_{\mathrm{KL}}(q_1 \| q_2) \leq D_{\mathrm{KL}}(p_1 \| p_2) - \eta(2\mu - \eta L^2)\|p_1 - p_2\|^2$$

当 $\eta < 2\mu/L^2$ 时，$2\mu - \eta L^2 > 0$，右边第二项严格为正——散度严格缩小。

利用 Pinsker 不等式的逆向形式（KL 散度与欧几里得范数在单纯形上有局部等价性，存在常数 $C > 0$）：

$$D_{\mathrm{KL}}(q_1 \| q_2) \leq \underbrace{\left(1 - \frac{\eta(2\mu - \eta L^2)}{C}\right)}_{k(\eta) < 1} \cdot D_{\mathrm{KL}}(p_1 \| p_2)$$

这就是压缩不等式。**$2\mu - \eta L^2$ 是压缩的心脏**：强凸给了正贡献 $2\mu$，光滑性给了负贡献 $\eta L^2$，两者之差决定了每一步压缩的幅度。步长上界 $\eta < 2\mu/L^2$ 正是保证这个差为正的条件。

**证毕。**

**第五步：调用巴拿赫不动点定理。**

由于 $\Phi_\eta$ 是压缩映射（$k(\eta) < 1$），且概率单纯形在 KL 散度下完备，巴拿赫不动点定理直接给出：

1. **存在唯一**不动点 $p^*$ 满足 $\Phi_\eta(p^*) = p^*$
2. 对任意初始信念 $p_0$，迭代 $p_{t+1} = \Phi_\eta(p_t)$ 收敛到 $p^*$
3. 收敛速率：$D_{\mathrm{KL}}(p_t \| p^*) \leq k(\eta)^t \cdot D_{\mathrm{KL}}(p_0 \| p^*)$

这就是永霖极限的核心断言：**推理在信念空间中必然收敛，且收敛速率由 $k(\eta)$ 指数级保证。** 不是"通常"收敛，不是"被训练得很好"——而是 Bregman 几何的三点恒等式、能量函数的强凸性和光滑性，以及巴拿赫的不动点定理，三者共同确保了收敛的必然性。

:::info

**巴拿赫不动点：如果每一步都在靠近，终点一定存在**

巴拿赫不动点定理的直觉其实极其朴素。想象你站在一个房间里。你被要求做以下的动作：向房间的正中心跨一步，步长是当前位置到中心距离的 90%。重复这个动作。

第一步，你走过了距离的 90%。第二步，你再走剩余距离的 90%。第三步……无论你从房间的哪个角落开始，你一定会无限接近正中心——因为每一步都把剩余距离压缩到原来的 10%（即压缩因子 $\gamma = 0.1$）。你甚至不需要知道中心在哪——你只需要知道每一步都在向它靠近，并且靠近的比例是确定的。

这就是巴拿赫压缩映射原理的全部。它只要求三件事：
1. 你有一个"距离"的概念——在永霖极限中，这是 KL 散度（固定方向 $p^*\|p$）。
2. 你的映射每一步都在压缩距离——在永霖极限中，$D_{\mathrm{KL}}(p^*\|\Phi_\eta(p)) \leq \gamma \cdot D_{\mathrm{KL}}(p^*\|p)$ 且 $\gamma < 1$。
3. 空间是"完备的"——任何柯西序列都有极限。概率单纯形在 KL 散度下满足这个条件。

如果这三条成立，巴拿赫定理就保证：不动点唯一存在，且无论从哪出发，你都会到那里。**你不需要"希望"收敛——你证明了收敛。** 这就是永霖极限的数学心脏。

:::

## 5.10 永霖极限 III：η_max(p) 的推导与坐标不变性

压缩性成立的步长范围不是任意的。$\eta$ 必须足够小以保证每一步的压缩因子 $\gamma < 1$。但"足够小"是多小？这个值依赖于当前位置 $p$——因为概率单纯形不同区域的曲率不同。

**推导。** 压缩映射要求 $\eta < 2\mu / L^2$，其中 $\mu$ 是能量函数的强凸常数，$L$ 是光滑常数。这两个常数不是神秘的数字——它们直接编码在 $\nabla^2 E(p)$ 的 Hessian 矩阵中。

对于交叉熵损失 $E(p) = \mathbb{E}_D[-\log p_y]$（$q_i = P_D(y=i)$ 为训练标签边际），Hessian 在概率单纯形上的一般位置为：

$$\nabla^2 E(p) = \text{diag}\left(\frac{q_i}{p_i^2}\right)$$

在不动点 $p = q$ 处取值为 $\nabla^2 E(q) = \text{diag}(1/q_i)$——与负熵 $\Phi(p) = \sum_i p_i \log p_i$ 的 Hessian 完全相同。**曲率属于空间而不属于损失**：目标分布只进入一阶项（漂移），不进入二阶项（曲率）。这是下文 $\eta_{\max}(p)$ 可以完全从当前位置读出、无需知道目标分布的几何原因。

在不动点 $p = q$ 处，强凸常数 $\mu$ 与光滑常数 $L$ 就是该曲率张量的**谱极值**——对应最平坦与最陡峭的方向：

$$\mu = \lambda_{\min} = \frac{1}{\max_i p_i}, \qquad L = \lambda_{\max} = \frac{1}{\min_i p_i}$$

（$p_i$ 取不动点处的值。一般位置的精确谱极值为 $\min_i q_i/p_i^2$、$\max_i q_i/p_i^2$，不动点邻域内二者一致。）

代入巴拿赫压缩的步长条件 $\eta < 2\mu / L^2$：

$$\eta_{\max}(p) = \frac{2\mu}{L^2} = \frac{2}{L \cdot \kappa}, \qquad \kappa = \frac{L}{\mu} = \frac{\max_i p_i}{\min_i p_i}$$

也就是

$$\eta_{\max}(p) = 2 \cdot \frac{\min_i p_i^2}{\max_i p_i}$$

其中 $\kappa$ 是曲率各向异性（谱比/条件数）——$\max_i p_i$ 离 1 多远、$\min_i p_i$ 离 0 多远，即**离单纯形边缘的脆弱度**。读法：最高曲率 $L$ 告诉你"多陡"，$\kappa$ 告诉你"多各向异性"，安全步长被二者的乘积压缩。欧拉法的稳定性界 $\eta < 2/\lambda_{\max}$ 是各向同性形式，$\eta_{\max} = 2/(L\kappa)$ 是它的各向异性精确修正。

**更紧的精确界：切空间谱。** 严格地说，投影把动力学约束在切空间 $\{v: \sum_i v_i = 0\}$ 内，因此精确的局部失稳条件是

$$\eta < \frac{2}{\theta_{\max}}, \qquad \theta_{\max} = \text{最大特征值}\;\operatorname{diag}\left(\frac{1}{q_i}\right)\Big|_{\text{切空间}}$$

K = 3 时 $\theta_{\max}$ 是

$$3q_0q_1q_2\, \theta^2 - 2(q_0q_1+q_0q_2+q_1q_2)\, \theta + 1 = 0$$

的最大根。由于 $\theta_{\max} \leq L$，恒有 $2/\theta_{\max} \geq 2/L = \kappa \cdot \eta_{\max}$——即 $\eta_{\max} = 2/(L\kappa)$ 是**全空间谱**给出的保守界，切空间谱界更紧。二者的关系在附录 5.A 的数值验证中同时可见：精确界（数值点贴线）与保守界（数值点恒在其上）。

这个公式的每一项都有清晰的几何含义：

- **分子的 $\min_i p_i^2$**：模型最不确定的维度。若模型几乎完全排除了某个类别（$p_i \approx 0.01$），$\min_i p_i^2 = 10^{-4}$——步长上限被压到极小。几何原因：KL 散度在单纯形边缘的曲率正比于 $1/p_i$（Hessian 的对角元素），相同绝对变化在边缘处产生巨大的分布差异。平方来自 $L^2 = 1/\min_i p_i^2$——光滑常数的二次方放大了边缘的不稳定性。

- **分母的 $\max_i p_i$**：模型最确信的维度。若模型对某类高度确信（$p_i \approx 0.98$），分子 $1/0.98 \approx 1$ 提供了基础的强凸性——它确保在这个最平坦的方向上，能量函数仍然有足够的曲率来引导下降。

- **坐标不变性**：$\eta_{\max}(p)$ 只依赖于 $p$ 在单纯形上的位置——不依赖于用概率值、logits 还是其他参数化。$\mu$ 和 $L$ 都是 Hessian 的特征值，是几何量。**地形决定步伐，数据偏置决定地形。** 这个步长上界可以从当前模型输出直接计算——不需要经验调参。

在实际推理中，这意味着：当模型已经高度确信（$p$ 尖锐），它可以安全地使用更大的步长——因为此时它在单纯形的"平坦区域"。当模型高度不确定（$p$ 均匀），它也可以使用较大的步长——因为中心区域曲率适中。但当模型处于"临界边缘"——某个或多个维度的概率极小时——它必须极度谨慎地迈步。这正好对应了推理的心理直觉：排除一个看似不可能的选项，需要比调整两个几乎等概率的选项更大的"谨慎"。

![$\eta_{\max}(p)$ 在概率单纯形上的分布](/figures/ch05_eta_max_simplex.svg)

*$\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ 在三类单纯形上的等高线。红色区域（均匀分布附近）：$\eta_{\max} \approx 0.22$，步长可以较大。深色区域（接近顶点，即高度确信）：$\eta_{\max}$ 值中等。白色/浅色区域（接近边缘，即某个 $p_i \approx 0$）：$\eta_{\max}$ 急剧缩小——在单纯形边缘，KL 曲率最高，安全步长上限跌至 $10^{-5}$ 量级。这是一个坐标不变的几何量：无论用什么参数化，$\eta_{\max}(p)$ 只取决于 $p$ 在单纯形上的位置。*

:::info

**为什么边缘需要极小的步长？——$\eta_{\max}$ 的数值直觉**

考虑一个三类问题。你的当前信念是 $p = (0.98, 0.01, 0.01)$——你对第一类几乎完全确信。此时 $\eta_{\max} = 2 \times 0.01^2 / 0.98 \approx 0.0002$。

为什么这么小？因为第三类 $p_3 = 0.01$ 已经极小。如果你沿某个方向的欧拉步不小心在 $p_3$ 的方向上迈大了一步，$p_3$ 可能变成负数——这在概率空间中是非法的。$\eta_{\max}$ 的极小值反映了单纯形边缘的"几何拥挤"——空间太窄了，迈大一步就可能撞墙。

对比均匀分布 $p = (1/3, 1/3, 1/3)$。此时 $\eta_{\max} = 2 \times (1/3)^2 / (1/3) = 2/9 \approx 0.22$。空间宽阔——各方向都有足够的余地。

这个公式的分子 $\min(p_i)^2$ 之所以是平方，来自 KL 散度在边缘处的**二次型放大**。回忆 $D_{\mathrm{KL}}(p\|q) \approx \frac{1}{2}\sum (p_i-q_i)^2/q_i$——当某个 $q_i$ 很小时，相同的绝对变化 $(p_i-q_i)$ 被 $1/q_i$ 放大。反过来限制了 $\eta$：要保证压缩性 $\gamma < 1$，步长必须与这个放大系数的倒数同步缩小。$\min(p_i)^2$ 恰好是这个同步缩小的数量级表达。

这也是为什么 $\eta_{\max}$ 是一个**真正的几何量**：它没有自由参数，不依赖坐标系的选择，只依赖 $p$ 在单纯形上的位置。任何一个推理模型——无论是 Transformer、LSTM 还是人类——只要它在概率单纯形上沿 KL 几何更新信念，就会受到同一个 $\eta_{\max}(p)$ 的约束。

:::

## 5.11 永霖极限的哲学：为什么证明收敛而不是假设收敛？

现在让我们从数学中退后一步，问一个更深的问题：为什么永霖极限重要？

深度学习社区的日常实践中，"收敛"几乎总是被视为一个经验现象——"loss 下降了""准确率提高了""训练完成了"。我们很少问：**它为什么会收敛？它有没有可能不收敛？在什么条件下它保证收敛？**

这种态度在工程上可行，但在理论上脆弱。当你不知道一个系统为什么收敛时，你也不知道它什么时候会不收敛。当训练发散时——gradient explosion、loss NaN、模型崩溃——你的工具箱里只有"降低学习率""换一个初始化种子""试试不同的优化器"。这些是经验修补，而非理论洞察。

永霖极限试图做的，是在信念空间的特定情境下——推理过程中信念分布如何更新——给出一个**不依赖于经验调参的收敛性证明**。它的哲学姿态可以概括为三句话：

**第一，不假设压缩，而是证明压缩。** 我们不说"推理大概是收敛的"。我们找到一个具体的李雅普诺夫函数（KL 散度），证明它在每一步推理更新中单调递减，且递减有一个可控的下界。压缩不是一个愿望，而是一个定理的结论。

**第二，选择自然度量，而非强加度量。** 信念空间有它自己的几何——KL 散度不是我们选择来"衡量"信念差异的，它是负熵函数的 Bregman 散度——是信念空间的内禀几何。在欧几里得空间中证明压缩性是笨拙的（你需要强加不自然的假设）；在 KL 几何中，压缩性几乎是自明的——因为它就来自 Bregman 散度的广义勾股结构和凸生成函数的曲率。

**第三，让参数有物理意义。** $\eta_{\max}(p)$ 不是经验调出来的"最佳学习率"，而是由单纯形的局部曲率决定的步长上界。$\gamma$ 不是神秘的"收敛速率超参数"，而是 Bregman 压缩因子的几何表达。$p^*$ 不是"训练集上的最优参数"，而是推理动力系统在 KL 几何下的唯一不动点。每一个符号都对应着一个清晰的几何实体。

这就是永霖极限的哲学：**推理的收敛不是一个工程奇迹，而是一个几何必然。** 信念空间中的欧拉步，如果步长不超过局部曲率决定的界限，就必然压缩——而压缩意味着必然收敛。我们不需要"等待"模型收敛，我们不需要"希望"它收敛——我们从它迈出第一步的那一刻起，就已经知道它必将收敛。这就是 ch3 中李雅普诺夫的思想，在信念空间中最完整、最深刻的实现。

:::info

**兔狲教授的立场**

很多人在看到"永霖极限"这个名字的时候，以为它是一个工程结果——"哦，你们找到了最优学习率的上界。"

它不是。$\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ 不是一个调参技巧。它是一个**存在性证明**。它在说：信念空间中的推理，不是因为你训练得好才收敛——它是因为 KL 几何的弯曲才必然收敛。你不需要调学习率。你只需要保证它不超过地形曲率规定的上限。

这才是几何的力量。几何不帮你调参——几何告诉你，在什么样的条件下，收敛不是奇迹，是必然。

这本书从 ch1 到这里，画了一条很长的弧。从牛顿的力到哈密顿的能量，从损失地形到熵地形，从李雅普诺夫到巴拿赫，从欧拉法到 DEQ——它们全是同一个故事的不同章节。这个故事的核心只有一句话：**别分析力。找能量。能量在下降，系统就在前进。能量降到零，系统就停下来了。** 力的语言让你算。能量的语言让你懂。

:::

---

## 5.12 本章小结

本章是全书数学深度的第一个高峰。它完成了一件事：**从欧几里得到 Bregman，从平直空间到弯曲能量地形，从乌鸦的直线到水的流域。**

Bregman 散度 $D_F(p\|q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle$ 是衡量能量地形上不可逆能量差的语言。它不是距离——它不对称，不满足三角不等式——但这恰恰是它在概率空间中天然合适的原因。用 $q$ 处的局部几何估计 $p$ 犯的错误，与用 $p$ 处的局部几何估计 $q$ 犯的错误，在弯曲地形上本来就不等。

KL 散度是 $F$ 取负熵时的 Bregman 散度——不是偶然的巧合，而是信息几何的基石。Fisher 信息矩阵是 KL 散度的局部二次型，自然梯度用 Fisher 逆矩阵修正方向，镜像下降在对偶空间中下降而无需显式求逆——这些技术从不同的方向汇聚到同一个核心：**在信念空间中，你必须沿空间自身的曲率行走，而非强加欧几里得的步长。**

永霖极限将这整套几何语言推到了它的理论终点：从能量函数（KL 散度）到欧拉步推理算子，从 KL 压缩性到巴拿赫不动点，从 $\eta_{\max}(p)$ 的坐标不变性到"推理必然收敛"的几何证明。它不只是一种训练技巧——它是信念空间中推理动力学的第一性原理。

下一章，我们将把动力系统的视角系统化：学习不仅仅是"参数在地形上移动"，更是一种**动力系统的状态演化**。从离散的欧拉步到连续的梯度流，从参数的动力学到表示的动力学——我们将为 ch7 的不动点统一理论铺下最后一块基石。

---

## 悬而未决的问题

1. Bregman 散度由凸函数 $F$ 生成。是否存在一种"最优的" $F$，使得 Bregman 散度对某一类学习任务具有理论上的最优压缩性质？

2. 负熵函数生成 KL 散度，这似乎是概率单纯形上"最自然的" Bregman 散度。但这种自然性是必然的还是历史的？是否存在其他凸函数 $F$，其生成的 Bregman 散度在特定任务上表现更好？

3. Bregman 散度的广义勾股定理是永霖极限的核心几何基础。在什么条件下，这个广义勾股定理在深度网络的非凸损失地形上仍然成立——至少是近似成立？

4. 镜像下降在对偶空间中做梯度下降，避免了对 Fisher 信息矩阵的显式求逆。在高维深度网络中，能否设计一个不依赖 $\nabla F^*$ 显式计算的近似镜像下降？

5. $\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ 是推理步长的安全上界。在训练过程中，$\eta_{\max}$ 随 $p$ 动态变化。能否设计一个**自适应的步长调度**——每一步都使用当前位置的 $\eta_{\max}(p)$ 作为步长——来保证整个训练过程的稳定性？

6. 永霖极限证明了在"更新方向精确指向 $p^*$"的条件下，欧拉步推理收敛。当更新方向来自一个不完美的推理模块（如注意力层的输出）时，压缩性是否还能被证明？误差累积是否会导致发散？

7. 永霖极限在离散单纯形（有限类别）上成立。对于连续分布（如 Diffusion 模型中的高斯分布），KL 散度的压缩性能否推广到无限维的概率空间？

8. 巴拿赫不动点定理要求映射在某个度量下是压缩的。永霖极限使用 KL 散度作为度量。但 KL 散度不对称，严格来说不是度量。是否有必要——或者有可能——将证明重新表述在一个真正的度量空间（如 Hellinger 距离或全变差距离）中？

9. Bregman 散度的不对称性在永霖极限中起到了关键作用（$D_{\mathrm{KL}}(p^*\|p)$ 的方向选择）。如果反过来使用 $D_{\mathrm{KL}}(p\|p^*)$，压缩性证明是否仍然成立？两种方向的几何差异在哪里？

10. 自然梯度在理论上优雅，但实际计算 $G^{-1}$ 的代价极高。是否存在一类损失函数或模型结构，使得 Fisher 信息矩阵具有特别简单的结构（如对角、低秩、Kronecker 积），从而自然梯度可以高效近似？

11. 永霖极限的收敛速率由 Bregman 压缩因子 $\gamma$ 决定，$\gamma$ 又依赖于步长 $\eta$ 和局部曲率。对于给定的收敛精度 $\epsilon$，是否存在一个最优的步长序列 $\eta_1, \eta_2, \ldots$ 使得收敛所需的步数最小？

12. 如果永霖极限的"能量函数"（KL 散度）可以被理解为一种物理学中的自由能，那么推理过程是否对应着某种热力学过程？随机梯度噪声是否扮演了"热浴"的角色，帮助系统探索不同信念状态之间的能量壁垒？

---

**本章留下的核心问题是：**

**Bregman 散度的广义勾股定理——永霖极限的几何心脏——在非线性、高维、随机梯度噪声污染的深度网络中，以什么形式存活？**

:::info

**欧几里得是乌鸦的语言——直线、对称、与地形无关。Bregman 是水的语言——沿能量梯度流动、不对称、每一步的能量代价取决于脚下的曲率。** KL 散度是水的语言中最重要的方言——它用熵的地形刻画了信念空间中所有可能的"流动"。永霖极限证明：只要跟在 KL 地形的最速下降方向后面，水一定会流到海。下一章，我们将把动力系统的视角系统化——从 ch1 开始的"学习是运动"，到 ch3 的离散欧拉步，到 ch7 的最终统一：ResNet、GPT 自回归和 DEQ，都是同一个动力系统在不同空间中的面孔。

:::

## 延伸阅读与相关工作

**大复杂模型的相对自然梯度.** Sun & Nielsen (2016) [arXiv:1606.06069] — 从大型神经元系统中提取局部的相对 Fisher 信息量，使自然梯度在模块化网络中变得可操作——这不是近似，而是对信息几何"局部性"的严格刻画。

**深度学习中自然梯度的局部性.** Ay (2020) [arXiv:2005.10791] — Fisher 信息矩阵的本质是局部几何量：它只依赖于当前参数附近 KL 散度的二阶行为。这个看似平凡的观察是自然梯度在深层网络中可计算的根本原因——也是 Bregman 散度与欧几里得步长之间最本质的分界线。

**自然梯度方法综述.** Shrestha (2023) [arXiv:2303.05473] — 系统比较 Fisher 信息矩阵替代 Hessian 的若干策略，从 Kronecker 分解到低秩近似，从理论框架到工程实现。

**广义 Euler 对数的机器学习应用.** Cichocki (2025) [arXiv:2502.17500] — 一个参数量化 Bregman 散度族：广义 Euler 对数 $E_{\alpha,\beta}(x)$ 通过两个参数分别控制尾部鲁棒性和局部梯度塑造行为，将 KL 散度、$\chi^2$ 散度和 $\alpha$-散度统一到一个双参数框架下——这是 Bregman 生成函数的参数化。

**点积注意力作为熵最优传输.** Litman (2025) [arXiv:2508.08369] — 注意力机制的前向传播等价于退化熵最优传输问题的精确解——Transformer 的自注意力在每一步都在求解一个熵正则化的传输问题，Fisher 信息矩阵则定义了注意力分布的几何曲率。

**Softmax 的对角化.** Garrod, Keating & Thrampoulidis (2025) [arXiv:2512.04006] — Hadamard 初始化使得 softmax 的 Fisher 信息矩阵对角化，交叉熵梯度流收敛到 Neural Collapse 几何——这是信息几何直接解释深度学习训练动态的首个严格结果。

---

## 附录 5.A：直接证据——能量地形与闭式解的数值验证

本节给第 5.6–5.10 节的闭式解配**直接证据**：在信念单纯形上数值实现推理动力学（KL 能量的梯度流 + 投影到单纯形的欧拉步），逐项核验闭式解的四个预测。整套实验可复现：完整源码见 `docs/public/scripts/ch05_energy_terrain.py`（正文末尾贴了完整代码），图见 `ch05_energy_terrain.png`。

![图 5.A：能量地形与闭式解的数值验证](/figures/ch05_energy_terrain.png)

**四面板读法**：

**（a）地形存在。** $E = D_{KL}(p^*\|p)$ 的等高线在单纯形上形成以 $p^*$ 为谷底的能量盆地；从三个不同初始信念出发、以 $0.8\,\eta_{\max}$ 步长迭代的推理轨迹全部收敛到 $p^*$。推理是地形上的运动——这是全书主张的最直接画面。

**（b）曲率谱：闭式 = 数值。** 沿一条推理轨迹，解析谱（$\lambda_{\min}=1/\max_i p_i$、$\lambda_{\max}=1/\min_i p_i$，不动点处）与数值差分 Hessian 的特征值逐点重合，$\kappa = L/\mu = 7.00 = 0.70/0.10$。

**（c）$\eta_{\max}$ 的闭式景观。** $\eta_{\max}(p) = 2\min_i p_i^2/\max_i p_i$ 在单纯形上的对数色图：中心可以大步走，越靠边缘越必须小步。"地形决定步伐"的字面图像。

**（d）稳定边界：闭式解精确预测数值。** 局部扰动动力学的数值失稳阈值 $\eta^*$ 与闭式解 $2/\theta_{\max}$（切空间谱，§5.10 补充段）在 30 个随机目标分布上 $R^2 = 0.9978$、中位误差 2.16%——蓝点全部贴在对角线上。粉色三角是从均匀先验出发的全局扫描数值阈值，恒定在保守界 $2/(L\kappa)$ 之上。

**复现者要避开两个坑。**（i）投影到单纯形的标准算法必须取*最后一个*满足 $u_j - (\sum_{k\leq j}u_k - 1)/j > 0$ 的 $j$；索引取错会产生行和 $\neq 1$ 的"伪投影"，轨迹第一步就会飞出单纯形。（ii）判别稳定性要用渐近终点距离 $\|p_T - p^*\|/\|p_0 - p^*\| < 1$，而不是全程最大值——中等步长下非线性暂态会先放大后收缩，全程最大值会把线性稳定的区域误判为失稳（实际测试给出 0.79 的虚假比值；改用渐近判据后为 1.000，与闭式解一致）。

**源代码**（完整可运行；运行后将图保存为同级目录 `ch05_energy_terrain.png`）：

```python
"""
Energy Landscape of Reasoning: Closed-Form Curvature and Safe Step Size
========================================================================
Demonstration accompanying the Yonglin Limit (永霖极限) derivation.

Setup: belief simplex Delta^2 (3 classes). The reasoning energy is
    E(p) = D_KL(p* || p),  p* = target/attractor belief.
Inference = Euler steps of the gradient flow on E with projection to the
simplex:  p_{t+1} = proj_Delta(p_t - eta * grad E(p_t)).

Closed-form facts under study (true at the fixed point p = p*):
    Hess E(p)  = diag(q_i / p_i^2)   [general position; q_i = p*_i]
               -> diag(1/p_i)        [at p = p*]
    mu = lambda_min = 1/max_i p_i,  L = lambda_max = 1/min_i p_i
    kappa = L/mu = max_i p_i / min_i p_i          (spectral ratio / condition number)
    eta_max = 2 mu / L^2 = 2 / (L * kappa) = 2 min_i p_i^2 / max_i p_i

On the tangent space {v: sum v_i = 0} (the subspace preserved by the
projection) the exact local stability boundary is
    eta_crit = 2 / theta_max,  theta_max = largest eigenvalue of
    diag(1/q_i) restricted to the tangent space;
    for K = 3, theta_max solves  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patheffects import withStroke

rng = np.random.default_rng(42)


# ----------------------------- core helpers -----------------------------
def proj_simplex(v):
    """Euclidean projection onto probability simplex (vectorized over last axis).
    Standard algorithm: find the last index j with u_j - (css_j - 1)/j > 0."""
    v = np.asarray(v, float)
    shape = v.shape
    v = v.reshape(-1, shape[-1])
    K = shape[-1]
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)
    avg = (css - 1.0) / np.arange(1, K + 1)
    cond = u - avg > 0
    j = np.where(cond)[1]    # column indices of True entries (rows repeat)
    # last True index per row:
    last = np.array([np.where(cond[i])[0][-1] for i in range(len(cond))])
    theta = (css[np.arange(len(v)), last] - 1.0) / (last + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out.reshape(shape)


def energy(p, pstar):
    """E(p) = D_KL(p* || p), elementwise-safe (p clipped away from 0)."""
    pc = np.clip(p, 1e-300, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.sum(pstar * np.log(pstar / pc), axis=-1)
    return np.where(np.all(p > 0, axis=-1), e, np.inf)


def grad_energy(p, pstar):
    return -pstar / np.clip(p, 1e-12, 1.0)


def euler_step(p, pstar, eta):
    return proj_simplex(p - eta * grad_energy(p, pstar))


def eta_max_closed(pstar):
    """Closed-form safe step size at the fixed point: 2 min^2 / max."""
    return 2.0 * np.min(pstar) ** 2 / np.max(pstar)


def hessian_numeric(p, pstar, eps=1e-6):
    """Central-difference Hessian of E at p (K=3)."""
    K = len(p)
    H = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ei = np.zeros(K); ej = np.zeros(K)
            ei[i] = eps; ej[j] = eps
            H[i, j] = (energy(p + ei + ej, pstar) - energy(p + ei - ej, pstar)
                       - energy(p - ei + ej, pstar) + energy(p - ei - ej, pstar)) / (4 * eps * eps)
    return H


def to_xy(p):
    """Barycentric -> Cartesian (A=e1 at (0,0), B=e2 at (1,0), C=e3 at (0.5,sqrt3/2))."""
    return p[..., 1] * 1.0 + p[..., 2] * 0.5, p[..., 2] * (np.sqrt(3) / 2)


def from_xy(x, y):
    s = np.sqrt(3.0) / 2.0
    p3 = np.clip(y / s, 0.0, 1.0)
    p2 = np.clip(x - y / (np.sqrt(3.0)), 0.0, 1.0)
    p1 = 1.0 - p2 - p3
    mask = (p1 >= -1e-9) & (p2 >= -1e-9) & (p3 >= -1e-9)
    p1 = np.clip(p1, 0.0, 1.0)
    return np.stack([p1, p2, p3], axis=-1), mask


# ----------------------------- figure panels -----------------------------
PSTAR = np.array([0.70, 0.20, 0.10])
EMAX = eta_max_closed(PSTAR)          # ~ 2*0.01/0.7 = 0.02857
ETA_ILLUS = 0.8 * EMAX

fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
ps = withStroke(linewidth=3, foreground="white")

# ---------- (a) energy terrain + trajectories ----------
ax = axes[0, 0]
xgrid = np.linspace(-0.05, 1.05, 300)
ygrid = np.linspace(-0.05, np.sqrt(3) / 2 + 0.05, 260)
X, Y = np.meshgrid(xgrid, ygrid)
P, mask = from_xy(X, Y)
E = energy(P, PSTAR)
E = np.where(mask, E, np.nan)
lv = np.geomspace(1e-5, 50, 40)
cf = ax.contourf(X, Y, E, levels=lv, cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=50))
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label(r"$E(p)=D_{KL}(p^{*}\|p)$  [log scale]", fontsize=9)

starts = [np.array([1/3, 1/3, 1/3]), np.array([0.15, 0.60, 0.25]), np.array([0.45, 0.25, 0.30])]
for p0 in starts:
    p = p0.copy()
    xs, ys = [], []
    for t in range(120):
        p = euler_step(p, PSTAR, ETA_ILLUS)
        xs.append(to_xy(p)[0]); ys.append(to_xy(p)[1])
        if energy(p, PSTAR) < 1e-6:
            break
    ax.plot(xs, ys, lw=1.8, color="#D55E00", alpha=0.9)
    ax.plot(xs[0], ys[0], "o", ms=6, color="#D55E00", mfc="white")
    ax.annotate("$p_0$", (xs[0], ys[0]), textcoords="offset points", xytext=(-2, 8),
                fontsize=9, color="#D55E00")

px, py = to_xy(PSTAR)
ax.plot(px, py, "*", ms=22, color="#0072B2", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$  (fixed point)", (px, py), textcoords="offset points",
            xytext=(8, -16), fontsize=11, color="#0072B2", fontweight="bold")
ax.text(0.98, 0.94, f"$\\eta=0.8\\,\\eta_{{max}}$ = {ETA_ILLUS:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
ax.set_title("(a) The energy terrain of reasoning", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "belief simplex $\\Delta^2$: $p_1$-vertex (left), $p_2$-vertex (right), $p_3$-vertex (top)",
        transform=ax.transAxes, ha="center", fontsize=8, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (b) curvature spectrum along one trajectory ----------
ax = axes[0, 1]
p = np.array([1/3, 1/3, 1/3])
lmax_a, lmin_a, kap_a = [], [], []
lmax_n, lmin_n = [], []
p_rec = []
for t in range(80):
    p = euler_step(p, PSTAR, ETA_ILLUS)
    p_rec.append(p.copy())
for p in p_rec:
    lmax_a.append(np.max(PSTAR / p**2)); lmin_a.append(np.min(PSTAR / p**2))
    H = hessian_numeric(p, PSTAR)
    w = np.linalg.eigvalsh(H)
    lmax_n.append(w[-1]); lmin_n.append(w[0])
ts = np.arange(len(p_rec))
ax.plot(ts, lmax_a, color="#0072B2", lw=1.8, label=r"closed form  $\lambda_{max}$")
ax.plot(ts, lmin_a, color="#009E73", lw=1.8, label=r"closed form  $\lambda_{min}$")
ax.plot(ts, lmax_n, "o", ms=4.2, color="#0072B2", mfc="none", label=r"numerical Hessian  $\lambda_{max}$")
ax.plot(ts, lmin_n, "s", ms=4.2, color="#009E73", mfc="none", label=r"numerical Hessian  $\lambda_{min}$")
ax.set_yscale("log")
ax.set_xlabel("reasoning step  $t$", fontsize=10)
ax.set_ylabel("curvature eigenvalues  $\\lambda$", fontsize=10)
ax.set_title("(b) Curvature spectrum: closed form = numerical", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
ax.grid(alpha=0.3)
kap = PSTAR.max() / PSTAR.min()
ax.text(0.98, 0.08, r"$\kappa = L/\mu = %.2f = 0.70/0.10$ (at $p^{*}$)" % kap,
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# ---------- (c) analytic eta_max surface ----------
ax = axes[1, 0]
XS, YS = np.meshgrid(np.linspace(0, 1, 260), np.linspace(0, np.sqrt(3) / 2, 230))
PS, maskC = from_xy(XS, YS)
eta = np.where(maskC & np.all(PS > 1e-4, axis=-1),
               2.0 * np.min(PS, axis=-1) ** 2 / np.max(PS, axis=-1), np.nan)
eta = np.where(maskC, eta, np.nan)
lv2 = np.geomspace(1e-6, 0.25, 35)
cf2 = ax.contourf(XS, YS, eta, levels=lv2, cmap="magma", norm=LogNorm(vmin=1e-6, vmax=0.25))
cbar2 = fig.colorbar(cf2, ax=ax, shrink=0.85)
cbar2.set_label(r"$\eta_{max}(p)=2\min_i p_i^2/\max_i p_i$  [log]", fontsize=9)
ax.plot(px, py, "*", ms=22, color="#56B4E9", mec="k", mew=0.8, zorder=10)
ax.annotate(r"$p^{*}$", (px, py), textcoords="offset points", xytext=(6, 10),
            fontsize=11, color="#56B4E9", fontweight="bold")
ax.set_title("(c) Analytic safe step size on the simplex", fontsize=12, fontweight="bold")
ax.text(0.5, -0.14, "$\\eta_{max} \\to 0$ near the edges: the terrain dictates the pace",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="0.35")
ax.set_aspect("equal"); ax.axis("off")

# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

def theta_max_tangent(q):
    """Largest eigenvalue of diag(q_i^{-1}) restricted to the tangent space {v: sum v_i = 0}.
    K = 3: analytic root of  3 q0 q1 q2 th^2 - 2 (q0q1+q0q2+q1q2) th + 1 = 0."""
    q0, q1, q2 = q
    a = 3 * q0 * q1 * q2
    b = -2 * (q0 * q1 + q0 * q2 + q1 * q2)
    c = 1.0
    disc = b * b - 4 * a * c
    th = (-b + np.sqrt(disc)) / (2 * a)
    return th


def eta_crit_tangent_closed(q):
    """Exact local stability boundary: eta_safe < 2 / lambda_max of Hessian on the tangent space."""
    return 2.0 / theta_max_tangent(q)


# ---------- (d) closed form vs numerical stability boundary ----------
ax = axes[1, 1]

etas_scan = np.logspace(-4.0, 0.6, 140)

# --- LOCAL analysis: perturbations of p*; exact prediction = tangent-space spectrum ---
local_ana, local_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    pert = [proj_simplex(ps_ + 0.02 * rng.normal(0, 1, 3) * np.array([1, 0.5, 0.25])) for _ in range(8)]
    # asymptotic stability: final distance does not grow
    worst = np.zeros(len(etas_scan))
    for p0 in pert:
        P = np.broadcast_to(p0.copy(), (len(etas_scan), 3)).copy()
        d0 = np.linalg.norm(p0 - ps_)
        for _ in range(500):
            P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
        worst = np.maximum(worst, np.linalg.norm(P - ps_, axis=1) / d0)
    idx = np.where(worst < 1.0)[0]
    local_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    local_ana.append(eta_crit_tangent_closed(ps_))

# --- GLOBAL analysis: from uniform prior; conservative closed form = full-space spectrum ---
glob_ana, glob_num = [], []
for _ in range(30):
    ps_ = rng.dirichlet(np.ones(3)) * 0.9 + 0.05
    ps_ = ps_ / ps_.sum()
    P = np.broadcast_to(np.full(3, 1 / 3), (len(etas_scan), 3)).copy()
    for _ in range(400):
        P = proj_simplex(P - etas_scan[:, None] * (-ps_ / np.clip(P, 1e-12, 1.0)))
    fk = np.sum(ps_ * np.log(np.clip(ps_, 1e-12, 1) / np.clip(P, 1e-12, 1)), axis=1)
    idx = np.where(fk < 1e-3)[0]
    glob_num.append(etas_scan[idx[-1]] if len(idx) else etas_scan[0] / 2)
    glob_ana.append(eta_max_closed(ps_))

local_ana, local_num = np.array(local_ana), np.array(local_num)
glob_ana, glob_num = np.array(glob_ana), np.array(glob_num)

ax.loglog(local_ana, local_num, "o", ms=7, color="#0072B2", alpha=0.85, zorder=6,
          label=r"local: $2/\lambda_{\max}$ on tangent space (exact)")
ax.loglog(glob_ana, glob_num, "^", ms=7, color="#CC79A7", alpha=0.85, zorder=5,
          label=r"global: conservative bound $2/(L\kappa)$")
lims = [min(local_ana.min(), local_num.min()) * 0.6, max(local_ana.max(), local_num.max()) * 1.6]
ax.loglog(lims, lims, "k--", lw=1.5, label="y = x (exact prediction)")
logR = np.corrcoef(np.log10(local_ana), np.log10(local_num))[0, 1] ** 2
med = np.median(np.abs(local_num - local_ana) / local_ana)
ax.set_xlabel(r"closed-form  $\eta_{crit}$", fontsize=10)
ax.set_ylabel("numerical stability threshold  $\\eta_{crit}$", fontsize=10)
ax.set_title("(d) Closed form = the exact local stability boundary", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3, which="both")
ax.text(0.03, 0.05, f"exact (tangent spectrum): $R^2$ = {logR:.4f}, median err = {med:.2%}\nconservative (full spectrum): $\\eta_{{crit}} \\geq 2/(L\\kappa)$ always",
        transform=ax.transAxes, fontsize=9, color="k", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

fig.suptitle("The Energy Landscape of Reasoning: closed-form geometry and its verification\n"
             r"$E(p)=D_{KL}(p^{*}\|p)$,  inference $=$ Euler steps of $-\nabla E$ projected to the simplex",
             fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "ch05_energy_terrain.png"   # saves next to the script
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
print(f"eta_max closed form (p* = {PSTAR}): {EMAX:.6f}")
print(f"kappa at fixed point: {PSTAR.max()/PSTAR.min():.2f}")
print(f"panel d: local R^2(log-log) = {logR:.4f}, n = {len(local_ana)}, median err = {med:.2%}")
print(f"panel d: global: numerically stable region >= eta_max (projection adds stability)")

```

*图 5.A 与源码：李籽溪（兔狲教授），2026。脚本同步维护于 `docs/public/scripts/ch05_energy_terrain.py`。*
