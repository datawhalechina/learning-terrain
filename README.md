<h1 align="center">学习的地形</h1>
<h3 align="center">推理几何与智能动力学导论</h3>

<div align="center">
  <img src="docs/public/cover_cn.png" alt="学习的地形" width="400">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![VitePress](https://img.shields.io/badge/built%20with-VitePress-646cff?style=flat-square)](https://vitepress.dev/)

</div>

---

《学习的地形》是一部从几何视角重新理解深度学习的开源书籍。全书 12 章、四卷、中英双语，回答一个统一的问题：**如果学习不是黑箱里的魔法，而是一片真实的地形——参数空间是地貌，损失函数是海拔，梯度是坡度的方向，训练是沿坡而下的运动——那么这片地形的结构是什么？推理如何在这片地形上发生？**

这不是教材，也不是论文合集。它是一次视角转换：从"力的逐项分析"走向"能量地形的整体刻画"——用哈密顿对牛顿做的同一件事，翻掉深度学习的基础叙事。

全书四卷，每卷三章：

- **卷一：学习的地形**（第1–3章）——建立几何直觉。参数空间、表示空间、损失地形、梯度场。从牛顿力学到哈密顿力学，从受力分析到能量地形。
- **卷二：行走与信念**（第4–6章）——优化器是行走的方式。非欧距离（Bregman散度、KL散度）。动力系统与不动点——ResNet 是显式欧拉法，GPT 自回归是隐状态欧拉迭代，DEQ 是不动点。
- **卷三：推理的几何**（第7–9章）——思维链不是推理本身，而是推理轨迹的可见投影。推理场、吸引子、验证器。长推理的地貌学——推理步数由地形决定，而非由问题难度决定。
- **卷四：算法的地貌学**（第10–12章）——经典算法几何化。线性回归、SVM、Attention、LoRA、扩散模型——每章一图，一公式，一实验。

全书完成，中英双语，含 30+ 张 Nature 印刷级 SVG 配图。

> **核心线索**：从 $F=ma$（1687）到扩散模型（2020），337 年的跨度中，有一个思想始终未变——**系统沿能量地形演化，从高能量到低能量，从无序到有序，从噪声到结构。** 学习如此。推理如此。生成如此。
>
> **力让你算。能量让你懂。几何让你看见。** 算——你会求梯度、跑 Adam。懂——你理解了 KL 几何的曲率为什么压小步长、动力系统的不动点为什么是收敛的终点。看见——你闭上眼，看见的不再是公式，而是地形：线性回归是一只椭圆碗的底。PCA 是数据云最长的轴。思维链是隐藏状态在单纯形上沿推理场的欧拉步轨迹。扩散是得分向量场指向"更多数据"的方向。十二章走完，你不是在荒野中蒙眼的盲人了。你看见了。

---

## 目录

### 卷一：学习的地形（第1–3章）

| 章节 | 核心问题 | 状态 |
| :--- | :--- | :---: |
| [第1章：为什么学习需要几何](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch01) | 从 $F=ma$ 到 $H=T+V$：为什么力的逐项分析不够，必须走向能量地形 | ✅ |
| [第2章：身体与视野：参数空间与表示空间](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch02) | 参数空间是模型的身体，表示空间是模型的视野 | ✅ |
| [第3章：损失地形与梯度运动](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch03) | 损失函数在参数空间上画出一片地形，梯度下降是沿坡而下的运动 | ✅ |

### 卷二：行走与信念（第4–6章）

| 章节 | 核心问题 | 状态 |
| :--- | :--- | :---: |
| [第4章：行走的方式：优化器与正则化](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch04) | SGD、动量、Adam——你是大步流星，还是碎步谨慎？ | ✅ |
| [第5章：非欧世界：Bregman 散度与 KL 散度](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch05) | 信念空间不是平直的——Bregman 散度是熵地形上的能量差 | ✅ |
| [第6章：动力系统与不动点：从李雅普诺夫到 DEQ](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch06) | 学习是动力系统的演化。不动点、吸引子、稳定性——永霖极限 | ✅ |

### 卷三：推理的几何（第7–9章）

| 章节 | 核心问题 | 状态 |
| :--- | :--- | :---: |
| [第7章：思维链：推理轨迹的投影](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch07) | 你看到的 token 只是影子——真正的推理发生在隐藏状态空间 | ✅ |
| [第8章：推理场：吸引子与验证器](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch08) | 推理场的结构。正确答案有宽阔的盆地，错误答案是训练数据刻出的沟壑 | ✅ |
| [第9章：长推理与地貌重塑](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch09) | 推理步数由地形决定——不是模型"不够聪明"，是山脊太崎岖 | ✅ |

### 卷四：算法的地貌学（第10–12章）

| 章节 | 核心问题 | 状态 |
| :--- | :--- | :---: |
| [第10章：经典算法的几何](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch10) | 线性回归、PCA、SVM——经典算法在几何语言中从未如此清晰 | ✅ |
| [第11章：深度架构的几何](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch11) | Attention、LoRA、Transformer——深度架构的地貌学解读 | ✅ |
| [第12章：扩散与收敛](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch12) | 扩散模型的几何：前向是熵增，逆向是沿得分场流动——全书收束 | ✅ |

### 尾声：看见

| 章节 | 核心问题 | 状态 |
| :--- | :--- | :---: |
| [尾声：看见](https://datawhalechina.github.io/learning-terrain/epilogue) | 十二章走完，回望来时路——几何速查表、公式族谱、阅读路线图 | ✅ |

---

## 关于作者

**李籽溪（兔狲）**——独立 AI 研究者。致力于推理几何、优化理论、动力系统与深度学习的交叉研究。《推理王国》作者。将几何视角引入深度学习基础理论，用"能量地形"统一了从梯度下降到扩散模型的完整叙事。

兔狲教授住在中山大学黑石屋——至少书里是这么写的。

---

## 本地开发

本项目使用 [VitePress](https://vitepress.dev/) 构建。

```bash
# 安装依赖（需要 --legacy-peer-deps）
npm install --legacy-peer-deps

# 启动中文开发服务器
npm run docs:dev

# 构建中文生产版本
npm run docs:build

# 预览
npm run docs:preview
```

**英文版**使用独立的 `docs_en` 配置目录构建：

```bash
# 构建英文版
npm run build:en
```

---

## 技术细节

- **数学公式**：KaTeX 渲染，`$ ... $` 行内，`$$ ... $$` 块级
- **出图规范**：Nature 印刷级 SVG，Okabe-Ito 色板，单栏 3.46in / 双栏 7.09in
- **自定义容器**：`:::info` 支持叙事解释、跨章桥接和作者立场评论
- **中英双语**：中文源文件在 `docs/`，英文翻译在 `docs/en/`

---

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

本作品采用[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](http://creativecommons.org/licenses/by-nc-sa/4.0/)进行许可。

---

<div align="center">

**收敛不是希望。收敛是几何。你看见了。**

</div>
