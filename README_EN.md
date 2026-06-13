<h1 align="center">The Terrain of Learning</h1>
<h3 align="center">An Introduction to the Geometry of Reasoning &amp; the Dynamics of Intelligence</h3>

<div align="center">
  <img src="docs/public/cover_cn.png" alt="The Terrain of Learning" width="400">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![VitePress](https://img.shields.io/badge/built%20with-VitePress-646cff?style=flat-square)](https://vitepress.dev/)
[![Discussions](https://img.shields.io/badge/join-Discussions-6c5ce7?style=flat-square)](https://github.com/datawhalechina/learning-terrain/discussions)

</div>

---

<div align="center">

**🌐 [中文版](README.md) | [在线阅读](https://datawhalechina.github.io/learning-terrain/) | [💬 读者交流](https://github.com/datawhalechina/learning-terrain/discussions/2)**

</div>

---

*The Terrain of Learning* is an open-source book that re-understands deep learning from a geometric perspective. 12 chapters across 4 volumes, bilingual (Chinese/English), answering one unifying question: **If learning is not magic inside a black box, but a real terrain—parameter space is the landscape, the loss function is elevation, the gradient is the direction of slope, and training is motion downhill—then what is the structure of this terrain? And how does reasoning happen upon it?**

This is neither a textbook nor a collection of papers. It is a perspective shift: from "force-by-force analysis" to "the holistic portrait of the energy landscape"—doing to the foundational narrative of deep learning what Hamilton did to Newton.

Four volumes, three chapters each:

- **Volume I: The Terrain of Learning** (Ch.1–3) — Establishing geometric intuition. Parameter space, representation space, loss landscape, gradient fields. From Newtonian mechanics to Hamiltonian mechanics, from force analysis to energy terrain.
- **Volume II: The Dynamics of Intelligence** (Ch.4–6) — Optimizers are ways of walking. Non-Euclidean distances (Bregman divergence, KL divergence). Dynamical systems and fixed points—ResNet is explicit Euler, GPT autoregression is implicit-state Euler iteration, DEQ is a fixed point.
- **Volume III: The Geometry of Reasoning** (Ch.7–9) — A chain of thought is not reasoning itself, but the visible projection of reasoning trajectories. Reasoning fields, attractors, verifiers. The geomorphology of long reasoning—the number of reasoning steps is determined by the terrain, not by the difficulty of the problem.
- **Volume IV: Algorithmic Landscapes** (Ch.10–12) — Classical algorithms geometrized. Linear regression, SVM, Attention, LoRA, diffusion models—one chapter, one diagram, one equation, one experiment.

Complete book, bilingual, with 30+ Nature-print-grade SVG figures.

> **The core thread**: From $F=ma$ (1687) to diffusion models (2020), across 337 years, one idea has never changed—**a system evolves along an energy landscape, from high energy to low energy, from disorder to order, from noise to structure.** Learning is this. Reasoning is this. Generation is this.
>
> **Force lets you compute. Energy lets you understand. Geometry lets you see.** *Compute*—you can find the gradient, run Adam. *Understand*—you grasp why KL geometry's curvature forces small steps, why a dynamical system's fixed point is the endpoint of convergence. *See*—you close your eyes and what you see is no longer formulas but terrain: linear regression is the bottom of an elliptical bowl. PCA is the longest axis of the data cloud. A chain of thought is the trajectory of Euler steps of hidden states along a reasoning field on the simplex. Diffusion is the score vector field pointing toward "more data." After twelve chapters, you are no longer a blind person in the wilderness. You see.

---

## Table of Contents

### Volume I: The Terrain of Learning (Ch.1–3)

| Chapter | Core Question | Status |
| :--- | :--- | :---: |
| [Volume Introduction: Parameter Space is the Body → Loss Landscape is the Slope → Gradient Descent is Walking → Representation Space is the View](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/) | A blind person dropped into the wilderness—the three chapters of Volume I build the book's central logical chain | ✅ |
| [Ch.1: Why Learning Needs Geometry](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch01) | From $F=ma$ to $H=T+V$: why force-by-force analysis is insufficient and we must move to the energy landscape | ✅ |
| [Ch.2: Body and View: Parameter Space and Representation Space](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch02) | Parameter space is the model's body; representation space is the model's view | ✅ |
| [Ch.3: Loss Landscape and Gradient Motion](https://datawhalechina.github.io/learning-terrain/01_the_terrain_of_learning/ch03) | The loss function draws a terrain over parameter space; gradient descent is motion downhill | ✅ |

### Volume II: The Dynamics of Intelligence (Ch.4–6)

| Chapter | Core Question | Status |
| :--- | :--- | :---: |
| [Volume Introduction: The Space Beneath Your Feet Is Not Flat](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/) | Curved space has rigorous formalization—from information geometry to dynamical systems to fixed points | ✅ |
| [Ch.4: Ways of Walking: Optimizers and Regularization](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch04) | SGD, momentum, Adam—do you stride boldly, or take cautious small steps? | ✅ |
| [Ch.5: Non-Euclidean Worlds: Bregman Divergence and KL Divergence](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch05) | Belief space is not flat—Bregman divergence is the energy difference on the entropy landscape | ✅ |
| [Ch.6: Dynamical Systems and Fixed Points: From Lyapunov to DEQ](https://datawhalechina.github.io/learning-terrain/02_dynamics_of_intelligence/ch06) | Learning is the evolution of a dynamical system. Fixed points, attractors, stability—the Yonglin Limit | ✅ |

### Volume III: The Geometry of Reasoning (Ch.7–9)

| Chapter | Core Question | Status |
| :--- | :--- | :---: |
| [Volume Introduction: Reasoning is the Flow of Hidden States Along a Reasoning Field](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/) | Don't look at tokens. Look at trajectories. Look at fields. Look at fixed points. Reasoning is not behavior—reasoning is geometry | ✅ |
| [Ch.7: Chain of Thought: The Projection of Reasoning Trajectories](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch07) | The tokens you see are shadows—real reasoning happens in hidden state space | ✅ |
| [Ch.8: The Reasoning Field: Attractors and Verifiers](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch08) | The structure of the reasoning field. Correct answers have wide basins; wrong answers are grooves carved by training data | ✅ |
| [Ch.9: Long Reasoning and Landscape Remodeling](https://datawhalechina.github.io/learning-terrain/03_geometry_of_reasoning/ch09) | Reasoning step count is determined by the terrain—it's not that the model "isn't smart enough," the ridges are too jagged | ✅ |

### Volume IV: Algorithmic Landscapes (Ch.10–12)

| Chapter | Core Question | Status |
| :--- | :--- | :---: |
| [Volume Introduction: Every Formula You've Learned is a Contour Line of a Landscape](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/) | Volume IV is not "applications." Volume IV is coming home—seeing every classical algorithm anew in geometric language | ✅ |
| [Ch.10: The Geometry of Classical Algorithms](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch10) | Linear regression, PCA, SVM—classical algorithms have never been clearer than in geometric language | ✅ |
| [Ch.11: The Geometry of Deep Architectures](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch11) | Attention, LoRA, Transformers—a geomorphological reading of deep architectures | ✅ |
| [Ch.12: Diffusion and Convergence](https://datawhalechina.github.io/learning-terrain/04_algorithmic_landscapes/ch12) | The geometry of diffusion models: forward is entropy increase, reverse is flowing along the score field—the book converges | ✅ |

### Epilogue: Seeing

| Chapter | Core Question | Status |
| :--- | :--- | :---: |
| [Epilogue: Seeing](https://datawhalechina.github.io/learning-terrain/epilogue) | Twelve chapters walked, looking back at the path—geometry cheat sheet, formula family tree, reading roadmap | ✅ |

---

## About the Author

**Zixi Li (Pallas's Cat)** — Independent AI researcher. Working at the intersection of reasoning geometry, optimization theory, dynamical systems, and deep learning. Author of *The Reasoning Kingdom*. Introduces a geometric perspective into the foundations of deep learning, unifying the complete narrative from gradient descent to diffusion models under "energy landscapes."

Professor Pallas's Cat resides in the Blackstone House at Sun Yat-sen University—at least, that's what the book says.

---

## Local Development

This project is built with [VitePress](https://vitepress.dev/).

```bash
# Install dependencies (requires --legacy-peer-deps)
npm install --legacy-peer-deps

# Start Chinese dev server
npm run docs:dev

# Build Chinese production
npm run docs:build

# Preview
npm run docs:preview
```

The **English edition** uses a separate `docs_en` config directory for building:

```bash
# Build English edition
npm run build:en
```

---

## Technical Details

- **Math**: KaTeX rendering, `$ ... $` inline, `$$ ... $$` block
- **Figures**: Nature-print-grade SVG, Okabe-Ito palette, single-column 3.46in / double-column 7.09in
- **Custom containers**: `:::info` supports narrative explanations, cross-chapter bridges, and author commentary
- **Bilingual**: Chinese sources in `docs/`, English translations in `docs/en/`

---

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

---

<div align="center">

**Convergence is not hope. Convergence is geometry. You see.**

</div>
