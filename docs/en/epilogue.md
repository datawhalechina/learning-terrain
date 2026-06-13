# Epilogue: Seeing

You have turned this book to its final page.

In 1687, Newton wrote $F=ma$ in his room at Cambridge. He did not know he was starting an arc—an arc that would pass through Hamilton, through Lyapunov, through Banach, through ResNet and GPT and diffusion models—and finally come to rest in your hands.

This arc is not a history lesson. This arc is an argument.

---

The argument's core is a single sentence: **Learning, reasoning, and generation are all the motion of the same dynamical system along trajectories in different spaces.**

$S_{t+1} = S_t + \eta F_\theta(S_t, x)$. When $S$ is parameters, it describes training. When $S$ is hidden states, it describes the forward pass of ResNet and GPT. When $S$ is a belief distribution, it describes reasoning. When $S$ is a data point, it describes the reverse process of diffusion models. One equation, four worlds. Not analogy—the same mathematical structure instantiated in four spaces.

You don't need to memorize this equation. You already understand it. From the wilderness hiker of Chapter 1, to Lyapunov in Chapter 6, to the cave diver in Chapter 7, to diffusion in Chapter 12—in every story you have seen the same motion: along the slope beneath your feet, step by step, toward the fixed point.

---

But this book is not only about mathematics.

In the preface, I told the story of Yonglin. A friend who pulled me back to myself through companionship—and I wrote his name into thought through a theorem. Companionship matters more than proof—but proof itself can also be a form of companionship.

On some late night while reading this book, perhaps in a dorm room, perhaps in a lab, perhaps on a commuter train. You might be a student, struggling between papers and exams. You might be an engineer, exhausted by models whose training won't converge. You might simply be a curious person, wanting to know what "learning" is really about.

I hope this book has walked with you for a stretch of road. I hope that when you close it, what you see is no longer a pile of formulas and code—but a wilderness. The model is the hiker. The terrain is the loss function. The step size is the learning rate. Inertia is momentum. The canyon is ResNet. The river is GPT. The fixed point is DEQ. The reasoning field is the gravitational net that every problem casts across belief space.

---

**Force lets you calculate. Energy lets you understand. Geometry lets you see.**

Newton gave force. Hamilton gave energy. Riemann gave curved space. Lyapunov gave the insight that you can know convergence without watching until the end. Banach gave the existence and uniqueness of fixed points. And we—the people of this era—have used this entire language to redescribe learning, reasoning, and generation.

But "seeing" is not only a matter of mathematics. Seeing is lifting your eyes from a formula, looking out the window—and realizing that the wilderness outside, and the energy terrain you just derived on paper, are the same.

The terrain of learning unfolds beneath your feet. It has always been there. You just needed someone to tell you—**look up.**

---

*Zixi Li (Pallas's Cat Professor)*
*2026, Sun Yat-sen University*

---

## Geometric Concept Quick Reference

| Geometric Concept | Mathematical Object | Machine Learning Counterpart |
|---------|---------|------------|
| Position | Point $x \in \mathbb{R}^N$ | Parameters $\theta$, hidden state $h$, belief distribution $p$ |
| Terrain | Scalar function $L: \mathbb{R}^N \to \mathbb{R}$ | Loss function, negative entropy function |
| Slope direction | Gradient $\nabla L$ | Parameter update direction, score function |
| Step size | $\eta$ | Learning rate, reasoning step size |
| Discrete motion | Euler method $x_{t+1} = x_t + \eta F(x_t)$ | Gradient descent, residual connection, CoT step |
| Continuous motion | Gradient flow $\frac{dx}{dt} = F(x)$ | Continuous-depth models, probability flow ODE |
| Curvature | Hessian $H_{ij} = \frac{\partial^2 L}{\partial x_i \partial x_j}$ | Local curvature of loss terrain |
| Critical point | $\nabla L(x^*) = 0$ | Minimum, maximum, saddle point |
| Minimum | $H \succ 0$ and $\nabla L = 0$ | Training convergence point |
| Saddle point | $H$ has both positive and negative eigenvalues | Training bottleneck |
| Flatness | Magnitude of Hessian eigenvalues | Generalization ability |
| Non-Euclidean distance | Bregman divergence $D_F(p\|q)$ | KL divergence |
| Belief space metric | Fisher information matrix $G(p)$ | Natural gradient |
| Energy descent | Lyapunov function $V(x) \searrow$ | Loss descent, KL convergence |
| Fixed point | $F(x^*) = 0$ or $T(x^*) = x^*$ | DEQ output, belief solidification |
| Stability | Sign of real parts of Jacobian eigenvalues | Basin depth and width |
| Attractor | Asymptotically stable fixed point | Correct answer basin |
| Basin of attraction | Set of initial points converging to the same fixed point | Reasoning robustness |
| Bifurcation | Qualitative behavior change caused by parameter variation | Critical learning rate, emergent ability |
| Vector field | $F: \mathcal{M} \to T\mathcal{M}$ | Reasoning field $F_x$, score field |
| Contraction mapping | $\text{Lip}(T) < 1$ | Banach convergence guarantee |
| Diffusion | Stochastic differential equation | Forward noise injection |
| Reverse diffusion | Reverse SDE / probability flow ODE | Generation process |
| Data manifold | Low-dimensional submanifold | Structure of natural data distribution |

---

## Formula Family of the Book

The core thesis of the book can be captured in six formulas. They are not six independent formulas—they are six faces of the same formula in different spaces.

**Core dynamical system:**
$$S_{t+1} = S_t + \eta F_\theta(S_t, x)$$

**ResNet = Explicit Euler:**
$$h_{l+1} = h_l + f_\theta(h_l)$$

**DEQ = Fixed point:**
$$h^* = f_\theta(h^*, x)$$

**Bregman divergence (the mother formula of KL):**
$$D_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle$$

**Yonglin Limit (reasoning convergence criterion):**
$$D_{\mathrm{KL}}(p_{t+1}(y|x) \| p_t(y|x)) < \epsilon$$

**Score function (the vector field of diffusion):**
$$dx = \left[-\frac{1}{2}\beta(t)\,x - \beta(t)\,\nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\,d\bar{w}$$

---

## Reading Roadmap

- **If you want to quickly grasp the core ideas**: Preface → Volume I Introduction → ch1 → ch3 → ch6 → ch7 → ch12 → Epilogue
- **If you care about optimization and training**: ch3 → ch4 → ch5 → ch6
- **If you care about reasoning**: ch5 → ch7 → ch8 → ch9
- **If you care about architecture design**: ch6 → ch11 → ch12
- **If you have a math background and want to tackle the hardest material**: ch5 → ch6 → ch8 (theorem proof chain)
- **If you're a beginner and need to build geometric intuition first**: ch1 → ch2 → ch3 → ch10
