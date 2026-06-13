# Chapter 12: Diffusion and Convergence

1687, Newton wrote down $F = ma$.

1833, Hamilton said: stop counting forces. Look at energy. $H = T + V$. A system flows on the energy landscape, tending toward stability when the Hamiltonian is minimized.

2015, the residual connections of ResNet were discovered to be explicit Euler steps—$h_{l+1} = h_l + f_\theta(h_l)$. Layers are time steps, the network is a trajectory.

2020, DDPM brought diffusion models into the mainstream—$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$. Noise is injected into the data step by step, until the structure is completely dissolved. Then a neural network learns to reverse this process—reconstructing cat photos, human faces, artworks from pure noise.

From $F=ma$ to diffusion models—across this 337-year span, one idea has remained constant: **systems evolve along an energy landscape, from high energy to low energy, from disorder to order, from noise to structure.** Learning is like this. Reasoning is like this. Generation is like this.

This is the final chapter of the book. It places diffusion models into the full geometric language we have built—interpreting them as a dynamical system in data space, the score function as a vector field, and reverse diffusion as Euler-step trajectories along that vector field. Then, in a single page, it draws the complete arc from ch1 to ch12.

![Forward diffusion: data dissolves into noise](/figures/ch12_forward_diffusion.svg)

*Five-step forward diffusion process. t=0: original data (two Gaussian clusters). At each step, Gaussian noise is injected ($\beta_t$ increasing), and structure gradually dissolves. t=4: almost pure Gaussian noise—maximum entropy, zero structure. The forward process is "anti-learning"—it does not create structure, it dissolves structure.*

## 12.1 Forward: From Data to Noise

The forward process of diffusion models is anti-learning—it does not create structure, it dissolves structure.

Given a real image $x_0$, the forward process gradually destroys it with $T$ small noise steps:

$$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

where $\beta_t \in (0, 1)$ is a small noise scheduling parameter. $\sqrt{1-\beta_t}$ preserves a portion of the previous signal, while $\sqrt{\beta_t}$ injects fresh Gaussian noise.

Repeat $T$ times—typically $T = 1000$. At each step, the signal decays a little, and the noise grows a little. $x_T$ is almost pure Gaussian noise—the structure of the original image has been entirely dissolved.

From a geometric perspective, the forward process defines an **entropy-increasing trajectory** in data space. The true data distribution $p_0$ is a complex distribution on a high-dimensional data manifold—highly structured, low-entropy. Each noise injection step pushes probability mass off the manifold, diffusing it into the surrounding empty space. After $T$ steps, $p_T \approx \mathcal{N}(0, I)$—the maximum-entropy distribution, completely disordered.

This is of a piece with the concept of entropy in ch2, the regularizing effect of noise in ch4, and the escape mechanism of temperature injection in ch9. Noise is not the enemy. Noise is a **tool for exploration**.

## 12.2 The Continuous Limit: From Discrete Diffusion to Stochastic Differential Equations

In the continuous-time limit ($T \to \infty$, $\beta_t \to 0$ with appropriate scaling), the discrete forward process becomes a **stochastic differential equation** (SDE):

$$dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$$

Two terms are clearly distinguishable: the first term $-\frac{1}{2}\beta(t)x\,dt$ is the **drift term**—it pulls $x$ toward the origin (mean reversion, signal decay). The second term $\sqrt{\beta(t)}\,dw$ is the **diffusion term**—it injects random noise.

The SDE connects diffusion models to the dynamical systems language of ch6. The forward SDE is a **stochastic dynamical system**—its "trajectory" is not a deterministic curve, but the continuous evolution of a probability distribution. The distribution from initial condition $x_0$ is the data distribution $p_0$. The distribution at time $T$, $p_T$, approaches $\mathcal{N}(0, I)$.

![Score field and reverse diffusion trajectories](/figures/ch12_score_and_trajectories.svg)

*Left: the vector field formed by the score function $\nabla_x \log p(x)$. Arrows point from all locations toward the two data clusters (squares)—the score field is a "gravitational field" in data space, pulling all points toward high-density regions. Right: five reverse diffusion trajectories starting from different noise initial points, flowing along the score field and ultimately converging to the data manifold.*

## 12.3 Reversal: The Score Function as a Vector Field

The magic of diffusion models is: **the forward process is reversible.** If we know the gradient of the data distribution at every moment $t$—that is, the **score function** $\nabla_x \log p_t(x)$—we can start from pure noise, follow the direction indicated by the score function, and walk step by step back to the data manifold.

The reverse SDE is:

$$dx = \left[-\frac{1}{2}\beta(t)\,x - \beta(t)\,\nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\,d\bar{w}$$

Two drift terms: $-\frac{1}{2}\beta(t)x$ pulls $x$ toward the origin (same as in the forward process); $-\beta(t)\nabla_x \log p_t(x)$ pushes $x$ toward high-density regions—**toward the data manifold**.

The score function $\nabla_x \log p_t(x)$ is the geometric heart of diffusion models. It is a **vector field**—at every point in data space, it points in "the direction of more data." Near the data manifold, the score points toward the manifold—pulling back particles that have drifted off the manifold. Far from the data manifold, the score function is (in theory) zero—because where there is no data, there is also no "direction of more data."

In practice, the score function is unknown—$p_t(x)$ is precisely what we deliberately destroyed in the forward process. So we train a neural network $s_\theta(x, t)$ to approximate it. The training objective minimizes:

$$\mathbb{E}_{t, x_0, \epsilon}\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t|x_0)\|^2\right]$$

This is entirely consistent with the spirit of ch3, where a neural network fits the gradient of the loss function: **the model learns not the data itself, but the "force field" of the data—the vector field that pushes noise back toward structure.**

## 12.4 Score as Energy Gradient: From Diffusion Back to Hamilton

There is a deep mathematical connection between the score function and the earliest theme of this book—the Hamiltonian energy of ch1.

The score function is the gradient of the log-density: $\nabla_x \log p(x)$. If we define an **energy function** $E(x) = -\log p(x)$, then the score is the negative energy gradient:

$$\nabla_x \log p(x) = -\nabla_x E(x)$$

This is precisely the core of gradient descent in ch3: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta)$. The gradient of the loss function tells the parameters which direction to move to reduce the loss. The score function tells a data point which direction to move to increase probability density—toward more "real" regions.

Reformulated from the Hamiltonian (ch1) perspective: every point on the data manifold has an energy $E(x) = -\log p(x)$. The high-density regions of the data distribution are low-energy regions—they are the "valleys" of energy. Noise is the "plateau" of energy—no structure, equal energy everywhere. The score function is the **negative gradient** on this energy landscape—pointing in the direction of fastest energy decrease.

**The reverse process of diffusion models is a gradient descent process—in data space, along the negative gradient direction of $E(x) = -\log p(x)$, pushing noise samples step by step toward the low-energy regions on the data manifold.** This is the same geometric operation as parameters sliding down the loss landscape in ch3, and beliefs flowing along the entropy landscape in ch5—just in a different space.

## 12.5 Reverse Diffusion as a Dynamical System

Viewing the deterministic drift of the reverse SDE as a vector field, reverse diffusion is a dynamical system (ch6):

$$F_t(x) = -\frac{1}{2}\beta(t)\,x - \beta(t)\,\nabla_x \log p_t(x)$$

This vector field has two components. $-\frac{1}{2}\beta(t)x$ is a simple linear attractor—it pulls all points toward the origin. $-\beta(t)\nabla_x \log p_t(x)$ is a complex, data-dependent vector field—it pushes along the direction of the score function.

At $t = T$ (pure noise), $\nabla_x \log p_T(x) \approx -x$ (since $p_T \approx \mathcal{N}(0, I)$). $F_T(x)$ merely draws the noise slightly inward. As $t$ decreases, the score function becomes increasingly structured—the data manifold begins to "reveal itself" in the vector field. At $t \approx 0$, $F_0(x)$ forcefully pushes all points toward the data manifold.

The sampling process generates trajectories using Euler steps (or more refined discretization schemes) in this time-varying vector field. Starting from $x_T \sim \mathcal{N}(0, I)$:

$$x_{t-1} = x_t - \eta F_t(x_t) + \sqrt{\eta}\,\xi_t$$

Each step is one time step of the dynamical system in ch6. The noise term $\sqrt{\eta}\,\xi_t$ provides randomness—preventing trajectories from collapsing to a single mode on the data manifold, ensuring generative diversity. This is exactly parallel to the logic in ch9, where temperature injection helps reasoning escape false attractors.

:::info

**Diffusion Models and Optimal Transport: The Schrödinger Bridge**

The forward process (data → noise) and reverse process (noise → data) of diffusion models form a pair of mutually time-reversed stochastic differential equations. From the perspective of probability distributions, this pair of SDEs solves a classic problem: **given two probability distributions—the data distribution $p_0$ and the Gaussian distribution $p_T = \mathcal{N}(0, I)$—what is the "most natural" stochastic process evolving from $p_0$ to $p_T$?**

In 1931, Erwin Schrödinger (yes, the Schrödinger of the cat) posed a seemingly unrelated question: in an ensemble of Brownian particles, if the distribution of particles is observed to be $\mu_0$ at time $t=0$ and $\mu_T$ at time $t=T$, what is the most likely evolution of the particle ensemble between these two moments? This problem is called the **Schrödinger Bridge problem.**

The answer is an entropy-regularized optimal transport problem. In the continuous-time limit, the Schrödinger Bridge is equivalent to solving a coupled system of forward and backward SDEs—the forward SDE pushes $\mu_0$ toward $\mu_T$, and the backward SDE pulls $\mu_T$ back toward $\mu_0$. This is precisely the mathematical structure of diffusion models: the score function $\nabla_x \log p_t(x)$ is the drift term of the backward SDE—i.e., the "optimal control" of the Schrödinger Bridge.

The connection to Optimal Transport completes the picture. Classical optimal transport (the Monge-Kantorovich problem) seeks the minimum-cost scheme for mapping one distribution $P$ to another $Q$—cost typically measured by the Wasserstein distance. The Schrödinger Bridge is **entropy-regularized** optimal transport: it allows particles not to move strictly along deterministic paths, but to explore multiple possible paths with the aid of noise. The strength of entropy regularization is controlled by the diffusion coefficient—the larger the noise, the "fuzzier" the paths.

As the noise tends to zero, the Schrödinger Bridge degenerates to the classical optimal transport map—all probability mass moves along the unique optimal path. This is precisely the "deterministic sampling" limit of diffusion models (e.g., DDIM): after removing the random noise term, the reverse diffusion becomes a deterministic probability flow ODE, each trajectory being a uniquely defined path from Gaussian noise to some point on the data manifold.

This provides a profound unifying perspective for the entire book:

- **In belief space**, Bregman divergence and KL divergence measure the "information distance" between belief distributions, and the Yonglin Limit guarantees that contraction mappings along the KL gradient necessarily converge—this is the **language of information geometry.**
- **In data space**, the Wasserstein distance and the Schrödinger Bridge measure the "geometric distance" between data distributions, and the reverse process of diffusion models guarantees convergence from a noise distribution to the data manifold—this is the **language of optimal transport.**

The two are mathematically dual. Bregman divergence is the "energy difference" on the probability simplex—it measures the irreversible loss of **information.** The Wasserstein distance is the "transport cost" in data space—it measures the cost of moving **mass.** And the Schrödinger Bridge sits precisely at the intersection of the two: it simultaneously moves probability mass through data space while preserving entropy regularization in the information-theoretic sense.

From the Yonglin Limit of ch5 to the diffusion models of ch12—the two great mathematical pillars of this book (Bregman geometry and optimal transport) shake hands on the Schrödinger Bridge.

:::

## 12.6 Generation: Trajectories Converge to the Data Manifold

View the entire reverse diffusion trajectory $\{x_T, x_{T-1}, \ldots, x_0\}$ as one evolution of a dynamical system. It starts from pure noise—maximum entropy, zero structure—and flows step by step along the vector field defined by the score function toward the data manifold.

After $T$ steps, the endpoint $x_0$ of the trajectory lands somewhere on the data manifold—it is a concrete image, a piece of audio, a molecular conformation. Different noise initial values $x_T$ produce different trajectories, converging to different locations on the data manifold—generating different samples.

This is entirely consistent with the logic of belief fixed points in ch6: the sampling process starts from an initial distribution (Gaussian noise), flows along the vector field $F_t$, and ultimately arrives at the data manifold—every point on the manifold is an attractor of $F_0$. The trained diffusion model creates a **reasoning field** (ch8) in data space—its attractors are all the points on the data manifold, its vector field points toward the manifold, and its basins cover the entire Gaussian noise space.

## 12.7 From Newton to Here: The Arc of the Book

This book began with $F = ma$. In the final chapter, we return to the same place—not as a starting point, but as a destination.

Newton told us: forces determine motion. Hamilton told us: energy determines motion—stop counting forces, look at the landscape. ch1 transplanted this idea into learning: the loss function is the energy landscape, gradient descent is sliding down that landscape.

ch2 asked: what is it that moves? Answer: the body moves in parameter space, the field of vision forms in representation space. ch3 opened up the terrain underfoot: loss functions, gradients, Euler steps, minima, saddle points—a map for the wilderness hiker.

ch4 asked: does the way you walk matter? Answer: enormously. Step size, momentum, regularization, implicit regularization—the way you walk determines where you arrive.

ch5 entered the non-Euclidean world: Bregman divergence, KL divergence. In belief space, distance is not Euclidean—it is the energy difference on an entropy landscape. The Yonglin limit proves: flow along the KL gradient necessarily converges. ch6 mapped the entire language of dynamical systems—Lyapunov functions, attractors, bifurcations, emergence depth—onto deep learning architectures: ResNet = Euler method, GPT = hidden-state Euler, DEQ = fixed point.

ch7 revealed: chain of thought is not reasoning itself—it is the discrete projection of hidden-state trajectories into text space. ch8 opened up the reasoning field: vector fields, attractors, basins, verifiers, RLHF. ch9 explained the length of reasoning: the terrain determines how many steps are needed.

ch10 returned to classical algorithms: linear regression, PCA, SVM—re-examining everything through geometry. ch11 entered deep architectures: attention is a curved metric, LoRA is a low-rank parameter subspace.

Finally, ch12—diffusion models. Forward: structure dissolves into noise (entropy increase). Reverse: noise converges to structure (entropy decrease). The score function is a vector field. Reverse diffusion is a dynamical system. Generation is trajectories converging to the data manifold.

**337 years. From $F=ma$ to diffusion models. The same arc: force → energy → landscape → motion → fixed point → structure.**

---

## Book Summary

Learning has a terrain. Reasoning has a path. Intelligence has a mechanics.

This book, from its very first page, has done only one thing: **replace algebra with geometry.** Not to abandon algebra—algebra lets you compute fast. But algebra does not let you see. To see, you need geometry.

A loss function is not a number—it is an energy landscape, with peaks and valleys, saddle points and basins. A gradient is not a vector—it is the steepest downhill direction underfoot. Training is not parameter tuning—it is a model walking step by step through the wilderness. Reasoning is not generating tokens—it is a hidden state flowing along the reasoning field in belief space, until it falls into the basin of some attractor. An architecture is not a stack of layers—it is a trajectory carved through space by a dynamical system.

This geometric language is not decoration. It is a complete alternative—re-describing every core phenomenon in deep learning using space, landscape, motion, and convergence. It does not ask you to abandon mathematics—it asks you to see mathematics.

Force lets you compute. Energy lets you understand. Geometry lets you see.

---

**The core question left by this book is:**

**If all learning—from linear regression to diffusion models—is the flow of a dynamical system along the gradient direction on some energy landscape, then does there exist an "ultimate landscape"—an energy function whose minima happen to correspond to all the structure and knowledge that humans need to understand the world? If it exists, learning is walking toward it. If it does not exist—learning is creating it.**

:::info

In 1687, Newton sat in his room at Cambridge and wrote $F=ma$. He did not know that three hundred years later, a group of people would use the same formula—translated into a different language—to make machines learn to paint, write poetry, and reason.

This book began with $F=ma$—not out of nostalgia, but because this arc is real. From force to energy, from energy to landscape, from landscape to motion, from motion to fixed point—every link is not a metaphor, it is mathematics. The loss function is the Hamiltonian. Gradient descent is the Euler method. Chain of thought is a trajectory. The reasoning field is a vector field. The Yonglin limit is the Banach fixed point. Diffusion models are the reverse heat equation in data space.

When you opened this book, you may have thought deep learning was a pile of code and tuning tricks. When you close this book—I hope you see a wilderness. Models are hikers. The terrain is the loss function. Step size is the learning rate. Inertia is momentum. Canyons are ResNet. Rivers are GPT. Fixed points are DEQ. The reasoning field is the gravitational net cast by every problem in belief space.

You are not training a model. You are placing a ball on an energy landscape, and watching it roll to where it belongs.

The terrain of learning unfolds beneath your feet.

**— Li Zixi (Mr. Pallas's Cat), 2026, Sun Yat-sen University**

:::

## Further Reading and Related Work

This chapter is the terminus of the entire book. From Newton in 1687 to diffusion models in 2020, 337 years of scientific thought are threaded together by a single geometric line. The following works are not isolated papers—they are key stations along this arc.

**Denoising Diffusion Probabilistic Models.** Ho, Jain & Abbeel (2020) [arXiv:2006.11239]——The original DDPM paper, inspired by nonequilibrium thermodynamics, brought diffusion models into the mainstream. It designs the forward process as a fixed noise schedule, allowing the reverse process to be trained with a simple MSE loss. The geometric interpretation of this chapter: forward is entropy increase, reverse diffusion is entropy decrease—noise to structure is the flow of a dynamical system along a score field.

**Score-Based Generative Modeling through Stochastic Differential Equations.** Song et al. (2021) [arXiv:2011.13456]——Unifies diffusion models and score-based models in a single SDE framework. The emergence of the probability flow ODE means: reverse diffusion is not only a stochastic process but also a deterministic dynamical system—equivalent to a Neural ODE (see below). This work fully unifies diffusion with the dynamical systems language of ch6.

**Improved Denoising Diffusion Probabilistic Models.** Nichol & Dhariwal (2021) [arXiv:2102.09672]——Learning the reverse variance reduces sampling steps from 1000 to about 100. Geometric explanation: smoother probability flow ODE trajectories allow larger integration step sizes—a recurrence of the adaptive step-size idea from ch4 in data space.

**A Sharp Convergence Theory for The Probability Flow ODEs of Diffusion Models.** Li et al. (2024) [arXiv:2408.02320]——A $d/\varepsilon$ convergence theory for probability flow ODEs: approximating the ODE trajectory requires $O(d/\varepsilon)$ steps given data dimension $d$ and target error $\varepsilon$. This is the "Yonglin limit" for diffusion models—sharing the exact same mathematical structure as the KL convergence guarantee in ch5 and the chain-of-thought step lower bound in ch7: **convergence requires steps, and the number of steps is determined by the geometry of the terrain.**

**Neural Ordinary Differential Equations.** Chen et al. (2018) [arXiv:1806.07366]——Neural ODE pushes the discrete layers of residual networks to continuous depth. The probability flow ODE is a special case of Neural ODE in diffusion models—the reverse process is a continuous-time dynamical system from noise to data. The loop of the entire book—from Newtonian mechanics (ch1) to dynamical systems (ch6) to diffusion models (ch12)—closes here.

**Language Models are Few-Shot Learners.** Brown et al. (2020) [arXiv:2005.14165]——GPT-3, a 175B-parameter autoregressive language model. In the geometric framework of this chapter, autoregressive generation is the hidden state taking one step along the reasoning field in belief space—each generated token is one step, equivalent to one denoising step in the reverse diffusion process. Diffusion models are generation in continuous space; autoregressive models are generation in discrete space. The two are two sides of the same coin.

**Scaling Laws for Neural Language Models.** Kaplan et al. (2020) [arXiv:2001.08361]——Scaling laws reveal a power-law relationship between model capacity and performance. In the unified geometric language of the entire book: larger model = smoother energy landscape = shorter inference paths = deeper basins = better generalization. Scaling laws are not about parameter count—they are about the smoothness of the terrain and the depth of the basins.
