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
