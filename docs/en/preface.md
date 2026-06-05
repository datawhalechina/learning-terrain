# Preface: Seeing the Terrain

Every formula you have ever learned is a contour line of a landscape.

$F = ma$ is a straight line Newton drew on paper — force equals mass times acceleration, the world a set of vectors to be analyzed one by one. $\hat{\beta} = (X^TX)^{-1}X^Ty$ is the coordinate of the bottom of an elliptical bowl — you always thought it was an algebraic result, but it is a geometric fact: the data carves a perfect quadratic surface in parameter space, and you found its lowest point. $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ is a step taken by a hiker in the wilderness — the slope underfoot tells him the direction, the learning rate determines how far he walks.

You cannot see these terrains — not because they do not exist, but because you have been reading them in algebra.

Algebra lets you compute fast. But algebra does not let you see. To see, you need geometry.

---

This book comes from a simple observation: **the deep learning community has spent too much time counting forces, and too little time seeing the terrain.**

We call the loss function "loss" — a number to be minimized. We call gradient descent "optimization" — as if it were merely a numerical algorithm. We discuss learning rates, momentum, AdamW — as if they were just tuning tricks. We write chain-of-thought — as if it were just a sequence of tokens. We train diffusion models — as if they were just denoising processes.

But if you lift your eyes from the formulas, you will see something entirely different.

The loss function is not a number — it is an energy terrain, with peaks and valleys, saddles and basins, cliffs and plateaus. The gradient is not a vector — it is the direction of the steepest slope underfoot. Training is not hyperparameter tuning — it is a hiker walking step by step through this wilderness. Chain-of-thought is not text — it is a trajectory of hidden states in belief space, the tokens mere ripples cast upward from depths you cannot see. A diffusion model is not a denoiser — it is an inverse heat equation in data space, the score function a vector field, generation the convergence of trajectories onto the data manifold.

This terrain has always been there. You have simply never drawn its map.

This book is that map.

---

The writing of this book has a hidden personal thread.

In November 2025, in my dormitory at Sun Yat-sen University, I finished a paper — on the convergence of reasoning. The core of the paper was an inequality: in belief space, if the step size of an Euler update does not exceed an upper bound determined by the local KL curvature, then reasoning necessarily converges to a unique fixed point. The proof of this inequality rests on three mathematical facts — the generalized Pythagorean theorem of Bregman divergence, the strong convexity and smoothness of the energy function, and Banach's contraction mapping principle — each leading into the next, converging on the same conclusion.

I gave it a name: **the Yonglin Limit.**

Yonglin is my friend. On many nights before that, when I was sinking deeper and deeper into my own thoughts, he simply sat beside me. He did not offer advice. He did not give methods. He did not try to "fix" me. He was just there. His presence pulled me back to myself — and I wrote his name into the ideas with a theorem. Companionship matters more than proof — but proof itself can also be a form of companionship.

The night I finished that paper, I realized something: if I could extract the geometric skeleton of that proof — Bregman, KL, contraction mapping, fixed point — and use the same language to describe loss functions, optimizers, ResNet, GPT, chain-of-thought, diffusion models… then I could write not just a paper. I could write a book.

This book is the product of that thought.

---

The structure of this book is simple. Four volumes, twelve chapters.

**Volume I** establishes geometric intuition. Why does learning need geometry? Where is the model's body — and how does its vision form? How does a loss function become an energy terrain? How does a wilderness hiker walk among slopes, step sizes, and momentum?

**Volume II** enters the heart of formalism. Optimizers are different ways of walking. Bregman divergence and KL divergence — in belief space, distance is not Euclidean. The Yonglin Limit: under what conditions does reasoning necessarily converge? Dynamical systems and fixed points — ResNet, GPT, DEQ are all different faces of the same dynamical system.

**Volume III** applies this language to reasoning. Chain-of-thought is not reasoning itself — it is the projection of a hidden-state trajectory onto text space. The reasoning field: every question lays down a gravitational web in belief space. The geomorphology of long reasoning: why do some questions need two steps and others two hundred?

**Volume IV** returns to the classics, to see everything anew. Linear regression, PCA, SVM — replacing algebra with geometry. Attention, LoRA — the geometric design language of deep architectures. Diffusion models — how noise becomes structure, where the entire book converges.

---

This book does not ask you to abandon algebra. Algebra lets you compute fast. But algebra does not let you see.

This book asks you to learn geometry — not for exams, not for publishing papers, not for sounding smarter in technical discussions.

It asks you to learn geometry so that the next time you write `optimizer.step()`, you see a wilderness beneath your feet. Your model stands somewhere on that terrain. The loss function has carved mountains and basins across it. The gradient tells it the slope underfoot. The learning rate is the step it takes. The optimizer you chose — SGD, Momentum, AdamW — is the way it walks. It is not "tuning hyperparameters." It is walking, step by step, across an energy terrain, toward the place it was always meant to go.

**Force lets you compute. Energy lets you understand. Geometry lets you see.**

---

*Li Zixi (Mr. Pallas's Cat)*
*2026, Sun Yat-sen University*
