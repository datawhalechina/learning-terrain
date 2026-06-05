# Chapter 9: Long Reasoning and Landscape Reshaping

Not all problems are born equal. Some problems—"What is the capital of France?"—have initial beliefs that slide into the basin of the correct answer after just one or two Euler iterations. Other problems—"Prove that $\sqrt{2}$ is irrational"—have initial beliefs that stand on a rugged plateau, surrounded by scattered false attractors, with the only path to the correct answer being a winding, narrow ridge.

If reasoning is a mountain road, then some problems are Highway G318—broad and smooth, you press the accelerator and coast; all you need to do is stay awake. Other problems are the Mêdog Highway—gravel on a cliffside, every step must be precise, one misstep sends you into a wrong basin, and there is no room to turn around.

You cannot make the Mêdog Highway shorter. The road is what it is—the belief-space distance from initial belief to correct answer is determined by the problem itself. But what you can do is **lay asphalt**. Tamp down the gravel. Widen the cliff-edge shoulder. Bulldoze the steep grades that make drivers afraid to accelerate—not to shorten the road, but to make each step larger, to lower the chance of a misstep.

This is the theme of this chapter: the length of reasoning is not determined by you—it is determined by the terrain. But the terrain can be reshaped.

In the previous two chapters, we discussed the trajectories of chain of thought (ch7) and the reasoning field that guides these trajectories (ch8). But one question remains unresolved: **why do some problems require two steps while others require two hundred?** The answer cannot simply be "the problem is harder"—we need a precise geometric theory that connects "reasoning length" to the terrain features of belief space.

This chapter is the final chapter of Volume III. It welds together the "emergence depth" of ch6 and the reasoning field perspective of ch7/ch8, forming a complete geometric understanding of reasoning length. We first derive a geometric lower bound on the number of reasoning steps—three factors determine the minimum number of steps required. We then analyze how training reshapes this landscape—from rugged to smooth, from multi-basin to single-basin. Finally, how temperature helps escape, and how belief freezing provides a stopping criterion.

## 9.1 Geometric Lower Bound on Reasoning Length

Why can't we reach the answer in a single step? Because the Euler step size is bounded—constrained by $\eta_{\max}(p)$ (ch5, §5.10). In certain regions of belief space, the safe step size is compressed to be extremely small—the model can only advance in tiny steps, no matter how "smart" it is.

**Theorem 1 (Geometric Lower Bound on Reasoning Length)**. Let the model start from initial belief $p_0$ and iterate along Euler steps in the reasoning field $F_x$, with step size adaptively adjusted at each step ($\eta \leq \eta_{\max}(p_t)$). Let $p^*$ be the target attractor, and $D_0 = D_{\mathrm{KL}}(p^*\|p_0)$ be the initial KL divergence. If $F_x$ is locally $\mu$-strongly convex and $L$-smooth along the path, then the minimum number of steps required to reach the $\epsilon$-neighborhood of $p^*$ satisfies:

$$T_{\min} \geq \frac{\log(\epsilon / D_0)}{\log(1 - \eta(2\mu - \eta L^2)/C)}$$

**Derivation**. From the contraction inequality of ch5: each step compresses the KL divergence by at least a factor of $k = 1 - \eta(2\mu - \eta L^2)/C$. After $T$ steps, $D_{\mathrm{KL}}(p^*\|p_T) \leq k^T D_0$. Requiring this to be $< \epsilon$ and taking logarithms yields the result.

Three geometric factors jointly determine $T_{\min}$:

1. **Initial distance $D_0$**: The more complex the problem, the larger $D_0$—but the logarithmic relationship means that a 10× increase in $D_0$ only increases $T_{\min}$ by about $-\log_{1/k}(10)$ steps. The exponential convergence rate of the contraction mapping ensures that "more complex" does not cause an explosive increase in step count.

2. **Local curvature $\mu, L$**: The smaller $\mu$ (flatter terrain), and the larger $L$ (more rugged terrain), the closer $k$ is to 1, and the larger $T_{\min}$. In the limit as $\eta(2\mu - \eta L^2) \to 0$, contraction vanishes—the model can never reach the correct answer.

3. **Step-size constraint $\eta_{\max}$**: At the edges of the simplex (where some $p_i$ is extremely small), $\eta_{\max}$ shrinks dramatically—even when the terrain is flat, the model can only advance in tiny steps.

:::info

**Pallas's Cat Professor: Complexity Does Not Explode**

There is a counterintuitive conclusion here worth pausing to savor. $T_{\min}$ depends **logarithmically** on the initial distance $D_0$—make the problem ten times more complex, and the minimum number of reasoning steps increases by only a constant.

What does this mean? It means the primary bottleneck on reasoning step count is not "how hard is this problem," but "how large a step can be taken on this road." $T_{\min}$ depends logarithmically on $D_0$—the initial distance. But it is **polynomially sensitive** to the step-size constraint and curvature. Make the terrain slightly more rugged, shrink $\eta_{\max}$ slightly, push $k$ slightly closer to 1—and the step count can multiply.

This explains why "What is the capital of France" and "Prove $\sqrt{2}$ is irrational" can differ by orders of magnitude in step count, even though their initial distances $D_0$ in belief space might differ by only a few times. Not because the second problem is "tens of times harder"—but because on the path through belief space to the correct answer, there is a region where $\eta_{\max}$ is crushed extremely low, and the model can only inch forward. **It's not that the road is longer. It's that on the same length of road, the speed limit on certain segments has been lowered to walking pace.**

And as we will see—training, scale, temperature—all of these techniques are, at bottom, raising the speed limit.
:::

With this lower bound, we know that reasoning length is determined by three terrain features. But what exactly does "ruggedness" of the terrain mean—can we give it a numerical measure?

## 9.2 Spectral Characterization of Terrain Ruggedness

**Definition 1 (Terrain Ruggedness)**. Given a reasoning field $F_x$, the **ruggedness** along the path from $p_0$ to $p^*$ is the coefficient of variation of the Hessian eigenvalues along the path:

$$\kappa(p_0 \to p^*) = \frac{\text{std}(\{\lambda_i(p_t)\}_{i,t})}{\text{mean}(\{\lambda_i(p_t)\}_{i,t})}$$

A large $\kappa$ means the terrain is non-uniform—some regions are extremely steep (requiring small step sizes), while others are extremely flat (direction unclear). A small $\kappa$ means the terrain is uniform—the model can advance rapidly with consistently large step sizes. $\kappa$ transforms "is this reasoning hard?" from an intuition into a numerical value that can be read off the Hessian eigenvalue spectrum.

**Theorem 2 (Association Between Ruggedness and Reasoning Failure)**. For Euler iteration with a fixed step size $\eta$, the probability of reasoning failure increases monotonically with $\kappa$. A rugged terrain means there are "steep slopes" along the path—a fixed $\eta$ may exceed the local $\eta_{\max}$, causing the system to diverge or jump into a wrong basin.

But the terrain is not immutable. Every training iteration, every gradient update, is reshaping this reasoning landscape.

## 9.3 The Landscape Reshaping Theorem of Training

**Theorem 3 (Landscape Reshaping by Training)**. Let $E_t(p)$ be the energy function at training step $t$. As $t \to \infty$, in the neighborhood of the correct answer $p^*$, three effects occur:

1. **Basin widening**: The eigenvalues of the Hessian $H_t = \nabla^2 E_t(p^*)$ decrease overall—$\|H_t\|_F \to \|H^*\|_F$ approaching from above. The correct attractor transforms from a sharp minimum into a broad basin.
2. **Elimination of spurious attractors**: For any fixed point $p_{\text{wrong}}$ that is not the correct answer, its basin volume $|\mathcal{B}_t| \to 0$, or the fixed point transforms from an attractor into a saddle point (its Jacobian develops positive eigenvalues).
3. **Ridge smoothing**: The path ruggedness $\kappa_t(p_0 \to p^*) \to 0$—the path from the initial belief to the correct answer gradually becomes uniformly flat.

**Corollary**: The number of steps $T_{\min}^{(t)}$ required for reasoning decreases monotonically with training, approaching the theoretical minimum $T_{\min}^{(\infty)}$.

These three effects explain a phenomenon we repeatedly observe during training: as training progresses, the model's CoT length on complex reasoning tasks systematically shortens. Not because the model has become "faster"—but because the terrain has become "flatter."

But model scale itself also affects terrain smoothness. Larger models are not just "more parameters"—they possess the ability to sculpt the reasoning field at finer scales.

## 9.4 Model Scale and Landscape Smoothness

**Theorem 4 (Scale-Smoothness Hypothesis)**. Let the model have $N$ parameters. After training converges, the path ruggedness $\kappa^{(N)}$ of the reasoning field decreases as $N$ increases:

$$\kappa^{(N)} = O(N^{-\alpha})$$

for some $\alpha > 0$ (dependent on architecture and training data). On small models, due to limited expressive capacity, certain regions of $F_x$ "have to" have steep gradient variations—$\kappa$ is large. Large models have sufficient degrees of freedom for fine-grained control—these regions are polished smooth, and $\kappa$ is small.

**Geometric explanation of emergent abilities**: For a given reasoning task, suppose there exists a continuous path from $p_0$ to $p^*$, but a segment of it has extremely high ruggedness—the $F_x$ of a small model is discontinuous in this region or fails to provide effective guidance. When the model scale $N$ crosses a threshold $N_{\text{crit}}$, $\kappa^{(N)}$ drops below a certain critical value for the first time—the entire path becomes traversable. From the perspective of behavioral testing, the model has "suddenly acquired" this reasoning ability. From the geometric perspective, this is not sudden—it is a path that has always existed, gradually becoming smooth and continuous as the parameter-space dimension grows.

:::info

**Pallas's Cat Professor: Emergence Is Not Magic**

The AI community has spilled a lot of ink on the word "emergence." From a behavioral perspective, some abilities do appear to pop into existence at certain scale thresholds—as if the model "had an epiphany." This is unsettling, because it hints at some kind of unpredictable, uncontrollable phase transition.

But Theorem 4 offers a much calmer picture. Emergence is not magic—it is **the recovery of path continuity when the parameter dimension crosses a critical value.**

Imagine a mountain road with a crack in the middle. When the model is small, the crack is two meters wide—uncrossable. As the model grows ($N$ increases), the crack gradually narrows—$O(N^{-\alpha})$. When $N$ exceeds a certain threshold, the crack becomes narrow enough to step across. From the behavioral-testing perspective, the model "suddenly acquired reasoning." But from the geometric perspective, the crack has been narrowing all along—every additional row of parameters shaves a few more millimeters off its width. It's just that you plotted the capability curve in linear coordinates and missed the narrowing process entirely.

Emergence is an illusion produced by the coarse discretization of behavioral testing. In the differential geometry of belief space, $d\kappa/dN$ is never infinite.
:::

Terrain, training, scale—the three jointly determine the shape of the reasoning field. But there is one more factor we have not yet considered: **randomness**. Temperature—stochastic noise at inference time—what can it do?

## 9.5 Stochastic Dynamics of Temperature Injection

**Definition 2 (Temperature-Perturbed Euler Step)**. At temperature $T > 0$, a step in the reasoning field is not a deterministic vector-field following, but a stochastic step with Gaussian noise:

$$p_{t+1} = \text{proj}_{\Delta}\left(p_t + \eta F_x(p_t) + \sqrt{2\eta T} \cdot \xi_t\right)$$

where $\xi_t \sim \mathcal{N}(0, I)$, and $\sqrt{2\eta T}$ is the noise amplitude.

**Theorem 5 (Temperature-Assisted Basin Escape)**. Suppose the model's current belief $p_t$ falls within the neighborhood of a false attractor $p_{\text{wrong}}$. At temperature $T$, the escape probability is given by an Arrhenius-type formula:

$$\mathbb{P}(\text{escape}) \approx 1 - \exp\left(-\frac{t}{t_0} \cdot \exp\left(-\frac{\Delta E}{T}\right)\right)$$

where $\Delta E = E(p_{\text{saddle}}) - E(p_{\text{wrong}})$ is the energy barrier that must be crossed to escape. The larger the temperature $T$, the higher the escape probability. The smaller $\Delta E$ (the shallower the false basin), the easier it is to escape. This is completely parallel to the mechanism in ch4 where SGD noise preferentially selects flat minima—temperature plays the same role in reasoning that SGD noise plays in training: **a terrain filter that pushes the model out of shallow, narrow false basins and retains it in deep, wide correct basins.**

It is like cooking a pot of soup. When the temperature is low, the ingredients settle at the bottom—whether or not that is where you want them. Turn up the heat, and convection surges—the shallow ingredients are flipped to the surface. Crank it higher, and even the deepest sediment is stirred up. Temperature does not give you "the correct answer"—temperature gives the entire system enough energy to cross any basin barrier that is not deep enough. Those shallow error basins—formed from accidental co-occurrences in the training data, $p_{\text{wrong}}$ with only a slight energy advantage—melt like spring snow in the face of temperature. But the deep correct basins, supported by mountains of evidence, with $\Delta E$ extremely high—even high temperature cannot shake them.

This is why increasing temperature at inference time does not necessarily hurt accuracy. Moderate temperature is a **belief escape tool**—it rescues trajectories trapped in shallow error basins and gives them a chance to find deeper correct basins. Temperature is not blind noise—in the context of the energy landscape, temperature is selective: it preferentially clears shallow basins and preserves deep ones.

Temperature is an escape tool—but there is one final question: **how does the model know when to stop?**

## 9.6 Belief Fixed Points: The Geometric Stopping Criterion for Reasoning

**Definition 3 (KL Freezing Criterion)**. At each step $t$ of the reasoning process, compute the KL divergence between the belief distributions of two consecutive steps: $\Delta_t = D_{\mathrm{KL}}(p_t \| p_{t-1})$. If $\Delta_t < \epsilon$ holds for $m$ consecutive steps, then the model's belief is judged to have **frozen**—reasoning terminates.

**Theorem 6 (Reliability of the Freezing Criterion)**. Suppose the reasoning operator is contractive ($k < 1$). Then:
1. $\Delta_t \to 0$—freezing necessarily occurs.
2. The KL divergence from the freezing point to the true fixed point does not exceed $\epsilon/(1-k)$.
3. If $\epsilon$ is chosen sufficiently small, the freezing point $p_T$ is necessarily a good approximation of $p^*$.

**Practical significance**: The KL freezing criterion transforms "when should reasoning end" from a manually set maximum token count into a problem automatically decided by the local geometry of belief space. Simple problems—KL drops rapidly, freezing early. Complex problems—KL drops slowly, requiring more steps. You no longer need to ask "how many tokens should be generated?"—you only need to ask "has belief frozen?"

---

## 9.7 Chapter Summary

The length of reasoning is not determined by you—it is determined by the terrain. Theorem 1 provides a geometric lower bound on the number of steps: $T_{\min}$ is jointly determined by the initial KL divergence, local curvature, and step-size constraint. Ruggedness $\kappa$ (Theorem 2) quantifies the non-uniformity of the path. Training reshapes the landscape (Theorem 3): basins widen, spurious attractors are eliminated, ridges are smoothed. Model scale smooths the terrain (Theorem 4)—emergent abilities are essentially a path becoming continuous and traversable when scale crosses a threshold. Temperature injection (Theorem 5) provides a stochastic escape mechanism for reasoning. The KL freezing criterion (Theorem 6) provides a geometric stopping condition for reasoning.

Volume III concludes here. From the projection of chain of thought (ch7), to the structure of the reasoning field (ch8), to the geomorphology of long reasoning (ch9)—we have used the geometric language built across nine chapters to deconstruct "reasoning" from a behavioral phenomenon into a geometric process that can be measured, analyzed, and designed in space.

---

## Open Problems

**The core question left by this chapter is:**

**If the length of reasoning is determined by the terrain of belief space—then can we use "terrain surgery" (targeted training interventions) to precisely shorten the reasoning length for specific types of problems, without affecting performance on other problems?**

:::info

**The terrain determines the pace.** You do not choose how many steps reasoning requires—the ruggedness of the ridge chooses for you. You do not judge when to stop—the solidification of belief judges for you. Temperature is the escape tool. Scale is the smoother. Training is the terrain-engineering project. Volume III concludes here. After the geometric journey through twelve chapters, there remains one final stop—Volume IV: The Geomorphology of Algorithms. There, we will use this entire language to re-examine the classic algorithms you thought you understood, and discover that they have never been so clear.

:::
