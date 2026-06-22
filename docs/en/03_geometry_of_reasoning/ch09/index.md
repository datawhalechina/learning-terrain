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

![Rough vs smooth terrain](/figures/ch09_rough_vs_smooth_tikz.svg)

You can feel the numerical meaning of $\kappa$ through two extreme examples. Consider a binary classification problem with the model's current belief distribution $p_t = (0.6, 0.4)$.

On a **smooth terrain** ($\kappa \approx 0.2$): the Hessian eigenvalues are nearly constant along the path — $\lambda_1 \approx 5$, $\lambda_2 \approx 4$. At every step, the safe step size is $\eta_{\max} \approx 0.1$, and the contraction factor $k \approx 0.7$. The KL divergence from $p_0$ to $p^*$ decays to 70% of its previous value each step. Within 10 steps, belief is essentially solidified.

On a **rugged terrain** ($\kappa \approx 1.5$): the Hessian eigenvalues fluctuate violently. Along one segment, $\lambda_1 = 100$ (a certain probability dimension is extremely steep), $\eta_{\max}$ is crushed to $10^{-4}$ — the model can barely take a step. Past this "speed-limit zone," the terrain abruptly flattens — $\lambda_1 = 0.5$, directions blur, and the model wanders across a nearly flat plateau. The same road: the first half it "dares not walk," the second half it "does not know where to walk."

This is the geometric essence of $\kappa$: **it does not measure the average steepness of the terrain — it measures the terrain's "temper."** A smooth terrain has a mild temper; you can use a single step size from start to finish. A rugged terrain is volatile; you need to adjust your stride to its every-meter mood. And Theorem 2 tells us: fixed-step Euler's method on a bad-tempered terrain will eventually go wrong.


*Rough vs smooth terrain. High-frequency ripples trap small models; large models smooth these out. $\kappa$—the roughness coefficient—quantifies this from the Hessian spectrum.*

But the terrain is not immutable. Every training iteration, every gradient update, is reshaping this reasoning landscape.


:::info

**Mr. Pallas's Cat: Training Is Pouring Water onto the Mountain**

There is something worth pausing to consider. Theorem 3 says training flattens the terrain — but how does this happen? Every gradient descent step you take moves your body through parameter space — how does that make the reasoning field $F_x$ smoother?

The answer lies in ch5's $\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$. This formula says: the safe step size is determined by how "crowded" the model is at the simplex edge. Early in training, the model's belief distribution is sharp — extremely certain about some classes, almost completely excluding others. $\min(p_i)$ is tiny, $\eta_{\max}$ is tiny — the terrain in these edge regions is "rugged."

As training progresses, the model sees more data, and its belief distribution becomes more "rounded" — no longer extremely certain, no longer extremely exclusionary. $\min(p_i)$ increases, $\eta_{\max}$ increases — the safe step size in the same direction automatically grows larger.

This is the geometric mechanism of Theorem 3: **training does not physically "bulldoze" the terrain flat — training flattens the terrain by making the model's belief distribution more uniform, indirectly raising the safe-step-size ceiling at every position.** The basin widens not because the Hessian's eigenvalues got larger — but because $\min(p_i)$ around $p^*$ got larger, $\eta_{\max}$ followed, and the model dares to take bigger steps at every position.

So training is not pouring water onto the mountain to make it shorter. Training is lengthening the climber's legs.

:::


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

![Temperature-assisted escape from belief basins](/figures/ch09_temperature_escape_tikz.svg)

*Energy landscape of temperature-assisted escape. Horizontal axis: belief coordinate $p$; vertical axis: energy $E(p)$. Shallow red basin: wrong attractor $p_{\text{wrong}}$ — the basin is shallow ($\Delta E$ small), easily escaped with slight heating. Deep blue basin: correct attractor $p^*$ — the basin is deep ($\Delta E$ large), unshaken even by high temperature. Orange trajectory: belief state starting from the wrong basin, gradually gaining energy under random thermal perturbations at temperature $T$, crossing the saddle (black dot), and ultimately sliding into the bottom of the correct basin under gradient guidance. Escape probability follows the Arrhenius formula: $\mathbb{P}(\text{escape}) \approx 1 - \exp(-(t/t_0) \cdot e^{-\Delta E/T})$ — higher temperature and smaller barriers produce faster escape.*

Temperature is an escape tool—but there is one final question: **how does the model know when to stop?**

:::info

**Mr. Pallas's Cat: Temperature and Honesty**

There is a piece of community folklore: setting temperature to 0 (greedy decoding) at inference time is the "safest" — the model always picks the most probable token and won't "babble."

This advice comes from a surface-level understanding of temperature: temperature = randomness, randomness = uncontrollable. But it completely ignores the existence of the energy terrain.

Setting temperature to 0 is equivalent to locking the belief state onto the gradient direction at the current position. If the current position happens to be at the bottom of an incorrect basin, your belief will be trapped there forever — because there is no stochastic perturbation to help you cross the basin's barrier. Zero temperature is the **prisoner of complacency** — it makes you more correct when you are correct, and makes you unable to correct yourself when you are wrong.

Moderate temperature is different. It gives you a tiny probability — $e^{-\Delta E/T}$ at each step — of crossing the barrier and escaping the wrong basin. Temperature is not "randomness" — temperature is **the minimum energy required to escape error**. It does not need to be large — it only needs to be slightly larger than the barrier of the shallowest incorrect basin.

This is why many experienced practitioners avoid temperature=0 on complex reasoning tasks. Not because they are pursuing "diversity" — but because they are leaving the model an escape route from wrong basins.

This criterion's cleverness lies in its **self-reflexivity**. It does not compare the model's belief against "the correct answer" — because during inference there is no correct answer to compare against. It only compares the model's belief today against its belief yesterday. If the belief hasn't changed for several consecutive days — it is either frozen, or dead. And Theorem 6 proves: when contractivity holds, it must be the former.

The choice of $\epsilon$ is a precision-efficiency tradeoff. $\epsilon$ too large — belief isn't frozen yet but you declare it frozen; reasoning terminates prematurely; the answer may be inaccurate. $\epsilon$ too small — belief has already frozen but you keep it walking; computation is wasted. Theorem 6's second clause gives you an error bound: if $\epsilon = 10^{-4}$ and $k = 0.7$, then the KL divergence from the freezing point to the true fixed point does not exceed $10^{-4} / 0.3 \approx 3.3 \times 10^{-4}$ — for the vast majority of applications, this precision already far exceeds the inter-annotator agreement level of human labelers.

The deepest practical insight may lie in Theorem 6's converse: **when the KL freezing criterion is NOT satisfied, what is the model telling you?** If $\Delta_t$ still hasn't converged after dozens of steps — not slowly decreasing, but oscillating or flatlining — that means contractivity does not hold in that region. The reasoning operator $\Phi_\eta$ is not contractive near the current belief. This could be because: $p_t$ is stuck on the boundary between two attractors (near a saddle), or has entered a region where contractivity has been destroyed ($k \geq 1$). In either case, the simplest intervention strategy is: **inject temperature.** Let stochastic perturbation nudge the belief out of this "stagnation zone," and let it restart contraction descent elsewhere.

Temperature is honesty. It admits your model may walk into the wrong basin. It provides a mechanism to walk out. Zero temperature is arrogance — it assumes your model will never step wrong.

---

:::info

**Mr. Pallas's Cat: The Closing of Volume III**

Volume III, from the first word of ch7, has been doing one thing: **dragging you from the water's surface down into the depths.**

You first saw chain of thought — that string of text you thought was "reasoning." We said: that is the shadow. The real reasoning is beneath your feet — in the several-hundred-dimensional hidden state space, a continuous curve pushed along by an invisible vector field.

Then you saw the field — the reasoning field $F_x$. It is a wilderness, with basins and traps. The correct answer has a broad basin in that wilderness — not because it is correct, but because the training data dug deep there. Wrong answers also have basins — far shallower, but deep enough to capture a belief that wanders in from the wrong direction. Verifiers fill in those traps. RLHF redistributes the energy. But you cannot create new basins — only reshape existing ones.

Finally, you learned that the length of reasoning is not up to you — it is determined by the terrain. Theorem 1 tells you the minimum number of steps required. Theorem 3 tells you how training shortens that road. Theorem 4 tells you that larger models make the road flatter. Theorem 5 tells you that temperature is an escape tool. Theorem 6 tells you that when belief solidifies — you can stop.

These three chapters together give a complete answer — not about "how models reason," but about "what the structure of reasoning is." Reasoning is not symbol manipulation. Reasoning is not logical deduction. Reasoning is not token generation.

**Reasoning is the flow of belief along a vector field on an energy terrain, until it reaches a fixed point.**

If you take away only one sentence, take this one. Not because it is precise — but because it draws you a map. The next time you see a model output a string of tokens, I hope you see not text. I hope you see the wilderness beneath its feet, the fixed points around it, the basin it is sliding toward — and all the ruggedness of that terrain.

You have learned to dive. Now, when you lift your head and look at the surface, you will no longer mistake the ripples for the river.

:::

:::


## 9.6 Belief Fixed Points: The Geometric Stopping Criterion for Reasoning

**Definition 3 (KL Freezing Criterion)**. At each step $t$ of the reasoning process, compute the KL divergence between the belief distributions of two consecutive steps: $\Delta_t = D_{\mathrm{KL}}(p_t \| p_{t-1})$. If $\Delta_t < \epsilon$ holds for $m$ consecutive steps, then the model's belief is judged to have **frozen**—reasoning terminates.

**Theorem 6 (Reliability of the Freezing Criterion)**. Suppose the reasoning operator is contractive ($k < 1$). Then:
1. $\Delta_t \to 0$—freezing necessarily occurs.
2. The KL divergence from the freezing point to the true fixed point does not exceed $\epsilon/(1-k)$.
3. If $\epsilon$ is chosen sufficiently small, the freezing point $p_T$ is necessarily a good approximation of $p^*$.

**Practical significance**: The KL freezing criterion transforms "when should reasoning end" from a manually set maximum token count into a problem automatically decided by the local geometry of belief space. Simple problems—KL drops rapidly, freezing early. Complex problems—KL drops slowly, requiring more steps. You no longer need to ask "how many tokens should be generated?"—you only need to ask "has belief frozen?"

![KL solidification curve: belief divergence decays exponentially with reasoning steps](/figures/ch09_kl_solidification_tikz.svg)

*Time evolution of the KL solidification criterion. Horizontal axis: reasoning step $t$; vertical axis: KL divergence between successive belief distributions $\Delta_t = D_{\mathrm{KL}}(p_t\|p_{t-1})$. Blue curve: $\Delta_t \sim k^t \Delta_0$ — the contraction factor $k<1$ guarantees exponential decay. Red dashed line: preset threshold $\epsilon$. When $\Delta_t$ first crosses below $\epsilon$ and remains there for $m$ consecutive steps (green shaded region), belief has solidified — reasoning terminates at $T_{\text{stop}}$ steps. Stronger contraction (smaller $k$) yields faster decay and smaller $T_{\text{stop}}$. The choice of $\epsilon$ determines solidification precision — smaller $\epsilon$ gives more precise solidification but requires more steps.*

---

## 9.7 Chapter Summary

The length of reasoning is not determined by you—it is determined by the terrain. Theorem 1 provides a geometric lower bound on the number of steps: $T_{\min}$ is jointly determined by the initial KL divergence, local curvature, and step-size constraint. Ruggedness $\kappa$ (Theorem 2) quantifies the non-uniformity of the path. Training reshapes the landscape (Theorem 3): basins widen, spurious attractors are eliminated, ridges are smoothed. Model scale smooths the terrain (Theorem 4)—emergent abilities are essentially a path becoming continuous and traversable when scale crosses a threshold. Temperature injection (Theorem 5) provides a stochastic escape mechanism for reasoning. The KL freezing criterion (Theorem 6) provides a geometric stopping condition for reasoning.

Volume III concludes here. From the projection of chain of thought (ch7), to the structure of the reasoning field (ch8), to the geomorphology of long reasoning (ch9)—we have used the geometric language built across nine chapters to deconstruct "reasoning" from a behavioral phenomenon into a geometric process that can be measured, analyzed, and designed in space.

---

## Open Problems

1. Theorem 1 gives $T_{\min}$ depending logarithmically on $D_0$ (the initial KL divergence) but polynomially sensitive to $\mu, L$, and $\eta_{\max}$. This means **the primary bottleneck on reasoning steps is not "how hard the problem is," but "how low the speed limit is along the path."** Can we experimentally verify this logarithmic-vs-polynomial dependence by analyzing the relationship between CoT length and the model's internal Hessian spectrum on large-scale reasoning datasets (MATH, GSM8K)?

2. The roughness $\kappa$ (Definition 1) is computed along the path $p_0 \to p^*$—but this path is unknown before reasoning completes. **Can we estimate $\kappa$ before reasoning begins?** For example, purely from the semantic embedding of the problem $x$ and the model's initial belief $p_0$—without actually traversing the full trajectory—can we predict the roughness of this path?

3. Theorem 3 (landscape reshaping by training) claims three effects—basin widening, false attractor elimination, ridge smoothing—occur monotonically during training. But does "monotonic" strictly hold? Is it possible that training involves **temporary "terrain deterioration"**—for example, during phases where the learning rate is too large and causes oscillations, the reasoning field $F_x$ temporarily becomes more rugged rather than smoother?

4. Theorem 4 (scale-smoothness hypothesis) asserts $\kappa^{(N)} = O(N^{-\alpha})$—roughness decays as a power law with model scale. What is the value of $\alpha$? Does it depend on model architecture (Transformer vs Mamba vs hybrid)? Is there a **critical roughness** $\kappa_{\text{crit}}$ such that when $\kappa^{(N)}$ first drops below it, a specific reasoning capability "emerges" in behavioral tests—and is this $\kappa_{\text{crit}}$ consistent across different tasks?

5. In Theorem 5, the temperature-assisted escape rate $\mathbb{P}(\text{escape}) \approx 1 - \exp(-(t/t_0) \cdot \exp(-\Delta E/T))$ depends on $\Delta E$—the energy barrier of the wrong basin. But what is $\Delta E$ in a real model? Does it equal the KL divergence between the wrong attractor $p_{\text{wrong}}$ and the nearest saddle? Can we estimate $\Delta E$ without actually running reasoning, purely by analyzing the Hessian of $E(p)$ at $p_{\text{wrong}}$?

6. The KL solidification criterion (Theorem 6) uses $\Delta_t = D_{\mathrm{KL}}(p_t\|p_{t-1}) < \epsilon$ for $m$ consecutive steps as the stopping condition. But does there exist a **theoretically optimal choice** for $\epsilon$ and $m$—determined jointly by the contraction factor $k$ and the required precision—rather than empirical tuning? Specifically, given $k$ and the allowable final KL error $\delta$, can we derive an analytic expression for $\epsilon$ and $m$?

7. Temperature injection (Theorem 5) and SGD noise (ch4, §4.7) share the same mathematical soul—both use random perturbations to help the system escape shallow local traps. But temperature operates in belief space, while SGD noise operates in parameter space. **If noise is injected in both spaces simultaneously—temperature during inference, small-batch SGD during training—do their escape effects superpose or cancel?** Is there an optimal "joint temperature–SGD noise" ratio?

8. The landscape reshaping by training (Theorem 3) describes the evolution of $F_x$ for a single problem $x$ during training. But **do the reasoning fields of different problems co-evolve** during training—i.e., do the roughness values $\kappa_x$ of all $F_x$ decrease at approximately the same rate? Or do the terrains of "easy problems" flatten first, while those of "hard problems" suddenly improve only late in training—corresponding to behavioral "unlocking" of different-difficulty tasks at different training stages?

9. If reasoning length is determined by terrain—can we **deliberately "pave roads" in belief space** to shorten reasoning? Specifically, at a certain training stage, we could targetedly increase the coverage density of training data in "regions where $\eta_{\max}$ is extremely small along the path"—a kind of "precision terrain surgery" rather than uniform data augmentation. Would such surgical training interventions be more efficient than uniformly adding more training data?

10. The $T_{\min}$ lower bound of Theorem 1 assumes consistent $\mu$ and $L$ along the entire path. But in actual reasoning, $\mu$ and $L$ are **segment-dependent**—some regions have sudden drops in $\mu$ (terrain flattens, direction unclear) or sudden increases in $L$ (terrain steepens, step size drops sharply). Can the lower bound be generalized to the **piecewise-constant** $\mu$ and $L$ case—giving a tighter $T_{\min}$ estimate that accounts for the local structure of the terrain?

11. Belief solidification (Theorem 6) provides a geometric stopping criterion for reasoning—but this assumes the reasoning operator is contractive ($k < 1$). In non-contractive regions ($k \geq 1$), $\Delta_t$ may oscillate without converging—belief swinging among multiple fixed points. **Can we automatically detect the loss of contractivity when the solidification criterion fails, and trigger intervention**—for example, temporarily injecting temperature to escape the current oscillatory region?

12. The three chapters of Volume III build the complete geometric picture from "reasoning is text" to "reasoning is a trajectory" to "reasoning is flow in a field." But the entire theory rests on one core assumption: **the reasoning field $F_x$ is time-invariant**—it does not change during a single reasoning episode (it depends only on the current belief $p$, not explicitly on the step index $t$). If $F_x$ itself changes during reasoning because of some effect (e.g., the model "learns something new" during long reasoning—test-time adaptation or in-context learning), would the positions of the fixed points also drift? This would elevate "dynamics of the reasoning field" to "**dynamics of the field of the reasoning field itself**"—a meta-level dynamical system.

---
**The core question left by this chapter is:**

**If the length of reasoning is determined by the terrain of belief space—then can we use "terrain surgery" (targeted training interventions) to precisely shorten the reasoning length for specific types of problems, without affecting performance on other problems?**

:::info

**The terrain determines the pace.** You do not choose how many steps reasoning requires—the ruggedness of the ridge chooses for you. You do not judge when to stop—the solidification of belief judges for you. Temperature is the escape tool. Scale is the smoother. Training is the terrain-engineering project. Volume III concludes here. After the geometric journey through twelve chapters, there remains one final stop—Volume IV: The Geomorphology of Algorithms. There, we will use this entire language to re-examine the classic algorithms you thought you understood, and discover that they have never been so clear.

:::

## Further Reading and Related Work

The core themes of reasoning geomorphology—length determined by terrain geometry, ruggedness quantification, temperature as escape tool, KL freezing criterion—find deep resonance in the following works:

1. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** DeepSeek-AI (2025). Nature. [arXiv:2501.12948] — Emergence of reasoning ability and test-time compute scaling. From our perspective, R1's "aha moment" corresponds to the trajectory passing through a critical saddle point in rugged terrain—emergence is not magic, but a path becoming continuously traversable when parameters or data cross a critical threshold. Theorem 4 (scale smooths terrain) provides a direct geometric explanation for this emergence.

2. **A Sharp Convergence Theory for The Probability Flow ODEs of Diffusion Models.** Li et al. (2024). [arXiv:2408.02320] — The $d/\epsilon$ iteration convergence theory for probability flow ODEs. From our perspective, the convergence rate of diffusion model probability flow ODEs and the convergence rate of chain of thought in this chapter share the same mathematical structure—both are necessary consequences of contraction mappings in probability space. The $d$ (data dimension) in diffusion models corresponds to the complexity of the initial belief distribution in reasoning—both determine the minimum number of iterations required to reach a given precision $\epsilon$.

3. **Score-Based Generative Modeling through Stochastic Differential Equations.** Song et al. (2021). ICLR 2021 Oral. [arXiv:2011.13456] — Score-based SDEs unify diffusion and generation. From our perspective, the reverse diffusion SDE is precisely the continuous-time version of our "temperature-assisted escape"—noise amplitude $T$ plays the same role in both generation and reasoning: a stochastic driving force for escape from shallow basins. The Arrhenius escape rate formula is identical in both cases—hinting at a deeper unity between generation and reasoning that goes beyond surface analogies.

4. **Stochastic Gradient Descent as Approximate Bayesian Inference.** Mandt, Hoffman & Blei (2017). JMLR 2017. [arXiv:1704.04289] — The stationary distribution of SGD. From our perspective, temperature in reasoning and the learning rate in SGD are both terrain parameters controlling "exploration vs. exploitation," merely acting in different spaces (belief space vs. parameter space). This parallel suggests a deeper symmetry: training dynamics and reasoning dynamics may be projections of the same terrain framework onto two different spaces.

5. **Scaling Laws for Neural Language Models.** Kaplan et al. (2020). [arXiv:2001.08361] — Power-law scaling of language model loss. From our perspective: larger models = smoother energy landscape = shorter reasoning paths. This is direct evidence for Theorem 4 (the scale-smoothness hypothesis): each order-of-magnitude increase in parameter count reduces reasoning field ruggedness, decreases the contraction factor $k$, and shortens the required reasoning steps $T_{\min}$.
