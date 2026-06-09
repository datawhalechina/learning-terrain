# Chapter 7: Chain of Thought: The Projection of the Reasoning Trajectory

You see a model output a long string of text.

> "First, we need to calculate the total area... then subtract the overlap... therefore the answer is 42."

You think to yourself: it's reasoning. It's thinking step by step, moving from the problem toward the answer. What you see is a string of text—token after token, logic linked ring by ring.

But from a geometric perspective, what you see is only a **shadow**.

The real reasoning is not these words. The real reasoning happens in the model's hidden state space—a set of vectors of several hundred or several thousand dimensions, in places you cannot see, flowing step by step along a vector field jointly shaped by the training data and the model architecture. The tokens you see—"first", "then", "therefore"—are merely a discrete projection of this hidden trajectory into text space.

Chain of thought is not reasoning itself. **Chain of thought is the visible projection of the reasoning trajectory.**

:::info

**Pallas's Cat Professor: Above and Below the Surface**

A friend of mine is a cave diver. He once told me something: in pitch-black underwater caves, the only thing you can see is not the current itself—but the fine sand and bubbles it carries. The sand tells you the water is moving, but the sand is not the water.

The tokens of chain of thought are your sand. The model outputs "first... then... therefore..."—you see the logical sequence, just as the diver in the dark water sees the trail of sand grains. But that trail is only a cross-section of the three-dimensional current, illuminated by chance by your flashlight.

The real reasoning happens much deeper. That several-hundred-dimensional hidden state space is the water—continuous, curved, pushed along by a vector field driven by a single problem. The chain of thought you see is only a cross-section of that surface, lit up momentarily.

That is the central claim of this chapter: **the explicit chain is the shadow; the implicit chain is the body.** And what we are about to do—definitions, theorems, derivations—is not meant to make you forget the surface. It is meant to teach you to dive.

:::

But if the word "projection" remains at the level of metaphor, it explains nothing. Why does CoT training improve the accuracy of direct answers? Why are some tokens critical while others are merely filler? Why does zero-shot CoT work? To answer these questions, we must turn the metaphor into geometry—first rigorously define the two objects, then analyze the relationship between them, and finally provide an explanation via theorems.

## 7.1 Formal Definition of the Two Types of Chains

Before performing any geometric analysis of chain of thought, we must rigorously distinguish two objects that are frequently conflated.

**Definition 1 (Explicit Chain)**. Given a problem $x$, the token sequence $(x_1, x_2, \ldots, x_T)$ generated autoregressively by the model is called the **explicit chain**. Each step $x_t$ is a discrete symbol sampled from the vocabulary $V$:

$$x_t \sim p_\theta(\cdot \mid x, x_1, \ldots, x_{t-1})$$

The explicit chain is visible, readable, and evaluable by humans. It is what "chain of thought" refers to in everyday language.

**Definition 2 (Implicit Chain)**. Given the same problem $x$ and the same model $\theta$, the hidden state sequence $(h_0, h_1, \ldots, h_T)$ is called the **implicit chain**. Here $h_0 = \text{encode}(x)$ is the model's initial hidden state after reading the problem; for $t \geq 1$:

$$h_t = \text{Transformer}(h_{t-1}, x_t)$$

That is, $h_t$ is the output after the previous hidden state $h_{t-1}$ and the newly generated token $x_t$ pass through the Transformer layers. The implicit chain is continuous, high-dimensional, and not directly readable by humans.

**The relationship between the two**. The explicit chain is a **discrete projection** of the implicit chain. At each step $t$, the hidden state $h_t$ is projected onto the vocabulary through a linear map $W_{\text{lm}} \in \mathbb{R}^{|V| \times d}$ and softmax:

$$p_\theta(\cdot \mid x, x_{<t}) = \text{softmax}(W_{\text{lm}} h_t)$$

Then $x_t$ is sampled from this distribution. The projection matrix $W_{\text{lm}}$ compresses a $d$-dimensional continuous vector into a $|V|$-dimensional probability vector—this is an irreversible, lossy operation. A great deal of geometric information in the hidden state is lost in this projection. Just as a three-dimensional curve projected onto two-dimensional paper—you see only a few discrete points, but not the continuous curvature between them.

You may ask: what is the practical significance of rigorously distinguishing these two objects—the explicit chain and the implicit chain? After all, in everyday use, we only care whether the model's output text is logically coherent.

The significance is this: **once you admit there is a current beneath the surface, you will begin to ask about the laws that govern that current.** You cannot ask about something whose existence you do not acknowledge.

Think about it. If chain of thought were truly just a sequence of words, then "training a model to output CoT" and "training a model to output any other text format" should be fundamentally no different—just as training a model to output JSON versus YAML is merely a formatting choice. But experiments repeatedly show otherwise. The effect of CoT training—on mathematical reasoning, logic puzzles, and code generation—does not merely come from "writing a few more tokens." It comes from something deeper, some structural transformation of the hidden state space.

To understand that something, we first need to give it a name. "Implicit chain" is that name.

With these two definitions, a natural follow-up question emerges: **how does the implicit chain itself move?** What laws govern each of its steps—$h_t \to h_{t+1}$?

## 7.2 Dynamics of the Implicit Chain: Euler's Method in Hidden State Space

Placing the implicit chain within the dynamical systems framework of ch6, everything becomes clear. $(h_0, h_1, \ldots, h_T)$ is not a string of independent vectors—it is a **discrete trajectory**. The update at each step precisely takes the form of the explicit Euler method:

$$h_{t+1} = h_t + F_\theta(h_t, x_{t+1})$$

where $F_\theta$ is the composite vector field of self-attention and FFN inside the Transformer layers (including residual connections). This is fully consistent with the form of ResNet (ch6, §6.11) and GPT autoregression (ch6, §6.13)—only in the context of chain of thought, the intermediate states $h_t$ are not arbitrary text-generation states, but correspond to conceptual stages of reasoning.

![ResNet residual block computation graph](/figures/ch07_residual_block_graph_tikz.svg)

*ResNet residual block computation graph. x forks: the main path through Weight→ReLU→Weight produces F_θ(x), while the skip connection feeds x directly to ⊕. Their sum x+F_θ(x) is the explicit Euler step h_{ℓ+1}=h_ℓ+f_θ(h_ℓ).*

**Property 1 (Trajectory Continuity)**. $F_\theta$ is Lipschitz continuous with respect to $h_t$. When the step size $\eta = 1$, the magnitude of discrete jumps in the trajectory is bounded by $\|F_\theta(h_t, x_{t+1})\|$. Small magnitude of $F_\theta$ means the hidden state evolves smoothly between adjacent steps—this is precisely the geometric signature of coherent reasoning. Large-magnitude jumps—"ruptures"—correspond to logical leaps or errors in reasoning.

**Property 2 (Self-Driven Nature)**. $F_\theta(h_t, x_{t+1})$ depends on $x_{t+1}$—the token that the model **itself chooses**. Chain of thought is a **self-driven** dynamical system: the model chooses a direction at each step (by choosing a token), and then advances along that direction. This "push-oneself" structure makes the chain-of-thought trajectory richer than that of feedforward ResNets—and also more dangerous: a single wrong choice can push the trajectory toward a completely wrong attractor.

:::info

**Pallas's Cat Professor: The Skier**

A ResNet's trajectory is like a marble rolling down a hill—its initial position and the terrain determine its entire fate. The marble has no choice. It simply obeys gravity.

But a chain-of-thought trajectory is not a marble. It is a skier.

The skier looks at the terrain ahead and chooses a route—veering left around a rock, carving right over a snow bank. With every choice they make, the terrain shifts slightly because of their new position. The slope of the next stretch depends on which fork they just chose.

This is the essence of a self-driven system: **the model chooses a token at each step, and that token determines the direction of the next hidden-state update.** It is reasoning, and at the same time it is paving the road for its own reasoning. Every time it says "therefore," it creates a slightly different footing for its next "so."

This explains why chain of thought sometimes goes off the rails. If the skier takes a wrong fork—turns left when they should have turned right—they may ski into a dead end. And they cannot turn back, because with every extra meter they ski, the way back becomes steeper. When the model utters a wrong "first," every subsequent "then" has to navigate a terrain that the "first" has already reshaped. The error is not added on top of the correct answer—the error changes the road to the answer.
:::

This dynamical description explains the geometric difference between "coherent reasoning" and "derailed reasoning"—but a more vexing question still hangs in the air. The opening of ch7 raised this: **after CoT training, even without outputting a chain of thought, the accuracy of direct answers improves**. Why?

## 7.3 Implicit Reasoning: Trajectory Movement Without Tokens

**Definition 3 (Implicit Reasoning)**. If the hidden state $h_t$ undergoes a significant movement between two consecutive steps ($\|h_{t+1} - h_t\| > \delta$ for some threshold $\delta$), but the token generated during this interval is a structural marker (e.g., "therefore", "so"—whose semantic content contains no new reasoning information), then the model is said to have performed **implicit reasoning** between these two steps.

Implicit reasoning occurs in hidden state space but does not appear in the explicit chain. In the brief interval when the model outputs the words "therefore the answer is", it has internally completed a substantive belief update. The words you see have not changed—but the current beneath the water has already shifted direction.

**Theorem 1 (Terrain Reshaping by CoT Training)**. CoT training is equivalent to superimposing an additional regularization term on the loss function $L(\theta)$—forcing the hidden state, before reaching the final answer, to pass through a series of "concept anchors". Let the training objective for $\theta$ be:

$$L_{\text{CoT}}(\theta) = L_{\text{direct}}(\theta) + \lambda \sum_{i=1}^k \ell(h_i, c_i)$$

where $\ell(h_i, c_i)$ is the distance between the hidden state $h_i$ and the concept anchor $c_i$. After training, there exists a positive definite matrix $\Delta H$ (determined by the statistical structure of the CoT training data) such that, in the neighborhood of the correct attractor $p^*$:

$$F_x^{(\text{CoT})}(p) = F_x^{(0)}(p) - \Delta H \cdot (p - p^*) + o(\|p - p^*\|)$$

The addition of $\Delta H$ causes the eigenvalues of the Hessian at $p^*$ to decrease overall—the basin widens. Even without explicitly outputting CoT tokens, the vector field $F_\theta$ has already been reshaped: the flow of hidden states will automatically pass through those conceptual nodes, ultimately reaching the basin of the correct answer.

**This is the geometric reason why CoT training improves direct-answer accuracy**: it is not teaching the model to "write a few more steps"—it is widening the correct reasoning path, so that even without the guidance of intermediate tokens, the trajectory can naturally flow toward the correct answer.

This runs completely counter to our everyday intuition. We think that "writing out the steps" is the process of reasoning itself—as if the model must "speak aloud" to complete its thinking. But Theorem 1 tells us: **CoT training changes not the model's "speaking habits," but where the model goes when it is silent.**

It is a bit like learning to ride a bicycle. Before you have learned, you might need to say aloud: "First look left, then push with the right foot, keep balance..." But once you have learned, you no longer need to speak—your body knows. Not because the speaking itself helped, but because in the process of "speaking while riding," your body internalized a set of motor patterns.

CoT training does something exactly parallel. During training, the model is forced to pass through a series of conceptual anchors ($c_1 \to c_2 \to \cdots \to c_k$) in hidden state space. The positions of these anchors are determined by the "standard chain of thought" in the training data. After repeatedly traversing this path, the loss terrain is depressed along the line connecting the anchors—even without explicitly walking through these anchors, the gradient naturally points in their direction.

**You don't need the model to say "let's think step by step"—you have already made its internal gradients say it.**

With this explanation in hand, the next question emerges naturally: not every step in a chain of thought is equally important. Some steps are critical reasoning leaps, others are merely filler rhetoric. Can we distinguish them—without reading the text—solely through the geometric features of the hidden states?

## 7.4 Geometric Classification of Reasoning Steps

**Definition 4 (Geometric Classification of Reasoning Steps)**. Given a hidden state sequence $(h_0, \ldots, h_T)$, define the **hidden state displacement** between adjacent steps $\Delta h_t = h_t - h_{t-1}$ and the **normalized displacement angle** $\cos \alpha_t = \langle \Delta h_t, \Delta h_{t-1} \rangle / (\|\Delta h_t\| \|\Delta h_{t-1}\|)$. Based on these:

- **Substantive step**: $\|\Delta h_t\|$ is significantly larger than the mean displacement. Corresponds to key reasoning tokens—numbers, logical turns, introduction of new concepts.
- **Structural step**: $\|\Delta h_t\|$ is close to or smaller than the mean, and $\cos \alpha_t \approx 1$ (consistent direction). Corresponds to structural markers such as "first", "then"—the hidden state is barely moving; the token is merely punctuation for the human reader.
- **Corrective step**: $\|\Delta h_t\|$ is large, and $\cos \alpha_t < 0$ (direction reversal). Corresponds to the model "realizing" that the previous step was erroneous and self-correcting.

This classification does not depend on the semantic content of tokens—it is purely based on the numerical features of the hidden state trajectory. You do not need to read the text output by the model; you only need to track the time series of $\|\Delta h_t\|$ and $\cos \alpha_t$ to identify which steps in a chain of thought are genuine reasoning leaps.

:::info

**Pallas's Cat Professor: Consequences of Classification**

This purely geometric classification method has a radical consequence. If I can judge which of your steps are genuine reasoning, which are filler, and which are self-correction—without reading your words, only by examining the displacement vectors of your hidden states—then in principle I can **know whether you are reasoning without you writing out a chain of thought.**

This means "reasoning" can be transformed from a behavioral concept ("the model output text that looks logical") into a geometric concept ("the model underwent a substantive directional displacement in hidden state space"). You no longer need human annotators to label a chain of thought as "high quality"—you only need to compute $\|\Delta h_t\|$ and $\cos \alpha_t$ at each step, then count how many steps are "substantive."

This is one step away from "automatable reasoning quality assessment": run a correlation analysis between $\|\Delta h_t\|$ and correctness scores, and you have a reasoning evaluator that does not require a human judge.
:::

After classification, a deeper question emerges: **does the entire trajectory eventually converge?** Converge to where? How many steps does convergence require?

## 7.5 Convergence of Chain of Thought: The Yonglin Limit Applied to Reasoning

Placing chain of thought within the framework of the Yonglin Limit (ch5, §5.9), convergence is no longer an empirical observation—it becomes the conclusion of a theorem.

**Theorem 2 (Belief Convergence of Chain of Thought)**. Suppose that at step $t$, the model projects the hidden state $h_t$ onto the probability simplex via softmax, obtaining the belief distribution $p_t = \text{softmax}(W_{\text{lm}} h_t)$. Define the energy function $E(p_t) = D_{\mathrm{KL}}(p^* \| p_t)$, where $p^*$ is the distribution of the correct answer. If the reasoning operator $\Phi_\eta$ satisfies the contractivity condition from ch5 ($\eta < 2\mu/L^2$), then:

1. For problems that can be correctly solved via CoT reasoning, there exists a unique fixed point $p^*$, such that starting from the initial belief $p_0$, $p_t \to p^*$.
2. The convergence rate is at least linear: $D_{\mathrm{KL}}(p^* \| p_t) \leq k^t D_{\mathrm{KL}}(p^* \| p_0)$, $k = 1 - \eta(2\mu - \eta L^2)/C < 1$.
3. The minimum number of reasoning steps: $T_{\min} \geq \log(\epsilon / D_0) / \log k$, where $D_0 = D_{\mathrm{KL}}(p^*\|p_0)$.

This theorem turns the abstract guarantees of ch5 into direct constraints on chain of thought. The larger $D_0$ (the farther the initial belief is from the correct answer), the larger $T_{\min}$—but only logarithmically: if $D_0$ increases tenfold, the number of steps increases by only about $-\log_{1/k}(10)$ steps. The smaller $k$ (faster contraction, stronger model), the fewer steps needed. The two are related logarithmically—this is the "free lunch" that the exponential convergence rate of contraction mappings bestows upon chain of thought.

![DEQ fixed-point iteration computation graph](/figures/ch07_fixed_point_deq_graph_tikz.svg)

*DEQ fixed-point iteration. h_k enters f_θ, output h_{k+1} feeds back as input. The convergence check ||h_{k+1}-h_k||<ε decides when the structure has closed — the fixed point h*=f_θ(h*,x) is not computed, it is converged to.*

![Ring-shaped fixed point basin](/figures/ch07_fixed_point_basin_tikz.svg)

*Ring-shaped fixed point basin $(\theta_1^2+\theta_2^2-1)^2$. Fixed points can be entire manifolds, not just isolated points.*

![Fixed point attractor](/figures/ch07_fixed_point_tikz.svg)

*Fixed point attractor: $F(x^*)=x^*$. Four trajectories from different starting points all converge to the same structure.*

![Cobweb diagram](/figures/ch07_cobweb_tikz.svg)

*Cobweb diagram for $x_{t+1}=f(x_t)$. The intersection of $y=f(x)$ and $y=x$ is the fixed point; the staircase shows convergence.*

---

## 7.6 Chapter Summary

Chain of thought is one of the most easily misunderstood phenomena in deep learning. You see words and think that is reasoning—but the words are only a projection. The real reasoning happens on the continuous curve $h_0 \to h_1 \to \cdots \to h_T$ in representation space. The explicit chain is a discrete projection of the implicit chain. Implicit reasoning can occur without corresponding tokens. CoT training is not teaching the model to "write a few more steps"—it reshapes the local geometry of the reasoning field and widens the correct basin. The Yonglin Limit provides formal guarantees for convergence—when contractivity holds, chain of thought necessarily converges to the correct answer, and the convergence rate is exponential.

In the next chapter, we will zoom in on the invisible force that guides the trajectory—the reasoning field. It is a vector field spanning the entire belief space. Correct answers have wide basins; wrong answers have only narrow crevices. How do verifiers and RLHF reshape the mountains and rivers of this field?

---

## Unresolved Questions

**The core question left by this chapter is:**

**If chain of thought is merely a discrete projection of the hidden state trajectory, can we bypass the token sequence—directly observing, analyzing, and guiding reasoning in hidden state space?**

:::info

**The words you see are only ripples on the surface of the water.** The real reasoning—the continuous flow of hidden states in representation space—happens in depths you cannot see. CoT training does not teach the model to write a few more steps; it reshapes the terrain of those deep waters. The Yonglin Limit guarantees: as long as the Euler step is contractive under KL geometry, reasoning will necessarily converge—whether or not you write the intermediate steps. In the next chapter, we will dive into those deep waters—the reasoning field. That is an invisible world composed of vectors, attractors, and energy basins.

:::

## Further Reading and Related Work

The geometric perspective on chain of thought—trajectory projection, implicit reasoning, basin reshaping—resonates with the following works from different angles:

1. **Let's Verify Step by Step.** Lightman et al. (2023). [arXiv:2305.20050] — Process supervision versus outcome supervision: stepwise verification improves mathematical reasoning to 78% on MATH. From the geometric perspective of this chapter, process supervision is essentially imposing a KL constraint on the intermediate belief distributions along the reasoning trajectory—each step is pulled toward the correct answer, rather than only checking the result at the end. Process supervision is equivalent to applying a corrective force at every step along the trajectory in belief space, pointing toward the center of the correct basin; outcome supervision only pulls at the end—any deviation in the middle goes unpenalized.

2. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.** DeepSeek-AI (2025). Nature. [arXiv:2501.12948] — Pure RL elicits reasoning ability in LLMs, with emergent self-reflection, verification, and dynamic strategy adaptation. From our geometric perspective, RL broadens the basin of the correct attractor in belief space—it does not teach the model new reasoning steps, but enlarges the attraction radius of each correct step. R1's "aha moment" corresponds precisely to the trajectory passing through a critical saddle point in rugged terrain.

3. **Chain of Images for Intuitively Reasoning.** Meng et al. (2023). [arXiv:2311.09241] — CoI transforms complex language reasoning into visual pattern recognition, demonstrating that visual intermediate representations are more effective than text-only CoT. From our perspective, the projection of the reasoning trajectory need not be limited to text—images are also a legitimate projection space of hidden state trajectories. This shows that "reasoning" and "projection modality" are orthogonal: the same implicit reasoning can project onto text, images, or any symbolic space.

4. **Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models.** Hu et al. (2024). NeurIPS 2024. [arXiv:2406.09403] — Sketchpad allows multimodal LMs to draw lines, boxes, and markers on a visual canvas. From our perspective, this is a two-dimensional projection of hidden state trajectories in visual representation space—each line on the canvas is a "token" of the hidden state in visual space. Sketchpad's success further validates this chapter's core thesis: the essence of reasoning lies not in the choice of projection space, but in the geometric structure of the hidden state trajectory.

5. **M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models.** Wang et al. (2025). [arXiv:2504.10449] — Mamba hybrid reasoning models achieve 3x inference speedup. From our perspective, the upper bound on reasoning speed is determined by the geometric complexity of the trajectory in belief space, not by the architecture itself. Mamba's advantage lies in the fact that its linear attention's degenerate structure aligns with low-curvature regions of the trajectory in certain reasoning tasks—it does not make reasoning faster, but avoids wasting computation where attention is geometrically unnecessary.
