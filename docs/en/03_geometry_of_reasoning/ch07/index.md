# Chapter 7: Chain of Thought: The Projection of the Reasoning Trajectory

You see a model output a long string of text.

> "First, we need to calculate the total area... then subtract the overlap... therefore the answer is 42."

You think to yourself: it's reasoning. It's thinking step by step, moving from the problem toward the answer. What you see is a string of text—token after token, logic linked ring by ring.

But from a geometric perspective, what you see is only a **shadow**.

The real reasoning is not these words. The real reasoning happens in the model's hidden state space—a set of vectors of several hundred or several thousand dimensions, in places you cannot see, flowing step by step along a vector field jointly shaped by the training data and the model architecture. The tokens you see—"first", "then", "therefore"—are merely a discrete projection of this hidden trajectory into text space.

Chain of thought is not reasoning itself. **Chain of thought is the visible projection of the reasoning trajectory.**

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

With these two definitions, a natural follow-up question emerges: **how does the implicit chain itself move?** What laws govern each of its steps—$h_t \to h_{t+1}$?

## 7.2 Dynamics of the Implicit Chain: Euler's Method in Hidden State Space

Placing the implicit chain within the dynamical systems framework of ch6, everything becomes clear. $(h_0, h_1, \ldots, h_T)$ is not a string of independent vectors—it is a **discrete trajectory**. The update at each step precisely takes the form of the explicit Euler method:

$$h_{t+1} = h_t + F_\theta(h_t, x_{t+1})$$

where $F_\theta$ is the composite vector field of self-attention and FFN inside the Transformer layers (including residual connections). This is fully consistent with the form of ResNet (ch6, §6.11) and GPT autoregression (ch6, §6.13)—only in the context of chain of thought, the intermediate states $h_t$ are not arbitrary text-generation states, but correspond to conceptual stages of reasoning.

**Property 1 (Trajectory Continuity)**. $F_\theta$ is Lipschitz continuous with respect to $h_t$. When the step size $\eta = 1$, the magnitude of discrete jumps in the trajectory is bounded by $\|F_\theta(h_t, x_{t+1})\|$. Small magnitude of $F_\theta$ means the hidden state evolves smoothly between adjacent steps—this is precisely the geometric signature of coherent reasoning. Large-magnitude jumps—"ruptures"—correspond to logical leaps or errors in reasoning.

**Property 2 (Self-Driven Nature)**. $F_\theta(h_t, x_{t+1})$ depends on $x_{t+1}$—the token that the model **itself chooses**. Chain of thought is a **self-driven** dynamical system: the model chooses a direction at each step (by choosing a token), and then advances along that direction. This "push-oneself" structure makes the chain-of-thought trajectory richer than that of feedforward ResNets—and also more dangerous: a single wrong choice can push the trajectory toward a completely wrong attractor.

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

With this explanation in hand, the next question emerges naturally: not every step in a chain of thought is equally important. Some steps are critical reasoning leaps, others are merely filler rhetoric. Can we distinguish them—without reading the text—solely through the geometric features of the hidden states?

## 7.4 Geometric Classification of Reasoning Steps

**Definition 4 (Geometric Classification of Reasoning Steps)**. Given a hidden state sequence $(h_0, \ldots, h_T)$, define the **hidden state displacement** between adjacent steps $\Delta h_t = h_t - h_{t-1}$ and the **normalized displacement angle** $\cos \alpha_t = \langle \Delta h_t, \Delta h_{t-1} \rangle / (\|\Delta h_t\| \|\Delta h_{t-1}\|)$. Based on these:

- **Substantive step**: $\|\Delta h_t\|$ is significantly larger than the mean displacement. Corresponds to key reasoning tokens—numbers, logical turns, introduction of new concepts.
- **Structural step**: $\|\Delta h_t\|$ is close to or smaller than the mean, and $\cos \alpha_t \approx 1$ (consistent direction). Corresponds to structural markers such as "first", "then"—the hidden state is barely moving; the token is merely punctuation for the human reader.
- **Corrective step**: $\|\Delta h_t\|$ is large, and $\cos \alpha_t < 0$ (direction reversal). Corresponds to the model "realizing" that the previous step was erroneous and self-correcting.

This classification does not depend on the semantic content of tokens—it is purely based on the numerical features of the hidden state trajectory. You do not need to read the text output by the model; you only need to track the time series of $\|\Delta h_t\|$ and $\cos \alpha_t$ to identify which steps in a chain of thought are genuine reasoning leaps.

After classification, a deeper question emerges: **does the entire trajectory eventually converge?** Converge to where? How many steps does convergence require?

## 7.5 Convergence of Chain of Thought: The Yonglin Limit Applied to Reasoning

Placing chain of thought within the framework of the Yonglin Limit (ch5, §5.9), convergence is no longer an empirical observation—it becomes the conclusion of a theorem.

**Theorem 2 (Belief Convergence of Chain of Thought)**. Suppose that at step $t$, the model projects the hidden state $h_t$ onto the probability simplex via softmax, obtaining the belief distribution $p_t = \text{softmax}(W_{\text{lm}} h_t)$. Define the energy function $E(p_t) = D_{\mathrm{KL}}(p^* \| p_t)$, where $p^*$ is the distribution of the correct answer. If the reasoning operator $\Phi_\eta$ satisfies the contractivity condition from ch5 ($\eta < 2\mu/L^2$), then:

1. For problems that can be correctly solved via CoT reasoning, there exists a unique fixed point $p^*$, such that starting from the initial belief $p_0$, $p_t \to p^*$.
2. The convergence rate is at least linear: $D_{\mathrm{KL}}(p^* \| p_t) \leq k^t D_{\mathrm{KL}}(p^* \| p_0)$, $k = 1 - \eta(2\mu - \eta L^2)/C < 1$.
3. The minimum number of reasoning steps: $T_{\min} \geq \log(\epsilon / D_0) / \log k$, where $D_0 = D_{\mathrm{KL}}(p^*\|p_0)$.

This theorem turns the abstract guarantees of ch5 into direct constraints on chain of thought. The larger $D_0$ (the farther the initial belief is from the correct answer), the larger $T_{\min}$—but only logarithmically: if $D_0$ increases tenfold, the number of steps increases by only about $-\log_{1/k}(10)$ steps. The smaller $k$ (faster contraction, stronger model), the fewer steps needed. The two are related logarithmically—this is the "free lunch" that the exponential convergence rate of contraction mappings bestows upon chain of thought.

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
