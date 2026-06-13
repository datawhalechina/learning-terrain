# Volume III: The Geometry of Reasoning

The first two volumes built a complete geometric language: space, terrain, gradient, step size, Euler method, Bregman divergence, KL geometry, dynamical systems, fixed points.

Now, we apply this entire language to a specific problem—perhaps the most perplexing one in deep learning.

**What is reasoning?**

Not "the model outputs the correct answer." Not "the chain of thought looks logical." Not "thinking step by step." These are phenomena, not definitions.

The three chapters of Volume III give a geometric definition: **Reasoning is the flow of hidden states along the reasoning field $F_x$ in belief space, starting from an initial belief, undergoing several Euler iterations, and eventually falling into the basin of some attractor.**

---

Chapter 7 dissects the most easily misunderstood phenomenon in deep learning—chain of thought. You see words and think that is reasoning—but the words are only a projection. The explicit chain (token sequence) is a discrete projection of the implicit chain (hidden state trajectory) onto vocabulary space. CoT training is not teaching the model to "write a few more steps"—it reshapes the local geometry of the reasoning field and widens the correct basin. The Yonglin Limit guarantees: when contractivity holds, chain of thought necessarily converges. And implicit reasoning—belief updates without corresponding tokens—can happen in silence.

Chapter 8 zooms in on the invisible force guiding the trajectory—the reasoning field. It is a vector field spanning the entire belief space. Every problem casts a gravitational net across the simplex: correct answers have wide basins; wrong answers have only narrow crevices. Verifiers superpose repulsive fields around wrong basins—filling in the traps. RLHF redistributes energy among all fixed points—doesn't create new capabilities, but changes which one the model prefers. **You don't choose the answer. The field chooses for you.**

Chapter 9 explains why some problems need two steps and others need two hundred—not because they are "harder," but because on certain segments of the path through belief space, the safe step size $\eta_{\max}(p)$ is squeezed to a minimum by local curvature. Terrain dictates pace. Training, scale, temperature—all these methods are essentially **laying asphalt**: smoothing the rugged mountain path so each step can be longer.

---

Someone asks: "Is the model really reasoning?"

Volume III's answer: **Don't look at the tokens. Look at the trajectory. Look at the field. Look at the fixed point.** Reasoning is not behavior—reasoning is geometry.

---

**Volume III: The Geometry of Reasoning**
- [Chapter 7: Chain of Thought: Projection of the Reasoning Trajectory](ch07)
- [Chapter 8: Reasoning Fields: Attractors and Verifiers](ch08)
- [Chapter 9: Long Reasoning and Landscape Reshaping](ch09)
