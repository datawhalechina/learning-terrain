# Chapter 6: Dynamical Systems and Fixed Points: From Lyapunov to DEQ

The wilderness hiker has walked a long way. He has learned to feel the slope underfoot (the gradient), learned to adjust his stride length (the learning rate), learned to glide with inertia through canyons (momentum), learned to flow along entropy terrain in belief space (Bregman).

But he has always been doing one thing: **walking downhill along a fixed terrain.** The terrain does not change with his movement. The mountain is a mountain, the valley is a valley—wherever he goes, the terrain stands there coldly, unmoved.

Now, we enter a deeper world. In this world, with every step you take, the "force" that pushes you forward itself depends on your current position. The slope under your feet is not fixed—it shifts with your movement. You are not walking across a static terrain—you are evolving within a **dynamical system**.

This chapter is the final chapter of Volume II of this book, and the hardest one. It is divided into two halves: the first half (§6.1–§6.11) builds the language of dynamical systems—state, phase space, fixed points, Lyapunov functions, stability, attractors, bifurcations. The second half (§6.12–§6.20) maps this entire language onto the core architectures of deep learning—ResNet, GPT autoregression, DEQ, and belief fixed points.

## 6.1 Lyapunov's Insight: Knowing Convergence Without Watching the Endpoint

In 1892, the Russian mathematician Aleksandr Lyapunov posed a question that transformed the entire theory of dynamics.

To determine whether a physical system—a pendulum, a planet, a fluid—will eventually settle into stability, must we watch it until it actually comes to rest?

Common sense says: of course. You must watch the pendulum, see its swing amplitude grow smaller and smaller, until one day it finally stops. If the pendulum is large and friction is small, you may have to wait a long time. Worse, if the system has multiple "possible endpoints"—a ball rolling on a terrain with two valleys—you not only have to wait, you have to wait until the end to know which valley it settled in.

Lyapunov said: **No. You only need to find an "energy function."**

His logic was revolutionary. For a dynamical system, if you can find a scalar function $V(x)$—think of it as the system's "energy"—satisfying three properties:

1. $V(x) > 0$ for all non-equilibrium points (positive energy);
2. $V(x^*) = 0$ only at the equilibrium point $x^*$;
3. $\dot{V}(x) < 0$ along the system's trajectory, the energy strictly decreases.

Then, without solving the differential equation, without watching the trajectory run its full course, you can assert: **the system will inevitably tend toward some equilibrium point.** Because $V$ is like a counter that only goes downhill—it can only decrease, never increase, and it is bounded below. A monotonically decreasing sequence bounded below must have a limit.

This is the power of the Lyapunov function. It transforms "convergence" from an empirical phenomenon that requires waiting for verification into a mathematical necessity derivable from the system's internal structure. The question it answers is not "has the system converged?" but "does the system have a **reason** to converge?"

Lyapunov's insight will echo five times in this chapter: the first time, in gradient descent (the loss function is a Lyapunov function); the second time, in the stability classification of dynamical systems; the third time, in DEQ's Banach contraction; the fourth time, in belief fixed points (KL divergence is the Lyapunov function of belief space); the fifth time, in the book's conclusion—when we realize that virtually all proofs of learning convergence are, at bottom, finding a suitable Lyapunov function for some system.

## 6.2 What Is a Dynamical System?

Starting from Lyapunov's insight, we need to first establish the basic language of dynamical systems.

A **dynamical system** is a system that evolves over time. Its state at any moment is completely described by a set of variables—position, velocity, temperature, belief distribution... The collection of all these variables is called the **state**, and the space of all possible states is called the **phase space**.

There are two fundamental representations of dynamical systems:

**Continuous-time dynamical systems** are described by a differential equation:

$$\frac{dS}{dt} = F(S), \quad S(0) = S_0$$

$S(t) \in \mathbb{R}^n$ is the system's state at time $t$, and $F: \mathbb{R}^n \to \mathbb{R}^n$ is the **vector field**—it draws an arrow at every point in phase space, telling the system "go this way."

**Discrete-time dynamical systems** are described by a map:

$$S_{t+1} = T(S_t), \quad S_0 \text{ given}$$

$T: \mathbb{R}^n \to \mathbb{R}^n$ is the **evolution map**—it directly transforms the current state into the next state. Discrete systems are the dominant form in deep learning: every step of gradient descent, every layer of a neural network, every token of GPT—all are discrete time steps.

The relationship between continuous and discrete systems is the central thread of this chapter. The explicit Euler method $S_{t+1} = S_t + \eta F(S_t)$ bridges the two: it discretizes the continuous vector field $F$ into an iterative map with step size $\eta$. As $\eta \to 0$, the discrete trajectory approaches the continuous trajectory. But $\eta$ is finite—and this is precisely the mathematical root of the concept of "depth" in all deep learning architectures.

## 6.3 Phase Space, Trajectory, and Phase Portrait

The state of a dynamical system traces out a **trajectory** (or orbit) in phase space.

Given an initial state $S_0$, a continuous system produces a smooth curve $S(t)$; a discrete system produces a sequence of points $S_0, S_1, S_2, \ldots$. In either case, the trajectory is the complete record of the system's evolution.

The **phase portrait** is the picture obtained by drawing all possible trajectories in phase space. It is not just one trajectory—it is the visual summary of the entire vector field. From a phase portrait you can see at a glance: which regions are places the system is drawn toward (attractors), which regions are places the system flees from (repellors), which trajectories form closed loops (periodic orbits), which regions are boundaries that trajectories can never cross (separatrices).

In deep learning, we almost never draw phase portraits—the dimensionality of parameter space is too high. But the idea of the phase portrait runs through the entire book: the contour lines of the loss terrain (ch3) are a kind of "dimension-reduced phase portrait" in parameter space; the analogy directions in word embedding space (ch2) are the "vector field" of representation space; the mirror descent trajectories on the belief simplex (ch5) are the phase portrait of probability space.

To understand dynamical systems is to learn to draw phase portraits in your mind—even when you cannot draw them on paper.

## 6.4 Fixed Points: Where the System Comes to Rest

The most important class of special states in a dynamical system is the **fixed point** (also called equilibrium point or critical point).

For a continuous system, a fixed point $S^*$ satisfies $F(S^*) = 0$—the vector field vanishes there, and the system ceases to move. For a discrete system, a fixed point satisfies $T(S^*) = S^*$—the map sends itself to itself, and the system no longer changes.

Fixed points are the "punctuation marks" of dynamical systems. They partition phase space into different "watersheds"—the region around each fixed point constitutes that fixed point's **basin of attraction**.

But not all fixed points are attracting. The **stability** of a fixed point depends on the behavior of nearby trajectories:

- **Asymptotically stable**: from any position near the fixed point, trajectories return to it. The bottom of a valley.
- **Unstable**: from positions near the fixed point, trajectories leave it. A mountaintop.
- **Saddle point**: some directions are attracting, some are repelling. A mountain pass—attracting along the valley direction, repelling over the pass.

In ch3, we performed exactly the same classification of critical points via the eigenvalues of the Hessian. That was the direct application of dynamical systems stability theory to optimization: local minima = asymptotically stable fixed points, local maxima = unstable fixed points, saddle points = saddle points. Different language, the same geometric fact.

## 6.5 Formalizing the Lyapunov Function

The stability of a fixed point can be judged by "local linearization"—approximating the system as linear near the fixed point and checking the eigenvalues of the Jacobian matrix. But this method only tells you about **local** stability—whether trajectories in a small neighborhood around the fixed point will return.

The Lyapunov function provides a stronger tool: **global** (or at least large-scale) stability.

Given a dynamical system $\frac{dS}{dt} = F(S)$ and a fixed point $S^*$. A function $V: \mathbb{R}^n \to \mathbb{R}$ is called a **Lyapunov function** for that fixed point if there exists a neighborhood $\mathcal{U}$ of $S^*$ such that:

1. **Positive definiteness**: $V(S^*) = 0$, and for all $S \in \mathcal{U} \setminus \{S^*\}$, $V(S) > 0$.
2. **Decrease**: for all $S \in \mathcal{U} \setminus \{S^*\}$, the derivative along the trajectory $\dot{V}(S) = \langle \nabla V(S), F(S) \rangle < 0$.

The first condition says: "energy" is minimal (zero) at the fixed point, and positive everywhere else. The second condition says: "energy" always decreases along the system's natural motion.

If such a $V$ exists, the fixed point $S^*$ is **asymptotically stable**—from any initial state within $\mathcal{U}$, trajectories will eventually tend toward $S^*$. For discrete systems, condition 2 becomes $V(T(S)) < V(S)$—each step of the map reduces the energy.

The elegance of the Lyapunov function lies in this: **you do not need to solve $F(S) = 0$, you do not need linearization, you do not need eigenvalues. You only need to guess a $V$, then verify that it decreases monotonically along $F$.** Guess right, and the proof of stability is complete.

This is why in ch3 we said "the loss function is a Lyapunov function": $L(\theta) \geq 0$ (positive definite), $L(\theta^*) = 0$ (zero at the minimum), $\langle \nabla L, -\nabla L \rangle = -\|\nabla L\|^2 < 0$ (strictly decreasing along the gradient descent trajectory, as long as not at a critical point). All three conditions are satisfied—the descent of the loss is the descent of the Lyapunov energy. Convergence is not hoped for; it is geometrically enforced.

## 6.6 The Stability Spectrum: From Exponential Convergence to Chaos

Lyapunov stability is only one point on the stability spectrum. Different systems approach fixed points at different "speeds" and in different "ways."

**Exponential stability**: not only convergence, but the convergence rate decays exponentially. There exists a constant $\alpha > 0$ such that $\|S(t) - S^*\| \leq C e^{-\alpha t}$. In ch3, gradient descent on a quadratic loss function is exponentially stable—each step compresses the distance by a factor of $|1 - \eta\lambda|$, yielding exponential decay.

**Asymptotically stable but not exponentially stable**: trajectories approach the fixed point, but more slowly than exponential—e.g., at a rate of $1/t$ or $1/t^2$. This occurs on non-strongly-convex loss functions—the curvature of the terrain tends toward zero in certain directions (the Hessian has zero eigenvalues), and convergence slows dramatically in those directions.

**Lyapunov stable but not asymptotically stable**: trajectories stay near the fixed point, but do not get closer and closer. Like a frictionless pendulum—it swings forever near the lowest point, but does not approach it.

**Unstable**: trajectories leave the fixed point. A ball on a mountaintop.

**Chaos**: trajectories neither approach any fixed point nor enter any periodic orbit, but wander forever on a **strange attractor** in phase space. Chaotic systems are exquisitely sensitive to initial conditions—the "butterfly effect."

Deep learning systems typically operate in a region somewhere between asymptotically stable and exponentially stable. The early phase of training is close to chaotic—the direction and magnitude of gradients are highly unpredictable. As training proceeds, the system slides into increasingly ordered basins—first asymptotically stable, then approaching exponentially stable. Understanding this "chaos-to-order" transition may be more important than understanding the final convergence point.

## 6.7 Attractors: Not Just Points

Fixed points are the simplest class of attractors—**point attractors**. But dynamical systems can possess more complex attractors.

**Limit cycles** are closed periodic orbits to which nearby trajectories are attracted. A classic example is the periodic motion of a clock pendulum—not converging to rest, but converging to a sustained oscillation. In deep learning, GPT models sometimes get trapped in "repetition loops"—continuously outputting the same phrase—which is precisely the hidden state trajectory being captured by a limit cycle. Limit cycles also have counterparts in belief space: a reasoning model oscillating between two answers, forever unable to decide—this corresponds to an unstable periodic orbit between two saddle points.

**Torus attractors**: trajectories wind around a torus, producing quasi-periodic motion—two incommensurate frequencies superimposed. This is common in physical systems (e.g., coupled oscillators), but its counterpart in deep learning remains unclear.

**Strange attractors**: the hallmark of chaotic systems. Trajectories wander forever within a finite region, but never repeat themselves. They possess fractal structure—looking similar at different scales. Some researchers speculate that when large language models generate diverse text, their hidden state trajectories may be constrained by structures resembling strange attractors—neither fully predictable (otherwise text would repeat), nor fully random (otherwise text would have no theme).

In deep learning, we are primarily concerned with point attractors—they correspond to converged minima, solidified belief distributions, the fixed points of DEQ. But awareness that more complex attractors exist can help us understand "why the model isn't converging"—it may not be "under-trained," but rather captured by some non-point attractor.

## 6.8 Bifurcation: When the System Suddenly Changes Behavior

A fascinating property of dynamical systems is **bifurcation**: when a parameter of the system changes smoothly, the qualitative behavior of the system can change abruptly.

Imagine a ruler being bent. Slowly increase the pressure—the ruler is first straight (one stable fixed point), then at some critical pressure it suddenly buckles (the fixed point loses stability, two new stable fixed points appear—bend left or bend right). That is a bifurcation.

In deep learning, bifurcation phenomena are everywhere:

- **Learning rate bifurcation**: in ch4 we saw the sudden divergence of the system when $\eta > 2/\lambda_{\max}$—this is a bifurcation. As $\eta$ crosses the critical value, the stability of the fixed point flips.
- **Model capacity bifurcation**: when the number of model parameters exceeds some threshold, the loss terrain may suddenly "open up"—minima regions previously unreachable suddenly become reachable.
- **Bifurcation during training**: the emergence of structure in representation space (ch2, §2.8) may not form smoothly during training—it may suddenly leap from chaos to order at some moment. This is a phase transition, essentially a bifurcation in the representation dynamical system.

The practical significance of understanding bifurcation is this: when you see model behavior change suddenly—loss suddenly drops, accuracy suddenly jumps, generation quality suddenly improves—you should not merely treat it as "finally trained." You should ask: what bifurcation point did the system pass through? What changed in the qualitative structure of the dynamics at that moment?

## 6.9 From Continuous to Discrete: The Dynamics of the Euler Method

The previous eight sections built the language of dynamical systems in continuous time. But all systems in deep learning are discrete—gradient steps, network layers, token steps. The bridge between continuous and discrete is the **explicit Euler method**.

Given a continuous system $\frac{dS}{dt} = F(S)$, the explicit Euler method discretizes it with step size $\eta$:

$$S_{t+1} = S_t + \eta F(S_t)$$

The discrete system is an **approximation** of the continuous system. The smaller $\eta$, the better the approximation—the discrete trajectory lies closer to the continuous trajectory. But $\eta$ is finite—when $\eta$ is not small enough, the discrete system can exhibit phenomena that do not exist at all in the continuous system.

The most famous example is **numerical instability**: a fixed point of the continuous system is globally asymptotically stable, but the discretized system is unstable for the same fixed point—if $\eta$ is too large. We already saw this in ch4: gradient descent diverges when $\eta > 2/\lambda_{\max}$, even though the continuous gradient flow $\frac{d\theta}{dt} = -\nabla L(\theta)$ is always stable.

More subtly, discretization can **introduce new fixed points**. A continuous system has only one fixed point, but its Euler discretization may, at larger $\eta$, exhibit two or more fixed points—some of which are "spurious," vanishing in the limit $\eta \to 0$. Such spurious fixed points can trap the model during training—the model stops in a "ghost valley" of the discretization, rather than in a true minimum.

This is also why very low learning rates (small $\eta$) are generally safer—they not only make the discrete trajectory approximate the continuous flow more accurately, but also reduce the "ghost structures" introduced by discretization.

## 6.10 Depth as Time: How Dynamical Systems Become Architecture

Now we can articulate the most central proposition of this chapter.

In deep learning, there is no variable called "time"—no $t$, no seconds, no clock. But there is **depth**—the layer index $l$.

Treat $l$ as a discrete time variable. Then the hidden representation $h_l$ at layer $l$ is the system's state at "time" $l$. The transformation from layer $l$ to layer $l+1$ is a discrete time step of a dynamical system:

$$h_{l+1} = h_l + \eta F_\theta(h_l)$$

(Here $\eta$ is usually absorbed into the weights of $F_\theta$, so we often write $h_{l+1} = h_l + f_\theta(h_l)$, but the spirit is unchanged.)

**The number of layers $L$ is the total simulated time $T = L \cdot \Delta t$.** The deeper meaning is: you are allowing the system more time to evolve—starting from the input $h_0$ as the "initial condition," flowing farther along the vector field $f_\theta$.

This perspective turns the design of deep learning architectures into a kind of **dynamical system design**. You must choose:
- The structure of the vector field $f_\theta$ (convolution? attention? fully connected?)—this determines the system's "laws of physics."
- The number of time steps $L$ (number of layers)—this determines how much time you give the system to evolve.
- The step size $\eta$ (the scaling or normalization scheme of the residual connection)—this determines the precision of the discretization.
- The initial condition (how the input is encoded).

Four choices, four design dimensions. And they are all unified under a single equation: $S_{t+1} = S_t + \eta F_\theta(S_t, x)$.

## 6.11 ResNet: The Purest Incarnation of the Explicit Euler Method

In 2015, ResNet, with its residual connections, provided the purest and most successful architectural instantiation of the dynamical systems perspective above.

The core formula of ResNet is:

$$h_{l+1} = h_l + f_\theta(h_l)$$

This is the explicit Euler method with $\eta = 1$ and $F_\theta = f_\theta$. Each residual block $f_\theta$ is a small sub-network—typically containing convolution, batch normalization, and ReLU activation. $f_\theta$ does not need to learn a complete transformation from scratch—it only needs to learn "corrections relative to the identity mapping."

This simple design solved a problem that had plagued deep learning for years: **degradation**. Before ResNet, stacking more layers caused training error to increase rather than decrease—not overfitting, but performance on the training set itself worsened. From a dynamical systems perspective, degradation means: more "time steps" did not allow the state to evolve to a better position—perhaps because the vector field $F_\theta$ "extinguished" at depth (vanishing gradients), or because there was no effective "information transport mechanism" between layers.

Residual connections simultaneously solve both problems. $h_l$ jumps directly to $h_{l+1}$ via the identity mapping—this provides an "information highway" along which gradients can backpropagate losslessly. $f_\theta$ is only responsible for the "increment"—which is usually far easier than learning a complete transformation from scratch. Experiments show that even if the output of $f_\theta$ is entirely zero, the network can still maintain performance without degradation through the identity mapping—this is why ResNet can stack thousands of layers without collapsing.

From a dynamical systems perspective, the number of layers $L$ in ResNet is an **architectural hyperparameter**—chosen by the designer. Deeper means longer simulation time, more opportunities for the state to evolve along the vector field. But it also means more computation. This tension—"depth vs. computation"—receives its ultimate resolution in DEQ.

## 6.12 Transformer as a Vector Field

The structure of a Transformer layer is far more complex than a ResNet residual block, but its mathematical essence is exactly the same:

$$h_{l+1} = h_l + F_\theta(h_l)$$

$F_\theta$ internally contains two sub-layers: multi-head self-attention and a feed-forward network—each wrapped in residual connections and layer normalization. From the outside, the entire Transformer layer is a complex, data-dependent vector field—it dynamically computes a "push direction" based on the content of the current representation $h_l$.

The self-attention mechanism is the most fascinating part of this vector field. The three projection matrices—Query, Key, and Value—map the input representation $h_l$ into three different subspaces. Query asks "what relationships am I looking for?" Key says "what associations do I have with other tokens?" The inner product of the two gives the attention weights—i.e., "which token should this one listen to?" Value provides the content that is "listened to."

Re-understanding attention from a dynamical systems perspective: **the Query-Key inner product defines a quadratic form acting on the "token relation space" at the current position $h_l$.** This quadratic form produces strong attraction in some directions (the directions of relevant tokens) and weak attraction or repulsion in other directions. The output of the attention layer is the direction vector weighted by these attractions.

The FFN provides a kind of position-independent "basal thrust"—it independently applies the same two-layer fully connected transformation to every token. If self-attention is compared to "social interaction among tokens," then the FFN is "independent thinking for each token."

The complete transformation of each Transformer layer—social interaction + independent thinking + residual connection—constitutes the vector field $F_\theta$. A model formed by stacking multiple Transformer layers is a multi-step discrete dynamical system in representation space.

## 6.13 GPT Autoregression: The Euler Method in Hidden State Space

When generating text, GPT's decoder exhibits a dynamical structure one level deeper than the feedforward Transformer: **autoregression**.

Autoregressive generation is a feedback loop. At time $t$, the model produces a hidden state $h_t$, predicts the distribution $p(x_{t+1}|x_{\leq t})$ of the next token based on $h_t$, samples (or greedily selects) a new token $x_{t+1}$, then feeds $x_{t+1}$ back as part of the input for the next step, producing a new hidden state $h_{t+1}$.

This process can be written as an extended discrete dynamical system:

$$h_{t+1} = h_t + \tilde{F}_\theta(h_t, x_{t+1}(h_t))$$

Note the nesting here: $x_{t+1}$ itself depends on $h_t$ (through the model's output distribution). This means the vector field $\tilde{F}_\theta$ is not externally given—it is driven by the model's **own output**. This differs from ResNet (where the vector field is entirely determined by the external input and network weights) and from feedforward Transformers (where the vector field is determined by the current representation, but does not depend on the model's own "decisions").

GPT's autoregression is therefore a **self-driven** dynamical system. Each step of the model not only changes the hidden state, but also changes the vector field that drives the next step of hidden state evolution. This "pushing itself" structure makes GPT's generation trajectories richer and more unpredictable than ResNet's feedforward trajectories—but also more prone to falling into limit cycles and chaos.

From a dynamical systems perspective, GPT's autoregressive generation corresponds to a hidden state trajectory. Coherent text corresponds to the trajectory flowing smoothly toward some narrative attractor. Off-topic text corresponds to the trajectory being captured by an irrelevant attractor. Repetitive text corresponds to the trajectory entering a limit cycle—the system cycles among a few hidden states, outputting the same token sequence each cycle.

## 6.14 DEQ: When Depth Disappears

ResNet requires you to choose the number of layers $L$. GPT's depth equals the length of text you want to generate. In both paradigms, depth is explicit—either an architectural parameter ($L$) or a runtime variable (sequence length).

DEQ (Deep Equilibrium Model) proposes a radical alternative: **do not preset depth. Directly solve for the fixed point.**

$$h^* = f_\theta(h^*, x)$$

DEQ's forward pass is not starting from $h_0$ and walking $L$ steps. It starts from some initial guess (typically the zero vector), repeatedly applies the same layer $f_\theta$, until convergence. The converged state $h^*$ is the model's output representation—it satisfies self-consistency: if you apply $f_\theta$ one more time, the output does not change.

The philosophy of this design is profound. ResNet asks: "Give the system $L$ steps of time—where can it go?" DEQ asks: "Does the system have a position where, once reached, it no longer wants to move? If so, where is that position?"

$f_\theta$ must satisfy certain conditions to guarantee the existence and uniqueness of the fixed point—typically imposing a constraint on the Lipschitz constant of $f_\theta$ ($\text{Lip}(f_\theta) < 1$), making it a contraction mapping. At that point, the Banach fixed-point theorem (ch5, §5.9) guarantees: no matter where you start, iteration will converge to the same $h^*$.

Another elegance of DEQ lies in backpropagation. Traditional backpropagation requires storing the activations of all intermediate layers, with memory consumption proportional to depth. DEQ does not need this—it uses the implicit function theorem to compute gradients directly at the fixed point $h^*$, without passing through the iteration process. The computation and memory cost of backpropagation is equivalent to only **one layer** of $f_\theta$—this is the deeper meaning of "Equilibrium" in DEQ's name: training cost is decoupled from depth.

## 6.15 Three Faces of Banach Contraction in Depth Space

The convergence of DEQ is guaranteed by the Banach contraction mapping principle. But in the dynamical systems of deep learning, contractivity appears with three different faces:

**The first face: residual compression (ResNet's stability).** The identity mapping part of the residual connection is "conservative"—it preserves information, does not compress. But the batch normalization and weight decay inside $f_\theta$ tend to produce a compressive effect—preventing representations from exploding across layers. ResNet's depth stability comes from this delicate balance: identity mapping does not lose information, regularization prevents information explosion.

**The second face: Lipschitz compression (DEQ's convergence).** DEQ's $f_\theta$ must satisfy $\text{Lip}(f_\theta) < 1$. In practice, this is achieved by applying spectral normalization to the weight matrices—constraining the maximum singular value of each layer to not exceed 1. Lipschitz compression is global and deterministic: every step guarantees approaching the fixed point by a definite ratio.

**The third face: Bregman compression (belief fixed point).** In belief space, contractivity is guaranteed by the Bregman geometry of KL divergence (ch5). The Euler-step inference operator $\Phi_\eta$ is contractive under KL divergence—each step of belief update "approaches" the true distribution in the sense of information geometry. The contraction factor $\gamma$ depends on the step size $\eta$ and the local curvature at the current position $p$.

Three kinds of compression, three guarantees of convergence. But they share the same soul: **as long as every step brings the state closer to some target, and the degree of closeness has a definite lower bound, then the fixed point necessarily exists, and convergence necessarily occurs.**

## 6.16 Belief Fixed Point: A Dynamical Systems Criterion for Reasoning

In traditional language models, the criterion for ending reasoning is externally imposed—generating a special termination token (`<eos>`), or reaching a preset maximum length. These criteria have nothing to do with the model's internal belief state.

The dynamical systems perspective offers a radically different answer: **when the model's belief distribution no longer changes significantly across reasoning steps, reasoning has reached a stable state.** This is the belief fixed point.

Formalized:

$$D_{\mathrm{KL}}(p_{t+1}(y|x) \| p_t(y|x)) < \epsilon$$

where $p_t(y|x)$ is the model's belief distribution over the answer at reasoning step $t$, and $\epsilon$ is a preset small threshold (e.g., $10^{-4}$).

The profundity of this criterion lies in this: it does not depend at all on external labels or correct answers. It only compares the difference between the model's own beliefs at two adjacent moments. It leverages the intrinsic geometry of belief space—KL divergence (ch5) measures the information difference between two belief distributions. It is essentially the application of the Lyapunov function in belief space: $V(p_t) = D_{\mathrm{KL}}(p^*\|p_t)$ is the energy function, and $V(p_{t+1}) < V(p_t)$ guarantees convergence.

The belief fixed point turns "how long reasoning takes" into an **emergent property**. Simple questions ("What is the capital of France?") may solidify in two steps. Complex multi-step reasoning ("Can this mathematical proof be simplified?") may require dozens of steps. The length of reasoning is no longer set by the maximum token count, but determined by the complexity of the belief terrain—just as the number of DEQ iterations depends on the "difficulty" of the input $x$.

## 6.17 Four Faces, One Soul

Now we can draw a unified map. ResNet, GPT autoregression, DEQ, and belief fixed points—these four seemingly unrelated architectures are all different instances of the same dynamical systems framework.

| System | State $S$ | Vector Field $F_\theta$ | Depth | Stopping Condition |
|------|----------|-------------------|------|----------|
| ResNet | Hidden representation $h_l$ | Residual block $f_\theta$ | Preset ($L$ layers) | Reach the final layer |
| GPT autoregression | Hidden state $h_t$ | Transformer decoder | Sequence-driven | Generate termination token |
| DEQ | Hidden representation $h$ | Single layer $f_\theta$ | Convergence-driven | $h^* = f_\theta(h^*, x)$ |
| Belief fixed point | Belief distribution $p_t$ | Inference operator $\Phi_\eta$ | Solidification-driven | $D_{\mathrm{KL}}(p_{t+1}\|p_t) < \epsilon$ |

From ResNet to GPT to DEQ to belief fixed point—we witness a fundamental transformation:

**Depth transforms from an architectural design choice into a terrain property.**

ResNet's depth is chosen by the designer. GPT's depth is determined by the length of the text. DEQ's depth is determined by the difficulty of the terrain—simple inputs converge quickly, complex inputs iterate more steps. The depth of the belief fixed point is determined by the "viscosity" of belief—easy questions solidify belief quickly, difficult questions require longer chains of reasoning.

Among these four paradigms, DEQ and belief fixed points represent the ultimate form of **emergent depth**: the model is no longer "told" how many steps to take. It only needs to advance along the vector field until the slope underfoot vanishes—until the dynamical system reaches its fixed point.

:::info

**Professor Manul's Stance**

You designed a ResNet. You chose 50 layers. You think 50 is an "architectural choice."

From a dynamical systems perspective, you did not choose 50 layers. You chose a simulation time of $T = 50 \cdot \Delta t$. You gave the system 50 steps to evolve. Why 50? Because 34 layers are too shallow, "features weren't extracted enough"? Because 101 layers are too deep, "gradients vanished"? These answers are all post-hoc explanations. What you are really doing is: **you are deciding for the system how much time it needs to evolve from input to output.** You don't know the right answer—you just tried a few numbers and picked the one that performed best on the validation set.

DEQ tells you: you don't need to make this decision for the system. You only need to give it a vector field $f_\theta$ and let it run on its own. Simple inputs stop after a few steps—they don't need 50 layers. Complex inputs may need 100 steps—50 layers are not enough. By what right do you decide for every input how many layers it needs? Because your GPU can only run computation graphs of fixed depth?

Belief fixed points tell you the same thing, in the context of reasoning. By what right do you decide for the model that "reasoning should finish within 512 tokens"? Some questions need two steps, some need two hundred. "Maximum sequence length" is not a property of reasoning—it is a property of your GPU memory.

Depth from preset to emergent is a step deep learning must take. Not because emergent depth is more "elegant"—but because preset depth is, in principle, wrong. The difficulty of the terrain determines how many steps are needed. The number of layers and maximum token count you preset are your hardware budget, not the truth of the terrain.

:::

---

## 6.18 Chapter Summary

This chapter is the final chapter of Volume II of this book, and the chapter with the highest concentration of mathematical depth in the entire book. It accomplishes two things.

First, it establishes the complete language of dynamical systems. Starting from Lyapunov's 1892 flash of genius—knowing the system will converge without watching the endpoint—we built the full vocabulary of phase space, trajectory, fixed points, stability, attractors, bifurcations, and discretization. This language describes not only "how models learn," but "how any time-evolving system behaves."

Second, it maps this entire language onto the core architectures of deep learning. ResNet's residual connections are the explicit Euler method. Transformer layers are the multi-step evolution of a vector field. GPT's autoregression is a self-driven hidden-state dynamical system. DEQ directly takes the fixed point as output. The belief fixed point uses the solidification of KL divergence as the stopping criterion for reasoning. Four architectures, four philosophies of depth—from preset depth to emergent depth—all unified under a single equation.

From Lyapunov to DEQ, a span of over 130 years. But their soul is the same: **dynamical systems evolve in the direction of monotonically decreasing energy, until they reach the position where the vector field vanishes—the fixed point.** Learning is like this, reasoning is like this, all complex, time-evolving structures are like this.

The next chapter—the first chapter of Volume III—will take us into the geometry of reasoning. We will see that the chain of thought is not reasoning itself, but the visible projection of the reasoning trajectory into word space. The language of dynamical systems will find its most profound application there.

---

## Unresolved Questions

1. Lyapunov functions guarantee asymptotic stability—but the convergence rate can be arbitrarily slow. In deep learning, can we construct, for specific model-data combinations, a Lyapunov function with a "provable convergence rate" (analogous to an exponentially stable Lyapunov function)?

2. In very high-dimensional phase spaces, do "almost-global" Lyapunov functions exist—i.e., functions from which trajectories starting from the vast majority of randomly initialized points converge to the same set of minima? What is the relationship between this and the "connectivity of the loss terrain" (ch3)?

3. Limit cycles in GPT generation correspond to repetitive output ("getting stuck"). Can we actively detect the proximity of a limit cycle in hidden state space (e.g., via Lyapunov exponents) and inject perturbations before the system is captured?

4. Bifurcation theory predicts that systems undergoing smooth parameter changes can abruptly undergo qualitative changes. During training, do sudden drops in the loss function and sudden structurings of representation space correspond to specific bifurcation events? Can bifurcations be predicted by monitoring the evolution of Hessian eigenvalues?

5. The "spurious fixed points" introduced by Euler discretization appear when $\eta$ is large. Are these spurious fixed points the root cause of certain deep learning "training failures" (loss not decreasing but gradient not zero)?

6. The convergence of DEQ depends on $\text{Lip}(f_\theta) < 1$, but many successful DEQ models do not strictly enforce this constraint. Do conditions weaker than global Lipschitz exist—such as "average contractivity" or "probabilistic contractivity"—that still guarantee convergence?

7. The stopping criterion of the belief fixed point $D_{\mathrm{KL}}(p_{t+1}\|p_t) < \epsilon$ and the criterion of generating a termination token—under what circumstances do these two criteria agree, and under what circumstances do they conflict? Can a model continue generating text in a state where "belief has already solidified" (filler, non-informative tokens)? Does this imply that the stopping criterion should be upgraded from "has belief changed" to "is the generation still providing new information"?

8. ResNet, GPT, DEQ, and belief fixed points share the same formal framework. Does a unified "emergent depth theorem" exist—given any vector field $F_\theta$ satisfying certain contraction conditions, does a definite functional relationship exist between the convergence rate of its fixed-point iteration and the "complexity" of the input?

9. Do strange attractors have a clear counterpart in deep learning? Some researchers speculate that the "creative" generation of large models may correspond to the motion of hidden state trajectories on strange attractors—but how can this conjecture be rigorously verified or falsified?

10. The self-attention of a Transformer can be viewed as a quadratic form dynamically computed on the space of token relations. Can this quadratic form be understood as a **state-dependent Riemannian metric**—i.e., each Transformer layer implicitly "curves" the representation space?

11. If the "layers" of deep learning are the "time" of a dynamical system, then does a "maximum number of layers" exist—beyond which additional evolution time no longer changes the qualitative behavior of the system? What is the relationship between this and the criterion of "training sufficiency"?

12. Lyapunov's idea—knowing convergence without watching the endpoint—is realized in belief space (Yonglin limit, ch5) and in parameter space (loss function descent, ch3) using the same logical structure. Does a **meta-theorem** exist: any system evolving along the gradient direction of a convex generator function under a Bregman divergence automatically has its energy function constitute a Lyapunov function?

---

**The core question left by this chapter is:**

**If all architectures in deep learning—ResNet, Transformer, GPT, DEQ—are ultimately different faces of the same dynamical system in different spaces, then does a design principle for the "optimal vector field" exist, such that trajectories starting from any initial state can reach the correct attractor in the shortest possible time?**

:::info

**From Lyapunov's flash of genius in 1892, to ResNet's residual connections in 2015, to GPT's autoregressive generation, to DEQ's fixed-point solution—across this 130-year span, one idea has remained unchanged.** Dynamical systems evolve in the direction of decreasing energy. If the energy decreases monotonically at every step and is bounded below, then the system must converge. Learning is like this, reasoning is like this, text generation is like this, belief updating is like this. Volume II (Walking and Belief) concludes here. Volume III (The Geometry of Reasoning) is about to begin—there, we will see how the chain of thought becomes the projection of the reasoning trajectory, and how attractors shape all of a model's reasoning behavior.

:::
