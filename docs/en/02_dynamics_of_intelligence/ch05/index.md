# Chapter 5: The Non-Euclidean World: Bregman Divergence and KL Divergence

A crow and a drop of water set out from the same mountaintop.

The crow spreads its wings and flies along a straight line — a perfectly straight segment in three-dimensional space — toward a distant lake. This is the Euclidean path. Between two points, the straight segment is the shortest. The crow does not care about the terrain beneath it — it flies through the air, unconstrained by slopes, valleys, and rivers.

The water droplet does not fly. It slides down the hillside, merges into a stream, meanders along the river valley, and flows along the gradient of the terrain toward the lowest point. Water does not take the straight line — it follows the direction of **steepest energy descent**. On a hillside, this is the steepest downhill direction; on a plain, it is the most natural convergence direction for water flow. From mountaintop to lake, the water's path is almost certainly longer than the crow's — but that is because the water is not taking the shortest path in space, but the **steepest descent path on the energy landscape**.

In the first four chapters, we have been speaking the crow's language — Euclidean distance, Euclidean gradient, Euclidean step size. But when we enter belief space — the space constituted by probability distributions — the crow's language begins to fail.

The reason is simple. The "difference" between two probability distributions $p = (0.5, 0.5)$ and $q = (0.51, 0.49)$ — the magnitude of this difference — and the difference between $p = (0.99, 0.01)$ and $q = (1.0, 0.0)$ — their Euclidean distances are both $\sqrt{(0.01)^2 + (-0.01)^2} \approx 0.014$. But any statistician will tell you: the former are two nearly indistinguishable distributions, while the latter is a leap from "almost certain" to "absolutely certain" — these are fundamentally different kinds of changes.

Euclidean distance cannot see this difference. Because it treats the probability simplex as a flat space — every coordinate direction equal, every region homogeneous. But probability space is not flat. Near the boundaries (where some $p_i$ approaches 0 or 1), the same absolute change corresponds to a huge change in information content. Near the center (where all $p_i$ are close to uniform), the same absolute change barely alters the information content.

We need a new kind of geometry — a geometry that flows along energy landscapes like water. The language of this geometry is called **Bregman divergence**.

## 5.1 From Taylor Expansion to Energy Difference

To understand Bregman divergence, the most natural entry point is not the definition, but a simple question: **when you approximate a convex function with a first-order Taylor expansion, how large is the error?**

Let $F: \mathbb{R}^n \to \mathbb{R}$ be a strictly convex and differentiable function. Given two points $p$ and $q$, the first-order Taylor expansion of $F(p)$ around $q$ is:

$$F(p) \approx F(q) + \langle \nabla F(q), p - q \rangle$$

Because $F$ is convex, this linear approximation always **underestimates** the true value of $F(p)$ (the graph of a convex function lies above its tangent lines). The gap between the true value and the approximation exactly captures the "degree of curvature" of $F$ over the interval from $q$ to $p$ — that is, the accumulated effect of $F$'s curvature along this interval.

Bregman divergence defines precisely this gap:

$$D_F(p \| q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle$$

This is the most important formula in this chapter. It has only three terms: $F(p)$ is the function value at $p$, $F(q) + \langle \nabla F(q), p - q \rangle$ is the estimate of $F(p)$ by linear extrapolation from $q$, and the difference between the two is the deviation between the "linear world" and the "curved world."

The generating function $F$ of the Bregman divergence acts like a lens — different $F$ define different ways of curving, and therefore define different "distances." But $D_F$ is not a distance in the traditional sense: it is asymmetric ($D_F(p\|q) \neq D_F(q\|p)$), and does not satisfy the triangle inequality. It measures **the "degree of surprise" when viewing $p$ from $q$** — how large an error you commit when using the local geometry at $q$ to estimate $p$. Conversely, when viewing $q$ from $p$, the size of the error is usually different — because the terrain curvature at $p$ and $q$ differs.

:::info

**Why is asymmetry not a defect?**

Imagine you are standing on a steep hillside (position $q$), looking toward a distant gentle valley (position $p$). From where you stand, you use the assumption "the terrain continues to be as steep as it is here" to estimate the elevation of the valley — you will severely overestimate, because you do not know that the slope ahead is diminishing. $D_F(p\|q)$ measures the magnitude of this overestimation.

Now reverse the situation. You stand in the gentle valley (position $p$), looking toward the distant steep slope (position $q$). You use the assumption "the terrain continues to be as flat as it is here" to estimate the elevation of the steep slope — you will severely underestimate, because you do not know that the slope ahead is increasing. $D_F(q\|p)$ measures the magnitude of this underestimation.

These two errors are not equal — because the causes of the errors are different. $D_F(p\|q) \neq D_F(q\|p)$ is not a mathematical imperfection, but the curvature of the terrain telling you: viewing $B$ from $A$ and viewing $A$ from $B$ are, from the outset, two different perspectives.

:::

![Bregman divergence: the gap between the tangent line and the true value on a convex function](/figures/ch05_bregman_geometry.svg)

*Geometric definition of Bregman divergence. The curve is the negative entropy function $F(p) = p\log p$, the dashed line is the tangent at $q=0.6$. $D_F(p\|q)$ (orange vertical segment) is the vertical distance from the tangent line to the function curve — it measures the error caused by the curvature of $F$ when using the local linear approximation at $q$ to estimate the function value at $p$. Different $q$ yield different tangent lines — this explains the asymmetry.*

## 5.2 The Energy Landscape: Physical Intuition for Bregman Divergence

Formulas are cold. But Bregman divergence has an extremely vivid physical interpretation — and this interpretation is structurally isomorphic to the logic of Chapter 3, where the loss function draws a landscape over parameter space. And the logic of Chapter 3 itself traces back to an even older story from Chapter 1.

Recall the opening of Chapter 1. Newtonian mechanics starts from forces — you must analyze every force individually. A small ball rolling on an inclined plane: you analyze gravity, normal force, friction. A pendulum in motion: you analyze tension and gravity. Force analysis is item-by-item, local, tedious — in complex systems, it quickly degenerates into a tangled mess. The revolution of Hamiltonian mechanics was to transform "analyze every force" into "track a single energy function" — the motion of the system is no longer determined by scattered local forces, but guided by a global potential energy landscape $V(x)$. When the Hamiltonian is minimized, the system tends toward stability — you do not need to solve all the forces, you only need to follow the energy downward.

This is the power of energy. It transforms piecemeal analysis into holistic flow.

Chapter 3 transplanted the same idea into parameter space: the loss function $L(\theta)$ is that energy landscape, and gradient descent is the small ball sliding down the landscape. Now, Chapter 5 pushes this idea one layer further: in belief space, **the negative entropy function $F(p) = \sum p_i \log p_i$ is that energy landscape**, and the Bregman divergence $D_F(p\|q)$ is precisely the irreversible energy difference that arises from the terrain's curvature as the small ball slides from $q$ to $p$. Our entire motivation for introducing non-Euclidean metrics is to be able to use Hamiltonian thinking in belief space as well — **replacing the piecewise analysis of every belief dimension with a single energy function**, so that the model need not get lost in complex probability constraints and coordinate transformations, and can simply follow the entropy landscape downward.

In Chapter 3, the loss function $L: \mathbb{R}^N \to \mathbb{R}$ draws a landscape over parameter space — every parameter vector $\theta$ is assigned an elevation $L(\theta)$, and the gradient $\nabla L$ tells the model the downhill direction. Here, the convex function $F$ draws another landscape over the probability simplex — every distribution $p$ is assigned an "entropy elevation" $F(p)$. The structures of the two landscapes are parallel: the Chapter 3 landscape is carved by the loss function, used to guide parameter movement; the Chapter 5 landscape is carved by the entropy function, used to guide belief updates. And the Bregman divergence $D_F(p\|q)$ is exactly the "irreversible height difference" on this entropy landscape.

Concretely: understand $F$ as some kind of **potential energy function** — like gravitational potential or electric potential. $F(p)$ is the potential energy of the system in state $p$. To go from state $q$ to state $p$, the system must at least expend an energy difference of $F(p) - F(q)$ — if the space were flat.

But the space is not flat. At different locations on the landscape, the same coordinate change corresponds to different energy changes — because the gradient $\nabla F$ points in different directions and has different magnitudes at each location. The first-order approximation $\langle \nabla F(q), p - q \rangle$ assumes that along the entire path from $q$ to $p$, the terrain is inclined exactly as it is at $q$. But on a curved landscape, this assumption is broken. The slope changes, curvature comes into play.

$D_F(p\|q)$ measures precisely: **the additional energy the system gains (or loses) due to the curvature of the landscape during the journey from $q$ to $p$.** It is the "irreversible energy difference" — from $q$ to $p$ and from $p$ to $q$, the direction of the terrain's curvature differs, and so does the energy surplus.

This physical interpretation directly yields the asymmetry of Bregman divergence. Sliding from mountaintop to valley, gravity does work for you, and you gain kinetic energy. Climbing from valley to mountaintop, you must do work against gravity, and you expend physical effort. $D_F(\text{valley}\|\text{mountaintop}) \neq D_F(\text{mountaintop}\|\text{valley})$ — this is not a mathematical defect, but the natural consequence of energy conservation on a curved landscape.

:::info

**"Water flow" in belief updating: Bregman from a concrete example**

Suppose you are a model reading a paper. Your current belief distribution is $q = (0.3_{\text{accept}}, 0.5_{\text{skeptical}}, 0.2_{\text{reject}})$. After reading a powerful argument, your belief becomes $p = (0.7, 0.2, 0.1)$.

In the Euclidean world, the magnitude of this change is $\sqrt{(0.4)^2 + (-0.3)^2 + (-0.1)^2} \approx 0.51$. But note — "skeptical" dropped from 0.5 to 0.2 (change -0.3), while "reject" dropped from 0.2 to 0.1 (change -0.1). Euclidean geometry considers the former 3 times as important as the latter. But Bregman (KL) does not see it this way.

When $F$ is negative entropy, KL divergence cares about **relative change** rather than absolute change. "Skeptical" goes from 0.5→0.2, a relative ratio of $0.2/0.5 = 0.4$. "Reject" goes from 0.2→0.1, also a relative ratio of $0.5$. In KL's eyes, these two changes carry similar amounts of information — because both mean the model's belief in the corresponding option is roughly halved. "Accept" goes from 0.3→0.7, a doubling in ratio — KL pays the greatest information cost for this.

This is why Bregman divergence is more suitable for belief space than Euclidean distance: **it cares about the multiplicative effects of ratios, not the additive effects of absolute differences.** The flow of water does not look at how far the coordinates have moved, but at how much energy the terrain has dropped.

:::

## 5.3 Entropy Topography: The Topographic Construction of Bregman

The previous section said that Bregman divergence measures "energy difference." But what kind of geometric structure allows us to talk precisely about this energy difference?

The answer is **entropy topography**. This is the most central construction of Bregman geometry. Given a convex function $F$, we can define two objects:

**Bregman energy surface**: $\mathcal{E}_F = \{ (p, F(p)) : p \in \text{dom}(F) \}$. This is the graph of $F$ — an $n$-dimensional surface embedded in $\mathbb{R}^{n+1}$. On this surface, every probability distribution has not only a coordinate position, but also an "elevation" assigned by $F$.

**Bregman tangent bundle**: At every point $q$, we can draw the tangent hyperplane of $F$ at that point: $T_q = \{ (p, F(q) + \langle \nabla F(q), p - q \rangle) : p \in \mathbb{R}^n \}$. The Bregman divergence $D_F(p\|q)$ is precisely the vertical distance from the $T_q$ plane to the $\mathcal{E}_F$ surface at $p$.

This construction is called "entropy topography" because it does something very profound: **it uses the graph of $F$ as the "absolute elevation," the tangent plane as the "local sea level," and the Bregman divergence is the relative height measured above the local sea level.** Different $q$ have different local sea levels — this explains the asymmetry. There is no globally unified zero elevation — this explains why Bregman divergence is not a distance.

This entropy topography construction also directly yields the generalized Pythagorean theorem for Bregman divergence. Given three points $p, q, r$, if $q$ and $r$ satisfy a certain "orthogonality condition" (i.e., $q$ is the projection of $p$ onto some Bregman ball), then:

$$D_F(p \| r) = D_F(p \| q) + D_F(q \| r)$$

This looks like the Pythagorean theorem — the Bregman divergence of the "hypotenuse" equals the sum of the Bregman divergences of the two "legs." But in Euclidean space, the Pythagorean theorem requires a right angle. In Bregman geometry, "right angle" is redefined — not as a zero vector dot product, but as **Bregman orthogonality**: $\langle \nabla F(q) - \nabla F(r), q - p \rangle = 0$.

This generalized Pythagorean theorem is not an aesthetic decoration. It is the **geometric foundation that allows the Banach contraction mapping principle to be applied to KL divergence** — as we will see in the Yonglin Limit, the reason the KL divergence decrease at each step of reasoning can be chained into an inevitably convergent sequence is precisely because KL divergence satisfies the Bregman generalized Pythagorean structure.

:::info

**The generalized Pythagorean theorem: why it is the heart of all convergence proofs**

In Euclidean space, the Pythagorean theorem $c^2 = a^2 + b^2$ requires two perpendicular legs. Perpendicularity means that two directions "cannot see each other's projection" — moving along one direction does not change your coordinate along the other.

In Bregman geometry, "right angle" is redefined as **Bregman orthogonality**: $\langle \nabla F(q) - \nabla F(r), q - p \rangle = 0$. This condition does not say that the Euclidean angle is 90°, but rather: the direction along the Bregman geodesic from $r$ to $q$, and the direction from $p$ to $q$, "do not interfere with each other" under the curvature of $F$.

Why is this crucial for the Yonglin Limit? Because when you take a step from $p_t$ (current belief) to $p_{t+1}$, then another step to $p_{t+2}$... each step follows the gradient direction of the KL entropy landscape. If each step is Bregman-orthogonal to the target direction, then the KL divergence decrease at each step can be **summed** like the Pythagorean theorem — the total decrease equals the sum of the decreases of the individual steps. This means you will not "retrace your steps" — each step brings you closer to the target, and the amount by which you get closer can be precisely calculated.

If the Bregman generalized Pythagorean theorem did not hold — if the KL divergence decrease along the update direction could not be summed — then even if KL decreases at each step, you could not guarantee that the overall sequence converges. You might spiral in a vortex — descending at each step, but never reaching the lake's center. The generalized Pythagorean theorem guarantees: as long as each step advances in the correct Bregman-orthogonal direction, the descent is cumulative, and convergence is inevitable.

:::

## 5.4 KL Divergence: When $F$ is Negative Entropy

Let us make $F$ concrete. Take $F(p) = \sum_i p_i \log p_i$ — the negative of Shannon entropy (note: it is **negative** entropy, because entropy itself is a concave function, and we need $F$ to be convex). Then $\nabla F(p)_i = 1 + \log p_i$. Substituting into the definition of Bregman divergence:

$$D_F(p \| q) = \sum_i p_i \log p_i - \sum_i q_i \log q_i - \sum_i (1 + \log q_i)(p_i - q_i)$$

Simplifying (using $\sum_i p_i = \sum_i q_i = 1$):

$$D_F(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} = D_{\mathrm{KL}}(p \| q)$$

This is KL divergence. **KL divergence is not some arbitrarily defined "distribution difference measure" — it is the Bregman divergence generated by the negative entropy function.** Its geometric properties are not externally imposed, but intrinsically determined by the convexity of negative entropy.

![KL divergence as an asymmetric surface. Steep on one side, flat on the other—$D_\mathrm{KL}(p\|q)\neq D_\mathrm{KL}(q\|p)$ is geometric, not a bug.](/figures/ch05_kl_surface_tikz.svg)

This observation fundamentally changes our understanding of KL divergence. Textbooks usually say "KL divergence is asymmetric, therefore it is not a true distance." But the Bregman perspective tells us: the asymmetry is precisely the nature of KL divergence as an energy difference. $D_{\mathrm{KL}}(p\|q)$ measures — when using distribution $q$ to encode samples from distribution $p$, how many extra bits are used on average per symbol (compared to the optimal encoding using $p$ itself). The information waste from $q$ to $p$ and the information waste from $p$ to $q$ were never the same thing to begin with.

:::info

**The "coding" intuition: why KL is "extra bits used"**

A fundamental fact in information theory: if the true distribution of the source is $p$, but you use distribution $q$ to design an optimal code (Huffman coding or arithmetic coding), the average number of extra bits used per symbol is exactly $D_{\mathrm{KL}}(p\|q)$.

For example. Suppose $p = (0.9_{\text{sunny}}, 0.1_{\text{rainy}})$ — you live in a place that is almost always sunny. You design a coding scheme that uses the shortest codeword for "sunny." This is the optimal code using $p$ — on average $H(p) = -0.9\log_2 0.9 - 0.1\log_2 0.1 \approx 0.47$ bits per symbol.

Now your friend visits from London. London weather is $q = (0.5, 0.5)$. He brings his "London coding scheme" to your city — allocating equal-length codewords to "sunny" and "rainy." Using the London scheme to encode weather reports in your city costs an extra $D_{\mathrm{KL}}(p\|q) = 0.9\log_2(0.9/0.5) + 0.1\log_2(0.1/0.5) \approx 0.53$ bits per symbol on average.

Conversely, if you bring your "sunny scheme" to London, the extra bits used are $D_{\mathrm{KL}}(q\|p) = 0.5\log_2(0.5/0.9) + 0.5\log_2(0.5/0.1) \approx 0.74$ bits — 40% more than the former. The reason is clear: using a code that almost only recognizes "sunny" to handle reports that are half "rainy" wastes more than using a code that recognizes both "sunny" and "rainy" to handle reports that are almost only "sunny."

**The essence of KL asymmetry is: the cost of using $A$ to explain $B$ is not equal to the cost of using $B$ to explain $A$.** In belief space, this corresponds exactly to what we want: going from wrong to correct, and going from correct to wrong, were never symmetric operations.

:::

Another reason KL divergence is naturally suitable for belief space is its **local quadratic approximation**. Performing a Taylor expansion near $q$:

$$D_{\mathrm{KL}}(p \| q) \approx \frac{1}{2} \sum_i \frac{(p_i - q_i)^2}{q_i}$$

The coefficients of this quadratic form are $1/q_i$ — in regions where $q_i$ is close to 0, the same absolute difference is enormously magnified. This is exactly what we mentioned at the opening: moving probability from 0.99 to 1.0, versus from 0.5 to 0.51, the information-theoretic "distance" is worlds apart. The local curvature of KL divergence automatically and correctly captures this non-uniform sensitivity.

![The asymmetry of KL divergence](/figures/ch05_kl_asymmetry.svg)

*KL divergence between two Gaussian distributions. $D_{\mathrm{KL}}(p\|q) = 2.45$ vs $D_{\mathrm{KL}}(q\|p) = 10.45$ for $p$ (narrow peak, orange) and $q$ (broad peak, blue) — a difference of more than fourfold. Left: $D_{\mathrm{KL}}(p\|q)$: in regions where $p$ concentrates (orange highlight), $q$'s probability density is small; the penalty comes from the expectation of $\log(p/q)$ under $p$. Right: $D_{\mathrm{KL}}(q\|p)$: in regions where $q$ spreads (blue highlight), $p$ is nearly zero, and $\log(q/p)$ explodes — the "breadth" of $q$ is extremely penalized in the face of narrow $p$. The asymmetry is not a defect — it reflects the irreversibility of the information coding direction.*

## 5.5 Fisher Information and Natural Gradient

The local quadratic approximation of KL divergence directly leads to another core concept: the Fisher information matrix.

Near $q$, the Hessian matrix of $D_{\mathrm{KL}}(p\|q)$ (the second derivative with respect to $p$, evaluated at $p=q$) is precisely the Fisher information matrix:

$$G(q)_{ij} = \mathbb{E}_{x \sim q}\left[ \frac{\partial \log q(x)}{\partial \theta_i} \frac{\partial \log q(x)}{\partial \theta_j} \right]$$

The Fisher information matrix is the **Riemannian metric** on the space of probability distributions — it is the local curvature of KL divergence at every point, and like $g_{ij}$ in Riemannian geometry, it defines the "true" distance and angle in probability space.

This leads to the **natural gradient**. Ordinary gradient descent moves along the Euclidean steepest descent direction in parameter space. But Euclidean distance in parameter space has no probabilistic meaning — a parameter changing from 0.5 to 0.51, and from 0.99 to 1.0, differ by the same amount in parameter space, but differ completely in distribution space.

Natural gradient corrects this. It moves along the **steepest descent direction in distribution space**:

$$\theta_{t+1} = \theta_t - \eta G(\theta_t)^{-1} \nabla L(\theta_t)$$

Multiplying the Euclidean gradient $\nabla L$ by the inverse of the Fisher information $G^{-1}$ is equivalent to straightening out the "skew" in parameter space — so that each step has the same "effective step length" in distribution space (rather than parameter space). This is completely parallel to what we discussed in Chapter 1: using a fixed Euclidean step size in a curved parameter space is like measuring a map on a sphere with a straight ruler — natural gradient uses the metric tensor to correct the direction, so that the step length remains consistent in the local geometry.

### 5.5.1 Information Geometry: Why All of This Is Inevitable

The Fisher information matrix, the natural gradient, mirror descent — they look like three independently discovered techniques. But they are not. They are three facets of a single unified mathematical framework. This framework is called **Information Geometry**, established by Shun-ichi Amari in the 1980s.

The core insight of information geometry is remarkably concise: **the space of probability distributions is not a flat vector space — it is a manifold with a natural Riemannian metric.** Each point on this manifold is a probability distribution $p$. The Fisher information matrix $G(p)$ is the Riemannian metric of this manifold — it defines the "true" distance between two points on the manifold: not the Euclidean difference of their parameter coordinates, but the length of the geodesic along the manifold's curvature.

But the depth of information geometry goes further. It reveals that this manifold possesses a rare mathematical structure: **dual flatness.**

On the same probability manifold, two different sets of "straight lines" (geodesics) can be defined. $m$-geodesics are linear combinations of probability distributions — "mixing" two distributions, like pouring two colors of paint together. $e$-geodesics are straight lines in the natural parameter space of exponential families — changing from one distribution to another "exponentially." These two sets of geodesics generally do not coincide — unless the manifold is flat. When they do not coincide, the manifold acquires a "curved" structure.

Paired with dual flatness is a family of differential-geometric objects called **$\alpha$-connections**. $\alpha = 1$ corresponds to the $e$-connection (exponential connection), $\alpha = -1$ to the $m$-connection (mixture connection), and $\alpha = 0$ to the Levi-Civita connection (the self-dual connection under the Fisher-Rao metric). Different $\alpha$ define different ways of "parallel transport" — under different connections, "transporting a vector $v$ from $p$ to $q$ without changing its direction" means entirely different things.

In this framework, the Bregman divergence acquires its ultimate positioning: **every Bregman divergence is generated by a pair of dually flat manifold structures.** Specifically, take a convex function $F$; it induces an $e$-flat structure on the primal manifold, and its convex conjugate $F^*$ induces an $m$-flat structure on the dual manifold. $D_F(p\|q)$ precisely measures the "degree of misalignment" between these two flat structures.

When $F$ is the negative entropy, the $e$-flat structure corresponds to exponential family distributions (Gaussian, Bernoulli, softmax), and the KL divergence $D_{\mathrm{KL}}(p\|q)$ is precisely the Bregman divergence between these two dually flat structures. This is why, in probability space, KL divergence is not "one choice among many" — it is the "natural divergence" **uniquely determined** by the intrinsic geometry of this manifold.

Information geometry provides a unified mathematical foundation for the four core concepts of this chapter:

- **Bregman divergence** (§5.1–§5.3) → arises naturally from the convexity of $F$ on a dually flat manifold.
- **KL divergence** (§5.4) → the special case when $F$ is the negative entropy; the natural divergence of exponential families.
- **Fisher information matrix** (§5.5) → the Riemannian metric of the probability manifold; the local quadratic form of KL divergence.
- **Mirror descent** (§5.6) → a geometric algorithm that alternates projection between the $e$-flat and $m$-flat structures.

This entire construction is not artificially assembled. It is the inevitable structure that emerges from the natural unfolding of differential geometry, starting from the single premise that "probability distributions form a manifold." You do not need to "choose" Bregman divergence to measure belief differences — the curvature of the manifold chooses it for you.

:::info

**The Pallas's Cat Professor: Amari Did Not Invent Information Geometry — the Probability Manifold Always Had This Structure**

Many people first encountering information geometry think: this is a beautiful mathematical framework constructed by Amari. But strictly speaking, Amari did not **invent** information geometry — he **discovered** it.

The space formed by probability distributions inherently possesses dual flat structure and dual $\alpha$-connections. No one "added" them — just as the curvature of the Earth's surface was not "added" by Gauss. Gauss measured the sum of the interior angles of a triangle atop the mountains of Hanover and found it exceeded 180° — he was not inventing spherical geometry; he was discovering the curvature of the Earth.

Similarly, when you perform gradient descent in belief space, whether each step you take is effective is not determined by the coordinate system you choose — it is determined by the Fisher metric of the probability manifold. You can walk in a straight line in parameter space, but the manifold bends your path. Natural gradient, mirror descent, the Yonglin Limit — none of them are "tricks." They are the only internally consistent ways of walking on a curved probability manifold.

This book has told a story from ch1 to ch5: from Newton's force to Hamilton's energy, from the loss terrain to the entropy terrain. Information geometry is the final chapter of this story in the 20th century: **space is not flat. Probability space is even less so.** You do not need to impose a metric — the metric is already there. You only need to learn how to measure it.

:::

## 5.6 Mirror Descent: Descending in the Bregman Landscape

Natural gradient requires computing the inverse of the Fisher information matrix — which is prohibitively expensive in high dimensions. But Bregman divergence offers another path that does not require explicitly computing $G^{-1}$: **mirror descent**.

The core idea of mirror descent is: do not perform gradient descent in the original parameter space, but instead perform it in the **dual space** (the mirror space). The process has three steps:

1. **Map to the dual space**: Map the current parameter $\theta_t$ via $\nabla F$ to the dual point $\mu_t = \nabla F(\theta_t)$.
2. **Descend in the dual space**: Take an ordinary gradient step in the dual space: $\mu_{t+1} = \mu_t - \eta g_t$ (where $g_t$ is the gradient of the loss function).
3. **Map back to the original space**: Via $\nabla F^*$ (the gradient of the convex conjugate of $F$), map $\mu_{t+1}$ back to $\theta_{t+1} = \nabla F^*(\mu_{t+1})$.

The composite update of these three steps is equivalent to solving:

$$\theta_{t+1} = \arg\min_\theta \left\{ \langle g_t, \theta \rangle + \frac{1}{\eta} D_F(\theta \| \theta_t) \right\}$$

This formula is the elegant core of mirror descent: **the new parameter $\theta_{t+1}$ found at each step must both make progress in the direction of the loss gradient (first term), and cannot stray too far from the current position — where "far" is measured by Bregman divergence (second term).**

When $F(\theta) = \frac{1}{2}\|\theta\|^2$, $D_F$ degenerates to Euclidean distance, and mirror descent degenerates to ordinary gradient descent. When $F$ is negative entropy, $D_F$ is KL divergence, and mirror descent becomes **steepest descent along KL geometry on the probability simplex**.

The true power of mirror descent lies in this: it automatically incorporates the geometry of the parameter space (encoded by the Hessian of $F$) into every update step. Near the edges of the probability simplex, the curvature of KL divergence is extremely high — the same Euclidean step length corresponds to a huge jump in distribution space. Mirror descent automatically reduces the step size in these regions through the regularization of $D_F$. This is precisely the **geometric origin of $\eta_{\max}(p)$** — which we will formalize precisely soon.

![Mirror descent vs Euclidean descent: trajectory comparison on the simplex](/figures/ch05_mirror_descent.svg)

*Optimization trajectories on the probability simplex (triangle). Starting point is the uniform distribution (center), target is $(0.7, 0.2, 0.1)$ (square). Euclidean projected descent (left, vermilion) moves toward the target along a straight line, but requires projection back to the simplex at the boundary. Mirror descent (right, green) advances along the natural gradient direction of KL geometry — automatically decelerates at the simplex edges, with the trajectory smoothly converging to the target. Stars mark final positions.*

:::info

**Why is it called "mirror"? Intuition for the dual space**

The name "mirror descent" comes from a very vivid geometric structure. Imagine you are standing in front of a curved mirror — the you in the mirror is not in the same space as the real you. The real you is in the "primal space" (parameter space), the mirror you is in the "dual space" (gradient space).

Ordinary gradient descent takes steps directly in the primal space: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$. This is natural in Euclidean space, but in curved Bregman space, "walking straight" in the primal space does not correspond to walking straight in distribution space.

Mirror descent's approach is: **first walk in the mirror, then map back.**
- Step 1 (map into the mirror): $\mu_t = \nabla F(\theta_t)$. $\nabla F$ is the mapping from primal space to mirror space (dual space). In the mirror world, all operations are Euclidean — you can safely use ordinary gradient steps.
- Step 2 (walk in the mirror): $\mu_{t+1} = \mu_t - \eta g_t$. This is an ordinary, Euclidean gradient step — the mirror world is flat.
- Step 3 (map back): $\theta_{t+1} = \nabla F^*(\mu_{t+1})$. $\nabla F^*$ is the inverse mapping of $\nabla F$ — returning from the mirror world to the real world.

The magic is this: Step 2 walks an **ordinary Euclidean step length** $\eta$ in the mirror, but after two mappings ($\nabla F^* \circ$ (Euclidean step) $\circ \nabla F$), the effect produced in the primal space is equivalent to **automatically adapting to the curvature of $F$**. You do not need to explicitly compute the inverse of Fisher information — the convex conjugate of $F$ naturally accomplishes this task.

:::

## 5.7 The Probability Simplex: Softmax as Projection

Before discussing the Yonglin Limit, we need one final geometric tool: the structure of the **probability simplex**, and the role of **softmax as a projection operator**.

The probability simplex is the set of all possible discrete probability distributions:

$$\Delta^{K-1} = \left\{ p \in \mathbb{R}^K : p_i \geq 0, \sum_i p_i = 1 \right\}$$

This is a $(K-1)$-dimensional geometric object — embedded in $K$-dimensional space, but itself having only $K-1$ degrees of freedom (because the constraint $\sum p_i = 1$ consumes one dimension).

Any vector $z \in \mathbb{R}^K$ (logits — the raw outputs of the model) can be projected onto the simplex via the softmax function:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

From a geometric perspective, softmax does the following: **it projects an arbitrary point in $\mathbb{R}^K$, along a specific direction, onto the probability simplex.** This projection direction is not Euclidean (not a perpendicular line segment), but **Bregman** — specifically, it follows the Bregman geodesic direction generated by the negative entropy function $F(p) = \sum p_i \log p_i$.

This implies an important fact: **the model's forward pass — logits → softmax → probability distribution — inherently carries the structure of KL geometry.** It is not an arbitrarily chosen normalization operation, but the natural projection in negative-entropy Bregman geometry.

This is closely related to the coordinate invariance of $\eta_{\max}(p)$. Because softmax is a Bregman projection, any Euler step taken in parameter space — $\theta_{t+1} = \theta_t + \eta \cdot \text{(some direction)}$ — when projected onto the simplex via softmax, has an "effective magnitude" that depends on where you are on the simplex. Near the edges (where some $p_i$ is small), the same parameter change causes a larger distribution change — hence, the safe step size is smaller. $\eta_{\max}(p)$ precisely quantifies this position dependence.

![Belief dynamics on the probability simplex. $q_0\to q_1\to q_2\to q^*$: reasoning as a trajectory in belief space, not token space.](/figures/ch05_belief_simplex_tikz.svg)

## 5.8 The Yonglin Limit I: Energy Function and Reasoning Operator

Now we can enter the theoretical core of this chapter — the Yonglin Limit.

The question answered by the Yonglin Limit is: **in belief space, does reasoning necessarily converge?** Not "usually so," not "empirically so," but "starting from any initial belief, under an appropriate step size, Euler-step reasoning must converge to a unique fixed point."

The proof of this proposition follows a clear mathematical chain. We break it into three parts.

**Step One: Define the energy function.**

In belief space, we need a quantity that can serve as "terrain elevation" — analogous to the role played by the loss function $L(\theta)$ in parameter space in Chapter 3. This quantity must satisfy two conditions: it must monotonically decrease during the reasoning process, and it must attain its minimum at the fixed point.

KL divergence is the natural candidate. Given a target distribution (the empirical distribution of the training data) $p^*$, define the energy function as the KL divergence between the model's current belief $p_t$ and $p^*$:

$$E(p_t) = D_{\mathrm{KL}}(p^* \| p_t) = \sum_i p^*_i \log \frac{p^*_i}{p_{t,i}}$$

(Note the direction of KL here is $p^*\|p_t$ — the extra cost of using the current model to "explain" the true data. This choice of direction is not arbitrary; it ensures the natural validity of the subsequent contractivity proof.)

$E(p_t) \geq 0$, and $E(p_t) = 0$ if and only if $p_t = p^*$. This satisfies the two basic conditions of a Lyapunov function (Chapter 3) — nonnegative everywhere, zero at the target.

**Step Two: Construct the Euler-step reasoning operator.**

The reasoning process is modeled as the discrete-time evolution of belief states on the simplex. At each step, the model, based on the current belief $p_t$ and input $x$, produces an update direction, and then takes a step of size $\eta$ along this direction:

$$p_{t+1} = \Phi_\eta(p_t) = \text{softmax}(\text{logit}(p_t) + \eta \cdot \text{update}(p_t, x))$$

Here the "update" direction can come from the gradient of the loss function, the output of an attention mechanism, or any other reasoning module. The key abstraction is: $\Phi_\eta$ is a map on the simplex, and $\eta$ is its step size.

This is the reasoning operator. Given the current belief and step size, it outputs the next belief. The entire reasoning process is the repeated application of $\Phi_\eta$:

$$p_0 \to p_1 \to p_2 \to \cdots \to p_T$$

![Belief update cycle computation graph](/figures/ch05_belief_update_graph_tikz.svg)

*Belief update cycle computation graph. q_t is transformed by operator T into q_{t+1}. The KL divergence check D_KL(q*||q_{t+1})<D_KL(q*||q_t) decides: continue iterating or converged to fixed point q*.*

**Step Three: Prove contractivity.**

The pivot of the Yonglin Limit is: $\Phi_\eta$ is a **contraction mapping** in the sense of KL divergence — each step brings the model closer to $p^*$, and this "closing in" has a definite lower bound.

Formally, there exists a constant $\gamma \in (0, 1)$ such that for all $p \neq p^*$:

$$E(\Phi_\eta(p)) \leq \gamma \cdot E(p)$$

Or equivalently:

$$D_{\mathrm{KL}}(p^* \| \Phi_\eta(p)) \leq \gamma \cdot D_{\mathrm{KL}}(p^* \| p)$$

The energy decays by at least a factor of $\gamma$ at each step. This is not a vague "energy decreases" — it is a definite contraction.

## 5.9 The Yonglin Limit II: Complete Proof of Contractivity

**Step Four: Prove contractivity.**

Now we present the complete proof of contractivity. This is the mathematical heart of the Yonglin Limit.

**Theorem**: Let the energy function $E(p) = D_{\mathrm{KL}}(p^*\|p)$ be $\mu$-strongly convex ($\mu > 0$), and let its gradient $\nabla E$ be $L$-smooth ($L > 0$). Take step size $\eta \in (0, 2\mu/L^2)$. Then the Euler-step reasoning operator $\Phi_\eta(p) = \operatorname{proj}_{\mathcal{P}}(p - \eta \nabla E(p))$ is a contraction mapping under KL divergence.

**Proof**. Let $q_1 = \Phi_\eta(p_1)$, $q_2 = \Phi_\eta(p_2)$. We need to prove there exists $\gamma < 1$ such that $D_{\mathrm{KL}}(q_1 \| q_2) \leq \gamma \cdot D_{\mathrm{KL}}(p_1 \| p_2)$.

**Step One: Apply the Bregman three-point identity.**

Recall §5.3: KL divergence, as the negative-entropy Bregman divergence, satisfies the three-point identity. Take the three points as $p_1, q_1, q_2$:

$$D_{\mathrm{KL}}(p_1 \| q_2) = D_{\mathrm{KL}}(p_1 \| q_1) + D_{\mathrm{KL}}(q_1 \| q_2) - \langle \nabla\phi(q_2) - \nabla\phi(q_1),\, p_1 - q_1 \rangle$$

where $\phi(p) = \sum_i p_i \log p_i$ (negative entropy). Rearranging:

$$D_{\mathrm{KL}}(q_1 \| q_2) = D_{\mathrm{KL}}(p_1 \| q_2) - D_{\mathrm{KL}}(p_1 \| q_1) + \langle \nabla\phi(q_2) - \nabla\phi(q_1),\, p_1 - q_1 \rangle$$

The first two terms are given by divergences, the third term is an inner product — we need to expand it.

**Step Two: Expand the inner product term.**

From $q_i = \operatorname{proj}_{\mathcal{P}}(p_i - \eta \nabla E(p_i))$ and the first-order optimality condition for projection onto the simplex, the inner product term can be expressed using the gradient difference of $E$. Exploit two key properties of $E$:

- **$\mu$-strong convexity**: $\langle \nabla E(p_1) - \nabla E(p_2),\, p_1 - p_2 \rangle \geq \mu \|p_1 - p_2\|^2$. Strong convexity means that the energy function has curvature at least $\mu$ in every direction — this is the guarantee that "the terrain has sufficient slope."

- **$L$-smoothness**: $\|\nabla E(p_1) - \nabla E(p_2)\| \leq L\|p_1 - p_2\|$. Smoothness means that the gradient does not change too drastically — this is the guarantee that "the terrain does not suddenly present cliffs." The negative contribution of smoothness (the upper bound of the inner product) is at most $\eta^2 L^2 \|p_1 - p_2\|^2$.

Combining the two, the exact contribution of the inner product term is:

$$\langle \cdots \rangle = \eta(2\mu - \eta L^2)\|p_1 - p_2\|^2$$

**Step Three: Obtain the contraction inequality.**

Substituting back, and using $D_{\mathrm{KL}}(p_1\|q_2) \leq D_{\mathrm{KL}}(p_1\|p_2)$ (projection onto the simplex does not increase KL divergence — this is a fundamental property of Bregman projection):

$$D_{\mathrm{KL}}(q_1 \| q_2) \leq D_{\mathrm{KL}}(p_1 \| p_2) - \eta(2\mu - \eta L^2)\|p_1 - p_2\|^2$$

When $\eta < 2\mu/L^2$, we have $2\mu - \eta L^2 > 0$, so the second term on the right is strictly positive — the divergence strictly shrinks.

Using the reverse form of Pinsker's inequality (KL divergence and Euclidean norm have local equivalence on the simplex, there exists a constant $C > 0$):

$$D_{\mathrm{KL}}(q_1 \| q_2) \leq \underbrace{\left(1 - \frac{\eta(2\mu - \eta L^2)}{C}\right)}_{k(\eta) < 1} \cdot D_{\mathrm{KL}}(p_1 \| p_2)$$

This is the contraction inequality. **$2\mu - \eta L^2$ is the heart of the contraction**: strong convexity provides the positive contribution $2\mu$, smoothness provides the negative contribution $\eta L^2$, and the difference between the two determines the magnitude of contraction at each step. The step size upper bound $\eta < 2\mu/L^2$ is precisely the condition that guarantees this difference is positive.

**QED.**

**Step Five: Invoke the Banach fixed-point theorem.**

Since $\Phi_\eta$ is a contraction mapping ($k(\eta) < 1$), and the probability simplex is complete under KL divergence, the Banach fixed-point theorem directly yields:

1. **There exists a unique** fixed point $p^*$ satisfying $\Phi_\eta(p^*) = p^*$
2. For any initial belief $p_0$, the iteration $p_{t+1} = \Phi_\eta(p_t)$ converges to $p^*$
3. Convergence rate: $D_{\mathrm{KL}}(p_t \| p^*) \leq k(\eta)^t \cdot D_{\mathrm{KL}}(p_0 \| p^*)$

This is the core assertion of the Yonglin Limit: **reasoning in belief space necessarily converges, and the convergence rate is guaranteed to be exponential by $k(\eta)$.** Not "usually" converges, not "is well-trained" — but rather the Bregman geometric three-point identity, the strong convexity and smoothness of the energy function, and Banach's fixed-point theorem, all three jointly guarantee the inevitability of convergence.

:::info

**Banach fixed point: if every step moves you closer, the destination must exist**

The intuition behind the Banach fixed-point theorem is actually extremely simple. Imagine you are standing in a room. You are asked to perform the following action: take a step toward the exact center of the room, with a step length equal to 90% of the distance from your current position to the center. Repeat this action.

On the first step, you cover 90% of the distance. On the second step, you cover 90% of the remaining distance. On the third step... No matter which corner of the room you start from, you are guaranteed to approach the exact center arbitrarily closely — because each step compresses the remaining distance to 10% of its previous value (i.e., contraction factor $\gamma = 0.1$). You do not even need to know where the center is — you only need to know that each step brings you closer to it, and the proportion by which it does so is fixed.

This is the entirety of the Banach contraction mapping principle. It demands only three things:
1. You have a notion of "distance" — in the Yonglin Limit, this is KL divergence (in the fixed direction $p^*\|p$).
2. Your mapping compresses the distance at each step — in the Yonglin Limit, $D_{\mathrm{KL}}(p^*\|\Phi_\eta(p)) \leq \gamma \cdot D_{\mathrm{KL}}(p^*\|p)$ with $\gamma < 1$.
3. The space is "complete" — every Cauchy sequence has a limit. The probability simplex under KL divergence satisfies this condition.

If these three hold, Banach's theorem guarantees: the fixed point exists uniquely, and no matter where you start, you will arrive there. **You do not need to "hope" for convergence — you prove convergence.** This is the mathematical heart of the Yonglin Limit.

:::

## 5.10 The Yonglin Limit III: Derivation of η_max(p) and Coordinate Invariance

The step size range for which contractivity holds is not arbitrary. $\eta$ must be sufficiently small to ensure that the contraction factor $\gamma < 1$ at each step. But how small is "sufficiently small"? This value depends on the current position $p$ — because the curvature of different regions of the probability simplex differs.

**Derivation.** The contraction mapping requires $\eta < 2\mu / L^2$, where $\mu$ is the strong convexity constant of the energy function, and $L$ is the smoothness constant. These two constants are not mysterious numbers — they are directly encoded in the Hessian matrix $\nabla^2 E(p)$.

For the cross-entropy loss $E(p) = \mathbb{E}_D[-\log p_y]$, the Hessian on the probability simplex has an analytic form:

$$\nabla^2 E(p) = \text{diag}\left(\frac{1}{p_i}\right)$$

This is a diagonal matrix — each diagonal element is $1/p_i$. In directions where $p_i$ is large (classes the model is confident about), the curvature is small and the terrain is flat; in directions where $p_i$ is small (classes the model almost rules out), the curvature is large and the terrain is steep.

The strong convexity constant $\mu$ is the minimum eigenvalue of the Hessian — corresponding to the flattest direction. On the simplex, the flattest direction is the dimension with the largest probability:

$$\mu = \frac{1}{\max_i p_i}$$

The smoothness constant $L$ is the maximum eigenvalue of the Hessian — corresponding to the steepest direction. The steepest direction is the dimension with the smallest probability:

$$L = \frac{1}{\min_i p_i}$$

Substituting into the Banach contraction step size condition $\eta < 2\mu / L^2$:

$$\eta_{\max}(p) = \frac{2\mu}{L^2} = \frac{2 / \max_i p_i}{(1 / \min_i p_i)^2} = 2 \cdot \frac{\min_i p_i^2}{\max_i p_i}$$

Each term in this formula has a clear geometric meaning:

- **$\min_i p_i^2$ in the numerator**: the dimension the model is most uncertain about. If the model almost completely rules out some class ($p_i \approx 0.01$), then $\min_i p_i^2 = 10^{-4}$ — the step size upper bound is squeezed to an extremely small value. Geometric reason: KL divergence at the simplex edge has curvature proportional to $1/p_i$ (diagonal elements of the Hessian), and the same absolute change produces a huge distributional difference at the edge. The square comes from $L^2 = 1/\min_i p_i^2$ — the square of the smoothness constant amplifies the instability of the edge.

- **$\max_i p_i$ in the denominator**: the dimension the model is most confident about. If the model is highly confident about some class ($p_i \approx 0.98$), then $1/0.98 \approx 1$ provides the foundational strong convexity — it ensures that even in this flattest direction, the energy function still has sufficient curvature to guide the descent.

- **Coordinate invariance**: $\eta_{\max}(p)$ depends only on the position of $p$ on the simplex — it does not depend on whether probability values, logits, or some other parameterization is used. $\mu$ and $L$ are both eigenvalues of the Hessian, which are geometric quantities. **The terrain determines the stride; the data bias determines the terrain.** This step size upper bound can be directly computed from the current model output — no empirical hyperparameter tuning is needed.

In practical reasoning, this means: when the model is already highly confident ($p$ is sharp), it can safely use a larger step size — because it is then in a "flat region" of the simplex. When the model is highly uncertain ($p$ is uniform), it can also use a relatively large step size — because the central region has moderate curvature. But when the model is at a "critical edge" — where the probability of one or more dimensions is extremely small — it must take steps with extreme caution. This exactly corresponds to the psychological intuition of reasoning: ruling out a seemingly impossible option requires greater "caution" than adjusting two nearly equally probable options.

![Distribution of $\eta_{\max}(p)$ on the probability simplex](/figures/ch05_eta_max_simplex.svg)

*Contour lines of $\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ on the three-class simplex. Red region (near the uniform distribution): $\eta_{\max} \approx 0.22$, step size can be relatively large. Dark region (near vertices, i.e., high confidence): $\eta_{\max}$ is moderate. White/light region (near edges, i.e., some $p_i \approx 0$): $\eta_{\max}$ shrinks drastically — at the simplex edge, KL curvature is highest, and the safe step size upper bound drops to the order of $10^{-5}$. This is a coordinate-invariant geometric quantity: no matter what parameterization is used, $\eta_{\max}(p)$ depends only on the position of $p$ on the simplex.*

:::info

**Why do edges require extremely small step sizes? — Numerical intuition for $\eta_{\max}$**

Consider a three-class problem. Your current belief is $p = (0.98, 0.01, 0.01)$ — you are almost completely confident about the first class. Then $\eta_{\max} = 2 \times 0.01^2 / 0.98 \approx 0.0002$.

Why so small? Because the third class $p_3 = 0.01$ is already extremely small. If your Euler step in some direction accidentally takes too large a step along the $p_3$ direction, $p_3$ could become negative — which is illegal in probability space. The extremely small value of $\eta_{\max}$ reflects the "geometric crowding" at the simplex edge — the space is too narrow, and taking a large step risks hitting the wall.

Contrast this with the uniform distribution $p = (1/3, 1/3, 1/3)$. Then $\eta_{\max} = 2 \times (1/3)^2 / (1/3) = 2/9 \approx 0.22$. The space is wide — there is sufficient room in all directions.

The reason the numerator $\min(p_i)^2$ is squared comes from the **quadratic-form amplification** of KL divergence at the edges. Recall $D_{\mathrm{KL}}(p\|q) \approx \frac{1}{2}\sum (p_i-q_i)^2/q_i$ — when some $q_i$ is very small, the same absolute change $(p_i-q_i)$ is amplified by $1/q_i$. This in turn constrains $\eta$: to guarantee contractivity $\gamma < 1$, the step size must shrink in proportion to the reciprocal of this amplification factor. $\min(p_i)^2$ is exactly the order-of-magnitude expression of this synchronous shrinking.

This is also why $\eta_{\max}$ is a **genuine geometric quantity**: it has no free parameters, does not depend on the choice of coordinate system, and depends only on the position of $p$ on the simplex. Any reasoning model — whether a Transformer, an LSTM, or a human — as long as it updates beliefs along KL geometry on the probability simplex, will be subject to the same $\eta_{\max}(p)$ constraint.

:::

## 5.11 The Philosophy of the Yonglin Limit: Why Prove Convergence Instead of Assuming It?

Now let us step back from the mathematics and ask a deeper question: why is the Yonglin Limit important?

In the daily practice of the deep learning community, "convergence" is almost always treated as an empirical phenomenon — "loss went down," "accuracy improved," "training completed." We rarely ask: **why does it converge? Could it possibly not converge? Under what conditions is convergence guaranteed?**

This attitude is viable in engineering but fragile in theory. When you do not know why a system converges, you also do not know when it will fail to converge. When training diverges — gradient explosion, loss NaN, model collapse — your toolbox only has "lower the learning rate," "change the initialization seed," "try a different optimizer." These are empirical patches, not theoretical insights.

What the Yonglin Limit attempts to do is, in the specific context of belief space — how belief distributions update during reasoning — to give a **convergence proof that does not depend on empirical hyperparameter tuning**. Its philosophical stance can be summarized in three sentences:

**First, do not assume contraction — prove contraction.** We do not say "reasoning probably converges." We find a concrete Lyapunov function (KL divergence), prove that it monotonically decreases at each reasoning update step, and that the decrease has a controllable lower bound. Contraction is not a wish — it is the conclusion of a theorem.

**Second, choose the natural metric rather than impose a metric.** Belief space has its own geometry — KL divergence is not something we choose to "measure" belief differences; it is the Bregman divergence of the negative entropy function — the intrinsic geometry of belief space. Proving contractivity in Euclidean space is clumsy (you need to impose unnatural assumptions); in KL geometry, contractivity is almost self-evident — because it comes directly from the generalized Pythagorean structure of Bregman divergence and the curvature of the convex generating function.

**Third, let parameters have physical meaning.** $\eta_{\max}(p)$ is not an empirically tuned "optimal learning rate," but the step size upper bound determined by the local curvature of the simplex. $\gamma$ is not a mysterious "convergence rate hyperparameter," but the geometric expression of the Bregman contraction factor. $p^*$ is not "the optimal parameter on the training set," but the unique fixed point of the reasoning dynamical system under KL geometry. Every symbol corresponds to a clear geometric entity.

This is the philosophy of the Yonglin Limit: **the convergence of reasoning is not an engineering miracle, but a geometric necessity.** Euler steps in belief space, if the step size does not exceed the bound determined by the local curvature, must contract — and contraction means inevitable convergence. We do not need to "wait" for the model to converge, we do not need to "hope" it converges — from the very moment it takes its first step, we already know it will converge. This is the Lyapunov idea from Chapter 3, realized in its most complete and profound form in belief space.

:::info

**Professor Pallas's Cat's stance**

Many people, upon seeing the name "Yonglin Limit," think it is an engineering result — "Oh, you found the upper bound of the optimal learning rate."

It is not. $\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ is not a hyperparameter tuning trick. It is an **existence proof**. It says: reasoning in belief space does not converge because you trained it well — it converges because of the curvature of KL geometry. You do not need to tune the learning rate. You only need to ensure it does not exceed the upper bound prescribed by the terrain curvature.

This is the power of geometry. Geometry does not tune your hyperparameters — geometry tells you, under what conditions convergence is not a miracle, but an inevitability.

From Chapter 1 to here, this book has traced a long arc. From Newton's forces to Hamilton's energy, from loss landscapes to entropy topography, from Lyapunov to Banach, from Euler's method to DEQ — they are all different chapters of the same story. The core of this story is a single sentence: **Do not analyze forces. Find energy. When energy decreases, the system advances. When energy reaches zero, the system stops.** The language of forces makes you compute. The language of energy makes you understand.

:::

---

## 5.12 Chapter Summary

This chapter is the first peak of mathematical depth in the book. It accomplishes one thing: **from Euclidean to Bregman, from flat space to curved energy landscape, from the crow's straight line to the water's watershed.**

Bregman divergence $D_F(p\|q) = F(p) - F(q) - \langle \nabla F(q), p - q \rangle$ is the language for measuring irreversible energy differences on energy landscapes. It is not a distance — it is asymmetric, does not satisfy the triangle inequality — but this is precisely why it is naturally suitable for probability space. The error of estimating $p$ using the local geometry at $q$, and the error of estimating $q$ using the local geometry at $p$, are never equal on a curved landscape.

KL divergence is the Bregman divergence when $F$ is negative entropy — not an accidental coincidence, but the cornerstone of information geometry. The Fisher information matrix is the local quadratic form of KL divergence, natural gradient uses the inverse Fisher matrix to correct the direction, mirror descent descends in the dual space without explicitly computing the inverse — these techniques converge from different directions onto the same core: **in belief space, you must walk along the space's own curvature, rather than imposing Euclidean step sizes.**

The Yonglin Limit pushes this entire geometric language to its theoretical endpoint: from the energy function (KL divergence) to the Euler-step reasoning operator, from KL contractivity to the Banach fixed point, from the coordinate invariance of $\eta_{\max}(p)$ to the geometric proof that "reasoning inevitably converges." It is not merely a training technique — it is the first principle of the dynamics of reasoning in belief space.

The next chapter will systematize the dynamical systems perspective: learning is not just "parameters moving on a landscape," but more fundamentally a **state evolution of a dynamical system**. From discrete Euler steps to continuous gradient flow, from the dynamics of parameters to the dynamics of representations — we will lay the final foundation stone for the unified fixed-point theory of Chapter 7.

---

## Unresolved Questions

1. Bregman divergence is generated by a convex function $F$. Does there exist an "optimal" $F$ such that the Bregman divergence possesses theoretically optimal contraction properties for a certain class of learning tasks?

2. The negative entropy function generates KL divergence, which appears to be the "most natural" Bregman divergence on the probability simplex. But is this naturalness necessary or historical? Do there exist other convex functions $F$ whose generated Bregman divergences perform better on specific tasks?

3. The generalized Pythagorean theorem of Bregman divergence is the core geometric foundation of the Yonglin Limit. Under what conditions does this generalized Pythagorean theorem still hold — at least approximately — on the non-convex loss landscapes of deep networks?

4. Mirror descent performs gradient descent in the dual space, avoiding explicit inversion of the Fisher information matrix. In high-dimensional deep networks, can we design an approximate mirror descent that does not rely on explicitly computing $\nabla F^*$?

5. $\eta_{\max}(p) = 2\min(p_i)^2 / \max(p_i)$ is the safe upper bound for the reasoning step size. During training, $\eta_{\max}$ varies dynamically with $p$. Can we design an **adaptive step size schedule** — using the $\eta_{\max}(p)$ of the current position as the step size at each step — to guarantee the stability of the entire training process?

6. The Yonglin Limit proves that Euler-step reasoning converges under the condition that "the update direction points exactly to $p^*$." When the update direction comes from an imperfect reasoning module (such as the output of an attention layer), can contractivity still be proved? Will error accumulation lead to divergence?

7. The Yonglin Limit holds on the discrete simplex (finite categories). For continuous distributions (such as Gaussian distributions in Diffusion models), can the contractivity of KL divergence be extended to infinite-dimensional probability spaces?

8. The Banach fixed-point theorem requires the mapping to be contractive under some metric. The Yonglin Limit uses KL divergence as the metric. But KL divergence is asymmetric and strictly speaking is not a metric. Is it necessary — or possible — to reformulate the proof in a genuine metric space (such as Hellinger distance or total variation distance)?

9. The asymmetry of Bregman divergence plays a key role in the Yonglin Limit (the choice of direction $D_{\mathrm{KL}}(p^*\|p)$). If one instead uses $D_{\mathrm{KL}}(p\|p^*)$, does the contractivity proof still hold? What is the geometric difference between the two directions?

10. Natural gradient is theoretically elegant, but the cost of actually computing $G^{-1}$ is extremely high. Does there exist a class of loss functions or model architectures such that the Fisher information matrix has a particularly simple structure (such as diagonal, low-rank, Kronecker product), so that natural gradient can be efficiently approximated?

11. The convergence rate of the Yonglin Limit is determined by the Bregman contraction factor $\gamma$, which in turn depends on the step size $\eta$ and the local curvature. For a given convergence precision $\epsilon$, does there exist an optimal step size sequence $\eta_1, \eta_2, \ldots$ that minimizes the number of steps required for convergence?

12. If the "energy function" (KL divergence) of the Yonglin Limit can be understood as a kind of free energy in physics, does the reasoning process correspond to some thermodynamic process? Does stochastic gradient noise play the role of a "heat bath," helping the system explore energy barriers between different belief states?

---

**The core question left by this chapter is:**

**The generalized Pythagorean theorem of Bregman divergence — the geometric heart of the Yonglin Limit — in what form does it survive in nonlinear, high-dimensional, deep networks contaminated by stochastic gradient noise?**

:::info

**Euclidean is the crow's language — straight lines, symmetry, independent of terrain. Bregman is the language of water — flowing along energy gradients, asymmetric, the energy cost of each step dependent on the curvature underfoot.** KL divergence is the most important dialect in the language of water — it uses the topography of entropy to characterize all possible "flows" in belief space. The Yonglin Limit proves: as long as you follow the steepest descent direction on the KL landscape, the water will inevitably flow to the sea. In the next chapter, we will systematize the dynamical systems perspective — from "learning is motion" in Chapter 1, to the discrete Euler steps of Chapter 3, to the final unification of Chapter 7: ResNet, GPT autoregression, and DEQ are all faces of the same dynamical system in different spaces.

:::

## Further Reading and Related Work

**Relative Natural Gradient for Learning Large Complex Models.** Sun & Nielsen (2016) [arXiv:1606.06069] — Extracts local relative Fisher information measures from large neuronal systems, making natural gradient tractable in modular networks — not an approximation, but a rigorous characterization of the "locality" of information geometry.

**On the Locality of the Natural Gradient for Deep Learning.** Ay (2020) [arXiv:2005.10791] — The Fisher information matrix is fundamentally a local geometric quantity: it depends only on the second-order behavior of KL divergence near the current parameters. This seemingly trivial observation is the root reason natural gradients are computable in deep networks — and the most essential dividing line between Bregman divergences and Euclidean step sizes.

**Natural Gradient Methods: Perspectives, Efficient-Scalable Approximations, and Analysis.** Shrestha (2023) [arXiv:2303.05473] — A systematic survey comparing strategies for replacing the Hessian with the Fisher information matrix, from Kronecker factorization to low-rank approximations, spanning theoretical frameworks to engineering implementations.

**Generalized Euler Logarithm and its Applications in Machine Learning.** Cichocki (2025) [arXiv:2502.17500] — A two-parameter unification of the Bregman divergence family: the generalized Euler logarithm $E_{\alpha,\beta}(x)$ controls tail robustness and local gradient shaping separately, subsuming KL divergence, $\chi^2$ divergence, and $\alpha$-divergence under a single parametrized framework — a parameterization of the Bregman generator itself.

**Scaled-Dot-Product Attention as One-Sided Entropic Optimal Transport.** Litman (2025) [arXiv:2508.08369] — The forward pass of attention is the exact solution to a degenerate entropic optimal transport problem — each step of Transformer self-attention solves an entropy-regularized transport problem, with the Fisher information matrix defining the geometric curvature of the attention distribution.

**Diagonalizing the Softmax.** Garrod, Keating & Thrampoulidis (2025) [arXiv:2512.04006] — Hadamard initialization diagonalizes the softmax Fisher information matrix, and cross-entropy gradient flow provably converges to Neural Collapse geometry — the first rigorous result where information geometry directly explains deep learning training dynamics.
