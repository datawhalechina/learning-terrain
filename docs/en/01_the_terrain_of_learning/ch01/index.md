# Chapter 1: Why Learning Needs Geometry

If reasoning is not a single answer but a structural activity, then how should we characterize that structure?

When a model reads a problem, it doesn't suddenly spit out an answer from nowhere. It transforms the problem into a representation, sends the representation through layer after layer of transformations, organizes local clues into global judgments, and then compresses several possible answers into a final output. In this process, there is direction, there is position, there is distance, there is path, there is curvature, there is approach and withdrawal, there is stability and instability.

Taken together, these are no longer mere "computation." It resembles something more like geometry.

So-called geometry is not just about drawing triangles, circles, and coordinate axes. The deepest work of geometry is to characterize structure. It asks: Where are the objects? How far apart are they? What does a direction mean? How does a space curve? How is a motion constrained? How does a system go from one state to another?

If we view learning as a structural problem, then geometry is not decoration — it is a necessity.

But before we discuss learning, we need to understand where the two words "space" and "motion" come from. They did not naturally take the form they have today.

## 1.1 From Force to Energy: A Story That Changed the World

In classical mechanics, what we are most familiar with is Newton's equation:

$$F = ma$$

It is powerful — so powerful that it could almost serve as the symbol of modern science. As long as we know what forces act on an object, we know how it accelerates; as long as we know how it accelerates, we can predict its future position.

But Newton's equation also has a very real problem: you must thoroughly analyze every force.

A small ball rolling on an inclined plane — you must analyze gravity, normal force, friction. A pendulum in motion — you must analyze tension and gravity. In a complex mechanical system with many components, many constraints, many directions, force analysis quickly becomes a tangled diagram. Of course you can keep calculating, but you will discover: what is really troublesome is not the motion itself, but how to describe that motion.

At this point, analytical mechanics offers a different language. It does not begin with "what is every force," but starts from a more holistic quantity: energy.

For example, the Hamiltonian. Roughly speaking, in many classical systems, the Hamiltonian can be understood as the total energy of the system:

$$H = T + V$$

where $T$ is kinetic energy and $V$ is potential energy. Kinetic energy describes how the system moves; potential energy describes what position, what situation, what structure the system is in.

Thus, the problem changes. We no longer fixate on every local force individually; instead, we place the entire system within an energy structure. The motion of the system is no longer just "what force acts where," but rather how it flows through an energy terrain.

This is a very important conceptual shift: **from item-by-item force analysis to a holistic characterization of the energy terrain.**

![Newton Force Analysis vs. Hamiltonian Energy Terrain](/figures/ch01_newton_vs_hamilton.svg)

*Left: In the Newtonian framework, the same particle is pulled by four component forces, and the analyst must enumerate the direction and magnitude of each one. Right: In the Hamiltonian framework, the same system is compressed into a potential energy contour map; the motion trajectory flows naturally along the terrain — two languages, the same physics, but the terrain on the right is visible at a glance.*

:::info

**Mr. Pallas's Cat's Position**

Newton gave us $F=ma$. This formula ruled science for two hundred years. But it has a fundamental limitation: you must thoroughly analyze every single force. A ball on an inclined plane — gravity, normal force, friction. A pendulum in motion — tension, gravity. A complex system with ten components — you end up analyzing thirty forces. This is not understanding; this is bookkeeping.

Hamilton said: stop counting forces. Look at energy. Place the entire system in an energy terrain. The little ball will naturally slide to the place of lowest energy. You don't need to chase every force — you only need to draw a terrain map.

The deep learning community spent ten years before realizing the same thing. We call it the "loss function." But we have been talking about it in Newton's language — "gradient descent," "parameter update," "optimization steps." These are all the language of forces. The language of forces lets you compute fast, but it doesn't let you see the terrain. Once you see the terrain, you discover: learning is not fumbling for the optimal solution in the dark — learning is sliding down the steepest slope across an energy terrain. You can't see the terrain because you've been counting forces all along.

What this book aims to do is lift your eyes from the table of forces and look at that terrain.

:::

## 1.2 Learning Can Also Be Seen as a Kind of Motion

Now let us return to artificial intelligence.

When training a model, we usually say: the model wants to minimize the loss function.

$$\min_\theta \, L(\theta)$$

Here $\theta$ represents the model parameters, and $L(\theta)$ is the loss function. The smaller the loss, the better the model performs on the training data.

Traditional textbooks usually go straight into gradient descent at this point:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

This is, of course, correct. But if we treat it only as an update formula, we miss what is truly beautiful about it.

This formula is actually saying: the model currently stands at a position $\theta_t$ in parameter space. The loss function forms a terrain over this space. The gradient $\nabla L(\theta_t)$ tells the model how the terrain changes in the vicinity of the current position. The learning rate $\eta$ determines how far the model moves at each step. Thus, training is not magic, but the motion of a model across a loss terrain.

At this point, learning becomes a problem of dynamics — mathematically isomorphic, in its structure, to the motion of an object in a potential energy terrain in Newtonian mechanics.

The model does not "suddenly become smart." The model moves through a space. It traverses some regions, avoids some regions, falls into some basins, passes through some saddle points, and finally comes to rest near a certain relatively stable position.

This is the terrain of learning.

![Loss Terrain and Gradient Descent Trajectory](/figures/ch01_loss_landscape.svg)

*Loss terrain in parameter space: the elliptical bowl-shaped surface corresponds to the anisotropic loss $L = \theta_1^2 + 3\theta_2^2$, the orange trajectory is the gradient descent path starting from the initial point, and the green dot is the convergence endpoint. Note that the trajectory does not plunge straight down but bends along the curvature of the terrain — this is precisely the meaning of "motion" rather than "parameter tuning."*

## 1.3 But What Exactly Is "Space"?

In the previous two sections we kept talking about "space," "terrain," "motion," as if the meanings of these words were self-evident. But if we stop and seriously ask — **what exactly is space?** — we will discover that the answer to this question is far from something as simple as "a place that holds things."

The concepts of space, point, vector, and distance did not fall from the sky all at once. They went through a long evolution. The ancient Greeks used geometry to understand land, architecture, celestial bodies, and order; modern physics used space to describe motion; modern mathematics generalized space into sets, manifolds, topology, metrics, probability distributions; artificial intelligence has brought space into parameters, representations, embeddings, and beliefs.

So to understand why learning needs geometry, we must first understand where the word "space" came from, what transformations it underwent, and why it ultimately became the fundamental grammar of learning systems.

Let us begin with Euclid.

In the *Elements*, Euclid built geometry upon a set of definitions, postulates, and common notions. Points, lines, planes, angles, circles — these were laid out one after another, and then the entire geometric world unfolded from them. Among the most famous definitions is: **A point is that which has no part.**

This statement is very strange. A point has no length, no width, no thickness. It cannot be cut, nor can it be seen. And yet, the entire geometric world must begin from this "thing that has no part."

Why? Because a point is not a tiny grain of sand. A point is the abstraction of position. When we say "here is a point," what we are really saying is not that there is an object here, but that here is a markable position. Points turn space into something that can be designated. Without points, space is just a vague expanse; with points, we can say: here, there, from here to there.

A line, then, is the trace left by the continuous motion of a point. A plane is the unfolding of a line. A solid is the extension of a plane. Thus, space is organized layer by layer: points provide position, lines provide connection, planes provide extension, solids provide containment. This is the fundamental intuition of Euclidean geometry: the world can be clearly divided, marked, connected, and proved.

We can tell a little philosophical story. This is not a historical conclusion in the strict scholarly sense, but a philosophical fable.

Imagine a person living in an ancient city-state. Every day he sees the agora, stone steps, walls, colonnades, roads, temples, and city walls. Human living space is carved by architecture into straight lines, planes, angles, and proportions. The city needs to measure land, engineering needs to plan structures, politics needs the agora, religion needs temples, trade needs roads. In such a world, the plane is not an abstract concept but a lived experience. People walk on the agora, measure land, build walls, design colonnades. Space is rendered flat, divisible, and provable by human engineering activity.

This is why Euclidean geometry is so captivating. What it offers is not a complete picture of the real universe, but a space that is governable, measurable, and provable. In this sense, Euclidean space is not merely a mathematical space; it also carries the traces of civilization — it comes from land, architecture, cities, engineering, and political order. It made humanity believe: as long as you find the right postulates, the world can be proved step by step.

## 1.4 Space Begins to Move: From Static Order to Dynamic Stage

By the time of modern physics, the role of space underwent a fundamental change.

In Euclid, space was primarily the place where geometric objects unfolded — points, lines, planes, and angles were constructed and proved within it — it was static. By Newton's time, space became the stage for motion. Objects moved in space, time flowed uniformly, forces changed the velocities of objects, trajectories described how objects went from one position to another.

At this point, a point was no longer just a geometric point — it could represent the position of a particle. Distance was no longer just the length between two points — it could participate in the computation of velocity, acceleration, and force. The position of an object could be written as $x(t)$, velocity as $v = dx/dt$, acceleration as $a = d^2x/dt^2$. Thus, space and time were bound together. Geometry was no longer just the study of static shapes, but became the language of motion.

This step is absolutely crucial for artificial intelligence. Because learning is exactly the same: we don't only care where the model is right now, we also care how it changes. If parameters are positions, then training is the trajectory of parameters changing over time. If hidden states are positions, then reasoning is the trajectory of hidden states changing across layers or steps.

The inspiration modern physics gives us is: **Space is not a backdrop. Once space combines with change, it becomes dynamics.**

## 1.5 Space Begins to Curve: Gauss Climbs Mountains and Riemann Lectures

Euclid's flat space reigned for two thousand years. Until the nineteenth century, when someone began to measure it seriously.

In the 1820s, Gauss was tasked with surveying the map of Hanover. He climbed to the top of a mountain, used mirrors to reflect sunlight, and precisely measured the sum of the interior angles of a triangle formed by three mountain peaks. If Euclid were correct, the sum should be exactly 180°.

Measurement result: the sum of the interior angles was slightly greater than 180°.

Gauss probably knew what this meant — the Earth is a sphere, and the sum of the interior angles of a triangle on a spherical surface is naturally greater than 180°. But he did not publish. The reason, it is said, was that he did not want to argue with the philosophical community, because Kant had just declared the flatness of Euclidean space to be an a priori necessity of human cognition — not an empirical fact, but a precondition of thought. Gauss did not want to stir up that trouble.

Riemann made this clear in his 1854 inaugural lecture. He proposed: **Curved space is completely self-consistent.** On a spherical surface, the sum of the interior angles of a triangle exceeds 180°; on a hyperbolic surface, it is less than 180°. This is not an error; this is a different geometry.

More crucially, Riemann said: the very concept of distance can be generalized. On a plane:

$$ds^2 = dx^2 + dy^2$$

On a general Riemannian manifold:

$$ds^2 = g_{ij} \, dx^i \, dx^j$$

where $g_{ij}$ is the metric tensor describing the local curvature of space. This step turned "space" from a fixed platform into an object that can curve, that can have different geometric properties in different regions.

When Einstein wrote down general relativity in 1915, he used Riemannian geometry: gravity is not a force, but the geometric effect of spacetime curvature. Objects move along geodesics, just as one travels the shortest path on a curved surface.

This has direct implications for deep learning. Parameter space is not necessarily flat. In some regions of parameter space, the curvature of the loss function is high (near sharp minima); in other regions, the curvature is low (near flat minima). If we walk through a curved parameter space with a fixed Euclidean step size, it is like using a straight ruler to measure a map on a sphere — going astray is inevitable. Natural gradient methods are precisely the direct application of Riemannian geometry to optimization: using the metric tensor to correct the direction of each step, so that the step size remains consistent within the local geometry of the parameter space.

![Triangles on Flat, Spherical, and Hyperbolic Surfaces: Sum of Interior Angles](/figures/ch01_curvature_triangles.svg)

*Triangles under three types of curvature. On a plane (curvature = 0), the sum of interior angles is exactly 180°; on a sphere (curvature > 0), the sum exceeds 180°, with all three sides being great-circle arcs; on a hyperbolic surface (curvature < 0), the sum falls short of 180°. The deviation Gauss measured atop the Hanover mountains was precisely the faint manifestation of Earth's curvature at macroscopic scale.*

## 1.6 Four Fundamental Words: Point, Vector, Distance, Space

Now let us gather these histories together and formally define four words that are crucial for learning.

**What is a point?** A point is the abstraction of position. In $n$-dimensional space, a point can be written as $p = (x_1, x_2, \ldots, x_n)$. In machine learning, an image flattened into a high-dimensional vector is a point in image space; a sentence encoded into an embedding is a point in semantic space; the full set of model parameters $\theta$ is a point in parameter space; an answer distribution $p(y|x)$ is a point in probability space. The significance of a point does not lie in how small it is, but in that it can be located.

**What is a vector?** If a point describes position, then a vector describes change. From point $a$ to point $b$, the vector $\mathbf{v} = b - a$ contains two pieces of information: direction (which way to go) and magnitude (how far to go). In learning, the gradient is a vector — it tells the parameters in which direction they should update. Word embeddings are vectors — they place semantic objects into a computable space. The query, key, and value in attention are vectors — they establish relationships between tokens through direction and similarity.

The most important operation on vectors is the inner product:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n u_i v_i$$

The inner product tells us two things at once: magnitude ($\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$) and directional relationship ($\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$). When the inner product is zero, the two vectors are orthogonal — in the space, they cannot "see" each other. When we write $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$, the $-\nabla L(\theta_t)$ here is a direction vector — it tells the model: if you want the loss to go down, go this way.

**What is distance?** The question distance answers is: how much do two objects differ? In Euclidean space, the most familiar is straight-line distance $d(a,b) = \sqrt{\sum_i (a_i - b_i)^2}$. But in artificial intelligence, Euclidean distance is not always the most appropriate.

Sometimes we care about whether directions are similar, so we use cosine similarity: $\cos(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$. "Cat" and "kitty" have nearly the same direction in word vector space, but their vector magnitudes may differ — cosine similarity considers them very close.

Sometimes we care about how different two probability distributions are, so we use KL divergence:

$$D_{\mathrm{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}$$

Note that KL divergence is not symmetric — $D_{\mathrm{KL}}(p\|q) \neq D_{\mathrm{KL}}(q\|p)$. Going from certainty to confusion, and going from confusion to certainty, have different geometric meanings. This is not a bug; it is precisely why it is naturally appropriate in belief space.

Distance is not a fixed answer. Distance is a measurement method we choose based on the structure of the space. What kind of space needs what kind of distance.

![Comparison of Three Distance Metrics](/figures/ch01_distance_metrics.svg)

*Left: Euclidean distance, with equidistance surfaces forming perfect circles, measuring straight-line length between two points. Middle: Cosine similarity, with equal-similarity lines being rays emanating from the origin; "cat" and "kitty" are similar because their directions are close, while "car" differs because its direction diverges — vector length is irrelevant. Right: KL divergence, between two Gaussian distributions $p$ (blue) and $q$ (orange); $D_\mathrm{KL}(p\|q) \neq D_\mathrm{KL}(q\|p)$ — the asymmetry reflects that "measuring the surprise of $q$ from the perspective of $p$" is not equivalent to the reverse direction.*

**What is space?** Most roughly, space is the place where objects can be placed, compared, and transformed. In Euclidean geometry, space allows points, lines, and planes to be constructed. In Newtonian mechanics, space allows objects to move. In vector spaces, space allows addition and scalar multiplication. In metric spaces, space allows us to measure distance. In probability spaces, space allows us to compare belief distributions.

So space is not empty. Space is itself structure. A space stipulates what can be compared, what can be moved, what can come close, what can be held invariant. When we say "learning happens in space," we are not saying that learning happens in some abstract backdrop — we are saying: learning depends on a certain structure; without structure, there are no learning paths; without paths, there is no reasoning.

## 1.7 Returning to Learning: Why Geometry Is Unavoidable

Now we can return to the central proposition of this book.

If learning is motion, then we immediately need to answer several questions.

First, **where does the model move?** It certainly does not move in the three-dimensional space of physical reality. It moves through parameter space, representation space, probability space, and belief space. Parameter space records the weight positions of the model, representation space records the structure of encoded inputs, probability space records the model's belief distribution over answers, and belief space records how judgments change during the reasoning process.

Second, **in what direction does the model move?** This requires direction. The gradient is a direction, attention is a direction, residual updates are a direction, and belief revision within a reasoning chain is also a direction.

Third, **how far has the model moved?** This requires distance. Euclidean distance describes straight-line differences between vectors, cosine similarity describes closeness of direction, KL divergence describes belief deviation. Different spaces demand different notions of distance.

Fourth, **why does the model stop?** This requires stability. A system may converge to a minimum, may enter oscillation, may diverge, or may arrive at a fixed point.

Thus we discover that learning needs geometry, not because geometry is elegant, but because without geometry, we cannot even formulate these questions clearly. Space, direction, distance, path, curvature, stability — these are not external metaphors; they are the structural language of learning systems.

Deep learning is often described as an optimization problem. This is certainly reasonable, but not yet sufficient. Optimization is concerned with "how to make the loss go down," whereas the mechanics of learning is concerned with: In what space does the model move? What terrain does this space have? What equations govern the motion? Which regions are stable? Which regions are dangerous? Which paths yield generalization? Which paths merely memorize the training data?

We can gather all of this together with a maximally general formula:

$$S_{t+1} = S_t + \eta F_\theta(S_t, x)$$

Here $S_t$ is the current state of the system, $x$ is the input, $F_\theta$ is some update field determined by the model, and $\eta$ is the step size. If $S_t$ is parameters, this is training. If $S_t$ is hidden representations, this is feature evolution in a deep network. If $S_t$ is a belief distribution, this is belief update during reasoning. If in the end $S^* = F_\theta(S^*, x)$ is satisfied, that is a fixed point — the system has found a position where its own push on itself is exactly zero.

This formula places learning, reasoning, state updates, residual networks, Transformer blocks, and fixed-point models all into the same language: **the current state, driven by a field, moves toward the next state.** It will be the central equation running through the entire book.

## 1.8 The Geometric Intuition of Reasoning

Finally, let us use this framework to take a fresh look at reasoning models. This will plant the seeds for the entire discussion of Volume III.

Reasoning is not a single answer, but a trajectory. A model starts from a problem, forming an initial state in representation space. Through transformations between layers, it continuously updates this state. It may head toward the basin of attraction of the correct answer, or toward the basin of attraction of a wrong answer. What we call training is precisely changing the terrain of this space — making correct paths easier to walk, and wrong paths harder to stabilize.

This also explains a very interesting phenomenon: why does chain-of-thought training sometimes improve the model's ability to output answers directly? Because chain-of-thought training is not just teaching the model to "write a few more sentences." At a deeper level, it may have changed the model's internal representation terrain. It teaches the model to form more structured intermediate states while reading the problem, enlarging the basins of attraction near correct answers, and narrowing the paths near wrong answers. Thus, even if the model ultimately does not explicitly output a chain of thought, it may have already learned a better way of moving internally.

The explicit chain of thought is merely the textual projection of a trajectory. The real reasoning unfolds in the geometric terrain inside the model.

---

## 1.9 Chapter Summary

This chapter has done one thing, but that one thing is the foundation of the entire book: **reconceiving learning from "formula updates" to "motion in space."**

This transformation rests on a deeper understanding that spans two thousand years: space is not immutable. Euclid's space is static order — points, lines, and planes are constructed and proved within it. Newton's space is a stage for motion — objects change position within it, and trajectories describe change. Riemann's space can curve — distance is no longer a straight line, but the shortest path along a geodesic. And in artificial intelligence, space becomes the arena where parameters, representations, probabilities, and beliefs undergo change.

A point is the abstraction of position. A vector is the direction of change. Distance is the measurement of structure. Space is the arena where these relations can hold.

If learning is motion, then we need space. If we have space, we need points, vectors, and distance. If we have distance, we can discuss approach and withdrawal. If we have direction, we can discuss gradients and updates. If we have terrain, we can discuss optimization and convergence. If we have convergence, we can discuss why reasoning stabilizes on a particular answer.

Therefore, geometry is not the decorative language of deep learning, but the fundamental grammar of learning systems.

In the next chapter, we will enter the most concrete page of this grammar: **Where is the model's body? How is its field of view formed?** — Parameter space and representation space, body and field of view; this is the first step in understanding the terrain of learning.

---

## Unresolved Questions

1. If learning is the motion of a model in some space, then does this "space" genuinely exist, or is it a language we construct in order to understand the model?

2. Can the loss function truly be regarded as a terrain? If so, is this terrain something the model itself sees, or something the researcher draws from the outside?

3. Gradient descent appears to be "descending a mountain," but the parameter space of deep learning often has millions or billions of dimensions. In such a space, are the low-dimensional metaphors of "valley," "saddle point," and "basin" still reliable?

4. Newtonian mechanics starts from forces, Hamiltonian mechanics starts from energy. Deep learning starts from loss. Can loss, then, be understood as a kind of "energy" in artificial intelligence systems? If so, can the approach of Lyapunov functions be used to rigorously prove convergence?

5. Euclid defined a point as "that which has no part." In artificial intelligence, an embedding vector has hundreds or thousands of dimensions — can it still be called a "point"?

6. Euclidean space emphasizes straight-line distance. But in semantic space, "cat" and "tiger" may be closer than "cat" and "table." Is this "closeness" geometric distance, statistical co-occurrence, or the projection of human meaning?

7. Is distance an objective fact? Or does every definition of distance — Euclidean, cosine, KL — implicitly embed a certain way of looking at the world?

8. If space can be curved, then are the parameter space and representation space of deep learning also curved? Does this curvature come from the data, from the model architecture, or from the training process?

9. In high-dimensional spaces, human low-dimensional intuition often fails. So when we use two-dimensional and three-dimensional diagrams to explain deep learning, are we actually helping understanding, or are we creating illusions?

10. Euclidean space arose from a flat, measurable, provable world experience. Might the age of artificial intelligence then give birth to a new conception of space: a space constituted by data, embeddings, probabilities, and beliefs?

11. If reasoning is not an answer but a trajectory, then how should we observe that trajectory? Is looking at the output text enough? Or must we enter into hidden states, attention structures, and probability distributions?

12. Can we establish a new theory of learning: one that no longer asks only whether loss decreases, but asks whether the model's state trajectory is stable, interpretable, and heading toward better structural regions?

---

**The central question this chapter leaves behind is:**

**If learning is a kind of motion, then in what space, exactly, should we situate that motion — and what does that space look like?**

:::info

**In what space does learning occur? Is this space flat or curved? What counts as "near," and what counts as "far"?** — These questions will guide the entire book: from parameter space to representation space, from loss terrain to belief space, from fixed points to reasoning trajectories. If you pause at this chapter and take away only one question, let it be this one. Because until you can answer "where is the model?", "how does the model learn?" will forever remain an empty phrase.

:::
