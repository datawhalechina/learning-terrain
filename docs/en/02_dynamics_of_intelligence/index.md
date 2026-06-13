# Volume II: Walking and Belief

At the end of Volume I, you stand in a wilderness. You know the terrain has slopes—the loss function carves mountains and basins. You know the body is in parameter space—every gradient step moves the model. And you know the vision is in representation space—the body's position determines how the model understands the world.

But there is a question you haven't asked: **Is this wilderness itself flat?**

You think it is. Because you're used to seeing the world through Euclidean eyes. The shortest path between two points is a straight line. The step size is just $\eta$. The distance is just $\sqrt{\sum (a_i - b_i)^2}$.

The three chapters of Volume II will tell you: **The space beneath your feet is not flat.** And "not flat" is not philosophical rhetoric—it has rigorous formalization, theorems, proofs, a complete mathematical chain from information geometry to dynamical systems to fixed points.

---

Chapter 4 begins with the most practical question: **Does the way you walk matter?** The answer: immensely. How large a step is safe? Should you carry inertia? Should you adapt your pace to the terrain? SGD, Momentum, Adam—they are not "different optimizers"; they are different walking strategies on the same wilderness. Regularization is not "preventing overfitting"—it is reshaping the terrain itself. And SGD noise tells you something deeper: how you walk determines where you end up.

Chapter 5 is the first mathematical peak of the book. The crow flies in a straight line—that is Euclidean language. Water flows along the energy terrain—that is Bregman language. In belief space, the "distance" between two probability distributions is not symmetric—the cost of going from certainty to confusion is not the same as going from confusion to certainty. KL divergence is the natural "energy difference" in belief space. The Yonglin Limit proves: as long as the Euler step is contractive under KL geometry, reasoning necessarily converges—not "usually," but geometrically compelled.

Chapter 6 is the finale of Volume II, and the chapter with the greatest mathematical density in the entire book. It maps the full language of dynamical systems—phase space, trajectories, fixed points, Lyapunov functions, attractors, bifurcations—onto the core architectures of deep learning. ResNet = explicit Euler method. GPT autoregression = Euler iteration in hidden state space. DEQ = fixed-point solving. Belief fixed point = KL solidification criterion. Four faces, one soul: **a dynamical system evolves along the direction of decreasing energy, until the vector field vanishes—the fixed point.**

---

Volume I let you see the terrain. Volume II teaches you to walk on curved terrain—and proves that, as long as you walk in the correct geometry, you will reach the destination.

But what is the destination? How do you know when you've arrived? Volume III—The Geometry of Reasoning—will answer.

---

**Volume II: Walking and Belief**
- [Chapter 4: Ways of Walking: Optimizers and Regularization](ch04)
- [Chapter 5: The Non-Euclidean World: Bregman Divergence and KL Divergence](ch05)
- [Chapter 6: Dynamical Systems and Fixed Points: From Lyapunov to DEQ](ch06)
