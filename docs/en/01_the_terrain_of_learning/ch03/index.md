# Chapter 3: Loss Terrain and Gradient Motion

At the end of the last chapter, we established a core pairing: parameter space is the body, representation space is the field of vision. The body stands at some position in parameter space, and its field of vision—its way of understanding the world—is entirely determined by this position.

But why does the body walk in this direction and not that one? Why does it sometimes walk fast and sometimes slow? Why does it ultimately stop at one place rather than another?

The answer lies underfoot.

## 3.1 Who Put a Ball on the Mountaintop?

Imagine you are blindfolded and airdropped into an unfamiliar wilderness. You don't know where you are, you don't know where the lowest valley lies, you don't know how vast or rugged the terrain beneath your feet is. The only thing you can perceive is the slope transmitted through the soles of your feet—tilting left or right, steep or gentle.

Your task is to find the lowest point in this wilderness. You cannot fly up and survey the whole landscape—you can only rely on the slope signal at your feet, walking step by step.

This is gradient descent. After random initialization, the model stands at some position in parameter space. It cannot see the entire terrain—it can only "feel" the gradient at its current position, which is the local tilt direction and steepness of the terrain beneath its feet. It takes a small step in the direction of the negative gradient, then feels again, takes another step, feels again... repeating until the slope underfoot approaches zero and it can no longer find a downhill direction.

But here is a crucial question: **different ways of walking will take you to completely different destinations.** Striding with large steps, you might leap over narrow valley floors or even tumble off a cliff. Tiptoeing cautiously, you are indeed safe, but you might not reach any valley floor before nightfall. Running with momentum, you can glide swiftly across flat plateaus, but you might also overshoot the best valley and end up in a suboptimal basin. Adjusting your stride automatically based on the terrain—slowing down at steep cliffs, speeding up on gentle plateaus—sounds clever, but you might get stuck at some saddle point for a long time.

The map of this wilderness is the loss function $L(\theta)$. It assigns an "elevation" to every position in parameter space, characterizing how poorly the model performs at that position. Different loss functions produce different wilderness landscapes. Different optimizers are different walking strategies employed on the same wilderness.

What this chapter aims to do is to unfold the map of this wilderness and clearly explain the walking strategies. We will first formalize—rigorously defining concepts like loss functions, gradients, the Euler method, critical points, and optimizers in mathematical language—and then use these definitions to tell a complete story: how a randomly initialized model, on this terrain, walks step by step toward its final destination.

## 3.2 Formalizing the Loss Function

The loss function $L(\theta)$ is a mapping from parameter space to the real numbers:

$$L: \mathbb{R}^N \to \mathbb{R}$$

It assigns a scalar value to every parameter vector $\theta$, measuring how poorly the model performs under the current parameters. The goal of training is to find the $\theta$ that makes $L(\theta)$ as small as possible.

In deep learning, there are three most common loss functions.

**Mean Squared Error (MSE)**. Used for regression tasks—predicting continuous values. Given $n$ samples, with true values $y_i$ and model predictions $f_\theta(x_i)$:

$$L_{\text{MSE}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - f_\theta(x_i))^2$$

The geometric meaning of MSE is: it draws a quadratic bowl in parameter space. For a linear model $f_\theta(x) = \theta^\top x$, $L_{\text{MSE}}$ is a strictly convex quadratic function of $\theta$—there is only one global minimum, and gradient descent will inevitably slide to the bottom of the bowl. But in deep networks, $f_\theta$ is nonlinear, and MSE is no longer convex; the bowl shape transforms into a complex terrain with multiple valleys and saddle points.

**Cross-Entropy (CE)**. Used for classification tasks—predicting discrete labels. For $K$ classes, let the one-hot encoding of the true label be $\mathbf{y}_i$, and the model's output probability distribution be $\hat{\mathbf{y}}_i = \text{softmax}(f_\theta(x_i))$:

$$L_{\text{CE}}(\theta) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log \hat{y}_{i,k}$$

Cross-entropy comes from information theory: it measures the KL divergence between the true distribution $\mathbf{y}_i$ and the model's predicted distribution $\hat{\mathbf{y}}_i$ (with the constant term removed). When the model's predicted probability for the correct class approaches 1, cross-entropy approaches 0; when the model's predicted probability for the correct class approaches 0, cross-entropy tends toward infinity—this is an extremely severe penalty. Cross-entropy almost always outperforms MSE in classification problems because its gradient signal remains strong when probabilities are near 0 or 1, whereas MSE "saturates" (gradient vanishing) in extreme probability regions.

**Mean Absolute Error (MAE)**. Also used for regression, but uses absolute value instead of squaring:

$$L_{\text{MAE}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} |y_i - f_\theta(x_i)|$$

The gradient magnitude of MAE is constant (unlike MSE, where larger errors produce larger gradients), making it less sensitive to outliers. However, from an optimization perspective, MAE is non-differentiable at $y_i = f_\theta(x_i)$, which makes it less commonly used in deep networks than MSE.

These three loss functions produce three different ways of "sculpting the terrain" in parameter space. For the same model and the same dataset, switching the loss function changes the terrain—the locations of minima, the shapes of valleys, and the distribution of saddle points can all be completely different.

Now, imagine you are already standing at some position in this wilderness. The loss function tells you the elevation under your feet—but you still need to know one thing: **in which direction will the elevation decrease?**

![3D Loss Bowl](/figures/ch03_loss_bowl_tikz.svg)

*3D loss bowl $L=\theta_1^2+\theta_2^2$. Every point on this surface is a possible model state; training is descent toward the bottom.*

![Gradient Descent Trajectory on Loss Bowl](/figures/ch03_loss_bowl_trajectory_tikz.svg)

*Gradient descent trajectory on the loss bowl. Red path shows the descent from initialization to minimum.*

## 3.3 Gradient: A Vector of Partial Derivatives

With the loss function $L(\theta)$ in hand, the next step is knowing which direction to walk to reduce the loss.

The gradient answers this question. The gradient is the vector formed by the partial derivatives of the loss function with respect to each parameter:

$$\nabla L(\theta) = \left( \frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots, \frac{\partial L}{\partial \theta_N} \right)$$

Each component $\partial L / \partial \theta_i$ answers: if I increase the parameter $\theta_i$ by a tiny amount, how much will the loss change? A positive value means increasing $\theta_i$ will increase the loss; a negative value means increasing $\theta_i$ will decrease the loss.

The gradient has two key geometric properties.

First, **the gradient points in the direction of steepest ascent of the loss.** This is not a convention but a mathematical fact. For any unit direction vector $\mathbf{u}$, the directional derivative along $\mathbf{u}$ equals $\nabla L \cdot \mathbf{u}$. By the Cauchy–Schwarz inequality, this quantity is maximized when $\mathbf{u}$ is aligned with $\nabla L$. Therefore, $-\nabla L$ is the direction of steepest descent of the loss—which is exactly the direction we want to walk.

Second, **the magnitude of the gradient $\|\nabla L\|$ indicates the steepness of the terrain.** In regions where the loss changes dramatically, the gradient is large (cliffs, precipices); in regions where the loss changes gently, the gradient is small (plateaus, basin floors). The magnitude of the gradient itself is an important signal—it tells the model how large its stride should be.

In deep networks, gradients are computed via backpropagation. Backpropagation is not the focus of this chapter, but from a geometric perspective, what it does is essentially: starting from the output end, it passes partial derivatives backward layer by layer along the network's computation graph, using the chain rule to efficiently compute the gradients of all parameters. Its geometric significance lies in this—it decomposes the global signal "how wrong is the model's output" into local signals of "how much responsibility each parameter bears for this error."

![Gradient Vector Field: Descent Directions on the Loss Terrain](/figures/ch03_gradient_field.svg)

*The gradient field of a non-convex loss function $L = \theta_1^2 + 2\theta_2^2 + 1.5\sin(1.8\theta_1)\cos(1.8\theta_2)$. The blue background shows loss contours, and the vermilion arrows point in the negative gradient direction (steepest descent). The orange trajectory is a gradient descent path, sliding step by step from the starting point (circle) to the endpoint (square). The diamond marks a saddle point—where the gradient is zero, but some directions go uphill and others go downhill.*

![Vector Field](/figures/ch03_vector_field_tikz.svg)

*Vector field $v(x)=-x$: every point in space has a preferred direction. The origin $x^*$ is a global attractor.*

Now you know the slope under your feet. But how should you take a step? How large should each step be? This leads to a question that is equally ancient in physics and mathematics: **how do you turn a continuous motion into a step-by-step discrete walk?**

## 3.4 Explicit Euler Method: Gradient Descent as a Discrete Dynamical System

With the gradient direction known, the model can move. But how?

The most natural idea is: at each step, walk a small distance in the negative gradient direction. This update rule is called gradient descent:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

where $\eta > 0$ is the learning rate, also called the step size.

![Computation Graph of Gradient Descent](/figures/ch03_gradient_descent_graph_tikz.svg)

*Computation graph of gradient descent. θ_t → ∇L(θ_t) → -η·∇L → θ_{t+1}, with a feedback loop below. Each training step is one execution of this graph.*

This formula is not arbitrary. It is the **explicit Euler method** applied to discretize the ordinary differential equation $\frac{d\theta}{dt} = -\nabla L(\theta)$.

More generally, consider a dynamical system driven by a vector field $F(\theta)$:

$$\frac{d\theta}{dt} = F(\theta)$$

The explicit Euler method discretizes this continuous-time system as:

$$\theta_{t+1} = \theta_t + \eta \, F(\theta_t)$$

Setting $F(\theta) = -\nabla L(\theta)$, we obtain gradient descent. From this perspective, **gradient descent is not an optimization trick, but the discrete-time evolution of a dynamical system.** The learning rate $\eta$ is not a "magic number" but the discretization step size in the Euler method.

This perspective is extremely important because it connects two worlds. In the continuous-time limit ($\eta \to 0$), the trajectory of gradient descent approaches the exact solution of the gradient flow:

$$\frac{d\theta}{dt} = -\nabla L(\theta)$$

Gradient flow is smooth, deterministic, and follows a curve of continuous sliding along the steepest descent direction. The gradient descent used in actual training is a discrete sampling along this smooth curve. If $\eta$ is too large, the discrete trajectory deviates from the continuous solution, exhibiting oscillations or even divergence; if $\eta$ is too small, the discrete trajectory is accurate but proceeds extremely slowly.

Equally important, **the explicit Euler method is precisely the mathematical prototype of ResNet and Transformer decoders.** ResNet's residual connection $h_{l+1} = h_l + f_\theta(h_l)$ is, in form, an Euler step—the hidden state $h_l$ advances along the vector field $f_\theta$. The causal autoregression of GPT models—where each step feeds the current output back as the input for the next round—is likewise an Euler iteration carried out in hidden state space. We will unfold this thread in detail in ch7 (Fixed Points).

![Explicit Euler Method: Discrete Step Size vs. Continuous Gradient Flow](/figures/ch03_euler_steps.svg)

*Left: Euler steps on a one-dimensional double-well loss function $L(\theta) = 0.25\theta^4 - 2\theta^2 + \theta + 6$. Green ($\eta=0.05$) descends stably to the minimum; orange ($\eta=0.3$) oscillates near the valley bottom; red ($\eta=0.8$) diverges due to excessive step size. Right: comparison of continuous gradient flow (blue curve) and discrete Euler steps ($\eta=0.2$, vermilion staircase)—discretization samples the smooth curve, and larger step sizes cause greater deviation.*

You follow the Euler method, step after step. The slope becomes gentler and gentler, your strides smaller and smaller. Finally one day—at some position—the slope under your feet disappears. No matter which direction you probe, the elevation no longer decreases.

You stop. But where have you stopped? At the deep bottom of a valley (a minimum)? On a mountaintop (a maximum)? Or at that strange kind of place—where walking forward goes downhill but walking sideways goes uphill (a saddle point)?

Distinguishing these three cases requires information finer than the gradient—you need to know whether the terrain under your feet bends upward or downward.

## 3.5 Classification of Critical Points: Minima, Maxima, and Saddle Points

Where does gradient descent ultimately go? If training proceeds smoothly, the loss stops decreasing and the gradient approaches zero. At this point the model has reached a **critical point**:

$$\nabla L(\theta^*) = 0$$

But a critical point is neutral—it could be a minimum, a maximum, or a saddle point. To distinguish them, we need second-order information.

The **Hessian matrix** $H(\theta)$ is the $N \times N$ matrix of second-order partial derivatives of the loss function with respect to all parameters:

$$H_{ij}(\theta) = \frac{\partial^2 L}{\partial \theta_i \partial \theta_j}$$

The Hessian describes the local curvature of the loss terrain at $\theta$. Its eigenvalues and eigendirections tell us: along different directions, the terrain bends differently—some directions bend upward (positive curvature), some bend downward (negative curvature), and some are nearly flat (near-zero curvature).

Based on the eigenvalues of the Hessian, critical points can be rigorously classified:

- **Local minimum**: $\nabla L = 0$ and all eigenvalues $\lambda_i > 0$ of $H$. Moving away from the point in any direction increases the loss. This is the ideal destination for gradient descent.

- **Local maximum**: $\nabla L = 0$ and all eigenvalues $\lambda_i < 0$ of $H$. Moving away from the point in any direction decreases the loss. In deep learning, maxima are extremely rare—because gradient descent actively avoids ascent directions, you can hardly ever "climb" onto a mountaintop.

- **Saddle point**: $\nabla L = 0$ and $H$ has both positive and negative eigenvalues. Some directions go "uphill" (positive curvature), and some go "downhill" (negative curvature). Saddle points are the most common type of critical point in high-dimensional parameter space—in fact, the probability that a random critical point is a minimum decays exponentially with the dimension $N$, while the probability that it is a saddle point approaches 1.

Saddle points are a subtle and important problem in deep learning. In low-dimensional spaces, a saddle point seems like an impossible place to linger—"there is always a downhill direction to escape through." But in extremely high-dimensional parameter spaces, a saddle point can possess a vast "flat plateau"—the vast majority of directions are flat or uphill, and only a very small number of directions (corresponding to the few negative eigenvalues of the Hessian) lead downhill. Gradient descent can linger on these plateaus for a long time, because the stochastic gradient rarely happens to point exactly along those few downhill directions.

- **Degenerate critical point**: $\nabla L = 0$ and $H$ has zero eigenvalues. In this case, second-order information is insufficient to determine the nature of the point; higher-order derivatives are needed.

In actual deep learning training, we almost always end up in regions where the Hessian eigenvalues are all non-negative (but not necessarily all strictly positive)—these regions may be true minima, or they may be minimum basins with flat directions.

![Saddle Point: $L = \theta_1^2 - \theta_2^2$](/figures/ch03_saddle_point.svg)

*The classic saddle surface $L = \theta_1^2 - \theta_2^2$. Along the $\theta_1$ direction (green curve), the surface bends upward—positive curvature, an "uphill" direction. Along the $\theta_2$ direction (red curve), the surface bends downward—negative curvature, a "downhill" direction. At the saddle point, the gradient is zero, but the Hessian has both positive and negative eigenvalues. In high-dimensional parameter spaces, the vast majority of critical points are saddle points.*

The Hessian gives us a kind of microscope—we can zoom in on a critical point and see how it bends in every direction. But what concerns the wilderness hiker is not just a single point: **what is the overall shape of this terrain like?** How long are the valleys? How wide are the basins? To go from one valley to another, are there low-elevation passages?

These questions are about the large-scale structure of the terrain—they cannot be answered by the Hessian at a single point alone, yet they are indeed shaped by the pattern of how the Hessian varies from point to point across space.

## 3.6 The Geometry of Valleys, Saddle Points, and Minima

With the Hessian formalized, we can more precisely describe the various geometric forms on the loss terrain.

A **valley** is not a critical point, but a region. It is a set of interconnected low-loss paths; walking downward along the valley floor, the loss continuously decreases and the path is stable. The typical feature of a valley is that along the valley floor direction, the Hessian has a very small positive eigenvalue (gentle), while in directions orthogonal to the valley floor, the Hessian has very large positive eigenvalues (steep). This means the model can move quickly along the valley floor, but if it deviates from the floor, the steep "valley walls" bounce it back.

A **flat minimum** is a minimum point where all eigenvalues of the Hessian are relatively small (but all positive). The curvature is low in all directions, meaning that small perturbations of the parameters will not significantly increase the loss. Flat minima generally correspond to better generalization—because the small differences between the test data distribution and the training data distribution correspond to a small displacement from the training minimum in parameter space. If the minimum is flat, this small displacement will not drastically increase the loss, and the model's performance on the test set will not significantly degrade.

A **sharp minimum** is a minimum point where the Hessian has at least one very large positive eigenvalue. The curvature is extremely high in some directions, and tiny parameter perturbations cause the loss to rise sharply. Sharp minima tend to generalize poorly—they may have precisely fitted noise patterns in the training data that do not exist in the test data.

The type of minimum where the model ends up at the end of training is jointly influenced by the optimizer, learning rate, batch size, and the length of the training trajectory. In general, the noise of small-batch SGD tends to push the model out of sharp minima, causing it to eventually settle into flatter basins—this may be one reason why SGD generalizes better than full-batch gradient descent.

In extremely high-dimensional parameter spaces, there is an even deeper observation: **minima may be connected to one another.** Not by straight lines (which might pass through high-loss regions), but by curved, low-loss paths. This suggests that parameter space may contain a vast, well-connected "archipelago" of low loss—different good models are not isolated islands, but belong to a broader low-loss terrain structure.

![Sharp Minimum vs. Flat Minimum](/figures/ch03_sharp_vs_flat.svg)

*Left: sharp minimum—the Hessian has large eigenvalues, and the function $L = 0.8\theta^2$ has extremely high curvature. A tiny parameter perturbation (dashed line) causes the loss to rise sharply ($\Delta L$ large). Right: flat minimum—the Hessian eigenvalues are small, and the function $L = 0.08\theta^4$ has gentle curvature. Even with a larger parameter perturbation, the increase in loss is small ($\Delta L$ small). Generalization ability is closely related to basin width: where the model stands at the end of training matters more for test-set performance than how deep the point is.*

So far, we have a complete "terrain description language"—the loss function defines the shape of the ground, the gradient indicates the direction of tilt underfoot, the Euler method turns continuous sliding into discrete steps, and the Hessian helps us distinguish valleys, peaks, and saddle points at critical points. We even know that good models tend to settle in broad, flat basins rather than in sharp crevices.

But one question remains unresolved. So far, we have assumed only one way of walking: at each step, walk a fixed-length stride in the direction of the current negative gradient. **Is this really the best way to walk?**

Return to the wilderness hiker metaphor. A clever hiker does not walk in only one rigid style. At steep cliffs, he slows down; on flat plateaus, he strides boldly. If he finds himself bouncing left and right at the bottom of a canyon, he learns to carry a bit of momentum—letting his body glide along the valley floor rather than re-sensing the slope at every step. If he has walked similar terrain before, he uses past experience to adjust his gait.

In other words: **the terrain is the same, but the way of walking can be completely different. Different walking strategies will take you to different destinations.** This is the problem that optimizers are designed to solve.

## 3.7 Optimizers: Different Ways of Walking on the Same Terrain

The formalizations in the previous four sections all rest on a single assumption: the update direction at each step is the negative gradient at the current position, and the step size is fixed at $\eta$. Mathematically, this is the explicit Euler method; in practice, this is plain SGD.

But there is more than one way to walk through the wilderness. Three key questions have driven the development of optimizers:

1. **Can the step size be adaptive?** At steep cliffs you should tiptoe slowly; on flat plateaus you can stride boldly. The "steepness" varies across different parameter dimensions—some directions have large curvature (large Hessian eigenvalues), others have small curvature—a fixed global step size cannot simultaneously accommodate all directions.

2. **Can we carry momentum?** At the bottom of a winding canyon, the gradient direction oscillates back and forth. If at each step you only look at the gradient at the current position, the model will bounce repeatedly between the two sides of the canyon, making painfully slow progress. If you carry a bit of momentum—letting past directions of motion influence the current stride—you can glide along the valley floor like a skier.

3. **Should the step size decay?** Early in training, the model is far from the optimum, and large strides enable rapid approach. Late in training, the model is fine-tuning at the valley bottom, and small strides avoid oscillation. Having the step size automatically decay over time or based on gradient history is a very natural idea.

The following four optimizers give different answers to these questions.

**SGD (Stochastic Gradient Descent)**. The plainest way of walking. At each step, walk a fixed step size $\eta$ in the direction of the mini-batch gradient $g_t = \nabla \hat{L}(\theta_t)$:

$$\theta_{t+1} = \theta_t - \eta g_t$$

The advantage of SGD is pure gradient following—no additional assumptions. The disadvantage is also this: it oscillates in the steep directions of canyon terrain, crawls like a snail in flat directions, and is easily trapped near saddle points. The stochasticity of the gradient (mini-batch noise) is, in some cases, an advantage—it provides random perturbations that help escape saddle points, and may also push the model out of sharp minima.

**Momentum (Polyak's Heavy Ball Method)**. Adds "inertia" on top of SGD—uses a velocity variable $v_t$ that accumulates an exponentially weighted average of past gradients:

$$v_{t+1} = \beta v_t + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

where $\beta \in [0, 1)$ is the momentum coefficient, typically set to 0.9. When $\beta = 0$, it reduces to SGD. The geometric intuition of momentum is clear: in the steep directions of a canyon, gradients alternate between positive and negative, and the accumulated momentum cancels out, yielding a small effective step size; along the valley floor direction, gradients are consistently aligned, momentum accumulates continuously, and the effective step size is amplified. This means momentum **automatically decelerates in oscillatory directions and accelerates in consistent directions**—no need to manually tune the learning rate per dimension.

**Nesterov Accelerated Gradient (NAG)** is a variant of momentum. The sole change is that it first "looks ahead" along the accumulated velocity direction, then computes the gradient at that lookahead position:

$$v_{t+1} = \beta v_t - \eta \nabla L(\theta_t + \beta v_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

This "lookahead" operation corrects the problem of standard momentum overshooting on curves, and in theory provides a faster convergence guarantee for convex functions ($O(1/t^2)$ vs. momentum's $O(1/t)$).

**Adam (Adaptive Moment Estimation)**. One of the most widely used optimizers today. It combines momentum (first moment $m_t$) and adaptive step sizes (second moment $v_t$):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $\beta_1 = 0.9$ controls the decay of the first moment, $\beta_2 = 0.999$ controls the decay of the second moment (exponential moving average of squared gradients), $\epsilon = 10^{-8}$ prevents division by zero, and the bias corrections $\hat{m}_t, \hat{v}_t$ fix the problem of $m_t, v_t$ being biased toward zero in the initial steps.

The geometric meaning of Adam is: $\sqrt{\hat{v}_t}$ estimates the "typical magnitude" of the gradient in that parameter dimension. In dimensions where the gradient varies wildly, $\sqrt{\hat{v}_t}$ is large, so the effective step size $\eta / \sqrt{\hat{v}_t}$ is small—automatically decelerating. In dimensions where the gradient varies little, $\sqrt{\hat{v}_t}$ is small, so the effective step size is large—automatically accelerating. Adam adjusts the step size independently for each parameter, making it excel at handling sparse gradients (such as embedding layers in NLP) and parameters of differing scales.

**AdamW** is a key correction to Adam. The original Adam is typically used together with $L_2$ regularization, but $L_2$ regularization under Adam's adaptive step size is not equivalent to weight decay. AdamW decouples weight decay from the adaptive step size:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

The final term $-\eta \lambda \theta_t$ is independent weight decay—at each step it pulls the parameters back toward zero by a small amount. This simple change significantly improves generalization in practice, making AdamW the de facto standard for Transformer training today.

![Optimizer Comparison: SGD vs Momentum vs Adam](/figures/ch03_optimizer_comparison.svg)

*Trajectory comparison of three optimizers on the Rosenbrock function $L = (1-\theta_1)^2 + 10(\theta_2 - \theta_1^2)^2$. This is a classic narrow-valley terrain—the global minimum is at $(1, 1)$, but the valley floor is extremely narrow and curved. SGD (left): fixed step size oscillates back and forth across the narrow valley, progressing slowly. Momentum (middle): inertia counteracts the lateral oscillations, accelerating the slide along the valley floor. Adam (right): adaptive step size + momentum, rapidly adjusts direction, coming closest to the minimum within 80 steps. Note that all three start from the exact same point and run for the same number of steps—the only difference is the walking strategy.*

:::info

**Mr. Pallas's Cat's Position**

You call the learning rate a "hyperparameter." You call the optimizer a "tuning choice." You call convergence "the loss stopped going down."

These are all the language of forces. Forces make you count—count step sizes, count momentum coefficients, count iteration steps. But forces do not let you understand.

From the perspective of energy, the learning rate is the size of the step you take on the energy landscape. The optimizer is your way of walking—walking straight, carrying momentum, or adjusting your stride automatically based on the firmness of the ground underfoot. Convergence is not about staring at the loss curve to see whether it flattens—it is about the system reaching a Lyapunov-stable point on the energy landscape. There, no matter which step you take, the energy will no longer decrease.

This is not a philosophical metaphor. Lyapunov turned it into a theorem in 1892. The loss function is a Lyapunov function—positive definite, zero at the minimum, monotonically decreasing along the gradient direction. If you accept this, then "will training converge" ceases to be an empirical question—you don't need to stare at the loss curve any more than you need to stare at a pendulum until it stops. You only need to verify three algebraic conditions. If the conditions hold, convergence is inevitable.

The deep learning community spent thirty years evolving optimizers from SGD to AdamW. But from Lyapunov's perspective, these are all different discretization schemes for the same thing—the same vector field, different step-size strategies. When you switch optimizers, you are not switching "algorithms"—you are switching walking styles. And what the walking style changes is not whether you can reach a destination—but which destination you reach. The noise of SGD pushes you into flat basins; the adaptive stride of Adam pulls you to the bottom of narrow valleys. You did not "choose an optimizer"—you chose a destination.

:::

![Energy Contours with Gradient Descent](/figures/ch03_energy_contour_tikz.svg)

*Energy contours with gradient descent path. Ellipses are level sets of $E(x)$; arrows show $x_{t+1}=x_t-\eta\nabla E(x_t)$.*

## 3.8 Putting It All Together: A Terrain Narrative of One Training Run

Now we have a complete vocabulary. Let us use it to tell a story.

The protagonist of the story is a randomly initialized model $\theta_0$. It stands at a random position in parameter space—like a hiker airdropped into an unknown wilderness, not knowing where it is, not knowing where the lowest valley lies, knowing only the slope underfoot.

It takes its first step. It computes the gradient $\nabla L(\theta_0)$ at its current position—that is, the downhill direction underfoot. It walks a small step in this direction, with step size $\eta$. This step is $\theta_1 = \theta_0 - \eta \nabla L(\theta_0)$.

Then it computes the gradient again, and takes another step. $\theta_2, \theta_3, \ldots, \theta_T$. Each step is an explicit Euler iteration; each step leaves a footprint in parameter space.

In the first few hundred steps, the loss drops rapidly. This is the "steep-slope zone" of the terrain—gradients are large, the direction is clear, and the model quickly slides from the high-loss plateau into some valley. During this phase, the model's field of vision (representation space) undergoes dramatic reorganization—it rapidly learns the lowest-level statistical regularities in the data.

After entering the valley, the loss decreases more slowly. The valley floor is broad—the Hessian has many small eigenvalues, and gradients in all directions are small. The model crawls slowly along the valley floor, fine-tuning its parameters. During this phase, the model refines its field of vision—it learns the more subtle structures in the data.

Sometimes, the model encounters a saddle point. The gradient is nearly zero, but some directions lead downhill. The noise in the stochastic gradient (randomness introduced by mini-batches) happens to amplify one of these downhill directions, pushing the model away from the saddle point and onward downward.

Eventually, the model settles in a flat basin. The gradient approaches zero, and all eigenvalues of the Hessian are non-negative but not large. The model no longer moves—not because it has "learned enough," but because the surrounding terrain is flat or gently upturned in all directions. It has reached a local minimum.

But the story does not end there. Because whether this minimum generalizes—whether it performs well on test data—depends on how wide it is. If the model stands in a broad, flat basin, then the small differences between training data and test data (corresponding to a small displacement in parameter space) will not significantly increase the loss—the model generalizes well. If the model stands at the bottom of a sharp crevice, then any small perturbation will cause the loss to spike—the model has overfitted.

The entire training process is the complete history of walking on the terrain. The direction and magnitude of the gradient determine the heading of each step; the eigenvalues of the Hessian determine the stability of the resting place; the learning rate determines the rhythm of the stride; the loss function determines the entire topography of the landscape.

This is why the loss terrain is not a metaphor, but a fact—it is something that can be rigorously described using partial derivatives, Hessian matrices, and dynamical systems.

---

## 3.9 Chapter Summary

This chapter completed the leap from "geometric intuition" to "formal definition."

The loss function $L: \mathbb{R}^N \to \mathbb{R}$ endows parameter space with terrain. The gradient $\nabla L$ is a vector—it points in the direction of steepest ascent, and its magnitude indicates the steepness of the terrain. Gradient descent $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ is the explicit Euler method—discretizing the continuous gradient flow $\frac{d\theta}{dt} = -\nabla L$ into iterations with step size $\eta$. Critical points $\nabla L = 0$ are classified by the eigenvalues of the Hessian matrix: all positive = minimum, all negative = maximum, mixed positive and negative = saddle point. The flatness of a minimum (the magnitude of Hessian eigenvalues) is closely related to generalization ability.

These definitions are not ornamentation. They constitute a complete geometric language, allowing us to precisely describe the seemingly mysterious process of "training": the model starts from a random position, follows the gradient direction, and walks step by step in Euler fashion across the loss terrain toward some critical point. And the nature of that critical point—whether it is a valley or a saddle point, flat or sharp—directly determines the model's ultimate capability and generalization performance.

In the next chapter, we will focus on the walking itself: **how different optimizers are different ways of walking, and how learning rate, momentum, and adaptive step sizes change the rhythm and direction of the stride.**

---

## Open Questions

1. Do different loss functions (MSE, cross-entropy, MAE) paint completely different terrains in parameter space, or are they different projections of the same "true terrain"? Are there systematic differences in the number and distribution of their minima?

2. The gradient points in the "direction of steepest ascent" in extremely high-dimensional spaces—but "steepest" is defined relative to the Euclidean norm. If the natural metric of parameter space is not Euclidean (as argued by the natural gradient method), does the meaning of "steepest" need to be redefined?

3. The explicit Euler method has a well-known stability condition: the step size $\eta$ must be smaller than $2 / \lambda_{\max}$ (where $\lambda_{\max}$ is the largest eigenvalue of the Hessian), otherwise the iteration diverges. In deep learning, we typically do not know $\lambda_{\max}$, yet we can still train stably. What mechanism provides this "unexpected stability"?

4. In high-dimensional parameter space, the probability that a random critical point is a saddle point approaches 1. So how does gradient descent escape saddle points? Is it that the random noise of SGD happens to provide perturbations in downhill directions, or is it that gradient descent itself, in discrete time, cannot be "trapped"?

5. The Hessian matrix has $N \times N$ elements—for a model with $N = 10^9$, the Hessian is completely incomputable. How can we estimate the flatness of a minimum in practice? Do "Hessian-free" flatness indicators exist?

6. Flat minima generalize better—is this a correlation or a causal relationship? Do there exist minima that are "flat but generalize poorly," or minima that are "sharp but generalize well"?

7. Are low-loss passages (paths connecting different minima) on the loss terrain universally present? If so, do they suggest a new way of analyzing models—studying "terrain connectivity" rather than "individual minima"?

8. Different optimizers (SGD, Adam, AdamW) tend to converge to different types of minima. Are there systematic, predictable differences in the Hessian spectra (eigenvalue distributions) of these minima?

9. The "terrain" of the loss function is determined by the training data. Does a small change in the training set—adding or removing a single sample—cause a dramatic shift in the positions of minima in parameter space, or are the positions of minima relatively stable?

10. During training, how does the spectrum of the Hessian (the distribution of eigenvalues) evolve over time? Does it undergo a sudden "collapse" at some stage—with a large number of near-zero eigenvalues appearing, signaling that the model has entered a broad basin?

11. Are the trajectories of the continuous-time gradient flow $\frac{d\theta}{dt} = -\nabla L$ and the discrete-time gradient descent $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ topologically equivalent when $\eta$ is finite? Or does discretization introduce dynamical behaviors that do not exist in the continuous system?

12. If we view the loss terrain as a physical system—analogous to the energy landscape of spin glasses—does "training" in deep learning correspond to some kind of physical quenching process? What kinds of exploratory behaviors correspond to the low-temperature limit (small learning rate) and the high-temperature limit (large learning rate)?

---

**The core question left by this chapter is:**

**If we could precisely compute all eigenvalues of the Hessian, could we predict all of a model's generalization behavior? Or does generalization depend on some terrain structure more global than local curvature?**

:::info

**The loss function paints a terrain on parameter space. The gradient tells us which way the ground tilts underfoot. The Euler step pushes us to the next position.** But how large should the step be? Walk in a straight line or carry momentum? Different terrains require different ways of walking—this is the story of optimizers. In the next chapter, we will compare SGD, Momentum, Adam, and various adaptive methods: they are not merely "different update formulas," but different walking strategies applied to the same terrain. Walk the right way, and even flat ground can lead to the valley bottom; walk the wrong way, and a single step near a cliff can send you tumbling down.

:::
