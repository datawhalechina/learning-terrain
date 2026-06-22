# Chapter 4: Ways of Walking: Optimizers and Regularization

At the end of the previous chapter, we stood before a contradiction.

On the one hand, we had acquired a complete language of terrain—the loss function describes the rises and falls of the ground, the gradient points out the direction of the steepest slope beneath our feet, Euler's method turns continuous descent into discrete steps, and the Hessian helps us distinguish valleys, saddle points, and minima at critical points. We also saw three radically different walking strategies—SGD, Momentum, and Adam—whose trajectories across narrow-valley terrain differ so dramatically they hardly seem to be traveling the same wilderness.

![SGD (red), Momentum (orange), and Adam (blue) on the same loss bowl. Different walking strategies produce different paths to the same minimum.](/figures/ch04_optimizer_paths_tikz.svg)

On the other hand, an unresolved question still hung over all of it: **How should you choose the step size? Should the step size change over time? And more importantly—can you reshape the terrain itself?**

A wilderness hiker does not merely walk. A clever hiker installs guardrails along cliffs, lays planks across marshes, and levels out excessively rugged mountain paths. He does not just read the map—he alters the map.

That is the subject of this chapter. It is divided into two halves: the first half concerns **step size**—how you choose it, how you let it vary over time, how you give different parameter dimensions different step sizes (this brings us back to the optimizers of ch3, but from a deeper perspective). The second half concerns **regularization**—how you directly intervene upon the terrain itself, smoothing sharp cliffs, filling in excessively deep crevasses, so that learning is no longer merely passive descent along existing gradients but an active reshaping of the landscape into something friendlier.

## 4.1 You Don't Know How Long the Road Is

Let us continue the wilderness-hiking metaphor.

You are blindfolded and air-dropped into the wilderness. All you know is the slope beneath your feet. You decide to walk in the direction of steepest descent. But you face the first decision: **How large a step should you take?**

Too small—you are certainly safe. Each step proceeds precisely along the direction of steepest descent; you won't stumble, you won't miss the valley floor. But the sun is setting. Your number of steps is finite. In a rugged high-dimensional wilderness, tiny-step progress may mean you never reach any real valley at all.

Too large—you move fast. But this wilderness is not smooth. In the instant you take a great stride, the curvature of the terrain may already have changed. You might step into thin air and fall off a cliff—loss no longer decreases but spikes upward. Or you might leap right over a narrow but extremely deep valley floor and land in a suboptimal broad basin.

Thus the choice of step size is, at heart, a **compromise between continuous terrain and discrete footsteps**. In the continuous world, the gradient flow $\frac{d\theta}{dt} = -\nabla L(\theta)$ is perfect—it follows a smooth curve, approaching the minimum arbitrarily closely. But in the discrete world, we can only sample the flow with finite-sized steps $\eta$. The smaller $\eta$, the closer the discrete trajectory to the continuous solution; the larger $\eta$, the larger the discretization error, until beyond some critical point—the discrete system loses stability entirely and diverges to infinity.

![Euler step decomposition: $x_{t+1}=x_t+\eta v(x_t)$. The update is the velocity direction scaled by the learning rate.](/figures/ch04_euler_step_tikz.svg)

![Computation Graph of Euler Step](/figures/ch04_euler_step_graph_tikz.svg)

*Euler step computation graph. x_t splits into two paths: one through v(x_t) scaled by η, the other direct to ⊕. They merge as x_{t+1}=x_t+η·v(x_t). Dynamics is not metaphysics — it is this graph.*

## 4.2 The Stability Boundary of the Learning Rate

How large is "too large" for a step size? This is not an empirical question—it has a rigorous mathematical boundary.

Consider the simplest possible quadratic loss function:

$$L(\theta) = \frac{\lambda}{2} \theta^2$$

The gradient descent update is $\theta_{t+1} = \theta_t - \eta \lambda \theta_t = (1 - \eta \lambda) \theta_t$. This is a geometric series. The condition for convergence is $|1 - \eta \lambda| < 1$, i.e.:

$$0 < \eta < \frac{2}{\lambda}$$

When $\eta = 1/\lambda$, the step reaches the minimum in a single stride (critical damping). When $\eta > 2/\lambda$, the system diverges—the absolute value of the parameter grows at every step, and the loss explodes.

For general non-quadratic loss functions, the role of $\lambda$ is played by the largest eigenvalue of the Hessian, $\lambda_{\max}$. In the steepest direction, the curvature of the terrain is greatest, and the discretization error is also greatest. If $\eta > 2 / \lambda_{\max}$, the system diverges along the steepest direction. Hence the necessary condition for global stability is:

$$\eta < \frac{2}{\lambda_{\max}}$$

But here lies a paradox. $\lambda_{\max}$ describes the curvature at the steepest part of the terrain. If you do not know $\lambda_{\max}$ for the entire terrain (and in practice you almost never do), how can you guarantee the step size is safe?

The answer offered by deep learning practice is surprisingly simple: **try a value first; if the loss explodes, reduce it; if the loss decreases too slowly, increase it.** This "empirical hyperparameter tuning" sounds inelegant, but its effectiveness rests on a deep geometric reason—in extremely high-dimensional parameter spaces, $\lambda_{\max}$ is typically contributed by a very small number of extremely steep directions. These directions are rapidly "conquered" early in training (parameters slide quickly into the valley floor along them), after which the effective terrain curvature drops sharply and the safety of $\eta$ automatically improves.

![Bifurcation diagram: as learning rate $\eta$ increases, behavior transitions from slow convergence to stable descent to oscillation.](/figures/ch04_bifurcation_tikz.svg)

But this raises another question: **should the step size remain constant throughout?**

Intuitively, no. Early in training, the model is far from the minimum, the absolute value of the loss is large, and large steps can quickly approach the target region. Late in training, the model is fine-tuning near the valley floor, and small steps can avoid oscillation and overshoot. The learning rate should **decay over time**.

![Stability boundary of the learning rate: convergence and divergence](/figures/ch04_learning_rate_stability.svg)

*Gradient descent on the quadratic loss $L = 2\theta^2$ ($\lambda=4$, critical step size $\eta_{\text{crit}} = 2/\lambda = 0.5$). Left: parameter trajectory—$\eta=0.1$ converges slowly, $\eta=0.3$ is stable, $\eta=0.5$ is critically damped (near-zero in one step), $\eta=0.65$ oscillates and diverges. Right: loss over time (log scale)—once the step size exceeds the critical value, the loss no longer decreases but grows exponentially.*


:::info

**Mr. Pallas's Cat: The $\lambda_{\max}$ You Don't Know**

$\eta < 2/\lambda_{\max}$ is a precise condition — but $\lambda_{\max}$ is the largest eigenvalue of the Hessian across the entire terrain. You don't know it. You cannot know it — for a billion-parameter model, the Hessian is a billion-by-billion matrix.

So every time you set a learning rate, you are making a geometric gamble. Your bet: $\lambda_{\max}$ does not exceed $2/\eta$. If it does — loss explodes, training collapses, and you stare at that red `NaN` on your screen, knowing you lost the bet.

But here is something more subtle. $\lambda_{\max}$ is not constant. It changes during training — typically dropping sharply in the first few steps (as the model rapidly "conquers" the steepest directions), then decaying slowly through the long fine-tuning phase. This means the safe $\eta$ you chose initially — the one that barely didn't blow up at step one — is already overly conservative by step ten.

You are using a global constant to cope with a locally varying geometric quantity. This is like driving from Guangzhou to Beijing: you glance at Guangzhou's speed limit sign before departing, then drive at exactly that speed the entire way. On the mountain roads of Hunan you might be just safe, but on the plains of Henan — you are crawling.

This is why adaptive optimizers (Adam, AdamW) are so effective in practice. They are not "better algorithms" — they are **mechanisms that estimate local curvature in real time and adjust step size automatically**. $\hat{v}_t$ is Adam's "local Hessian estimator" — not the exact $\lambda_{\max}$, but a rough proxy for $\lambda_{\max}$ in each parameter dimension. Adam's success is not due to more elegant mathematics — it is because it finally admits a fact you dared not face: **the $\eta$ you set was wrong after the very first step.**

:::

## 4.3 Learning Rate Schedules: The Lifecycle of Step Size

A learning rate schedule answers the question: how does $\eta$ vary with time $t$?

**Step decay.** The simplest strategy: every fixed number of steps, multiply the learning rate by a decay factor $\gamma < 1$. For example, halve $\eta$ every 30 epochs. Its advantage is simplicity; its disadvantage is that the timing and magnitude of the decay are set entirely by experience—you do not know at which moment the terrain will require a smaller stride.

**Cosine annealing.** A smoother scheme: let the learning rate decay along a cosine curve from the initial value $\eta_0$ to near zero:

$$\eta_t = \eta_{\min} + \frac{\eta_0 - \eta_{\min}}{2} \left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

where $T$ is the total number of steps. Cosine annealing descends slowly early in training (keeping large strides for exploration), accelerates its descent in the middle phase, and lands smoothly at the end. The geometric intuition: in the first half of training, the macroscopic structure of the terrain matters most—large strides can rapidly traverse plateaus and shallow valleys; in the second half, the microscopic details matter—small strides are needed for fine-tuning within the basin.

**Warmup.** A counterintuitive but crucial trick: in the very first few hundred steps of training, the learning rate does not decay from $\eta_0$ but instead **increases linearly from an extremely small value up to $\eta_0$**.

Why warm up? Because in the very first steps of training, the model parameters are randomly initialized, and both the direction and magnitude of the gradients are extremely unstable. Using a large learning rate immediately could push the model in a completely wrong direction in those initial steps—and this "initial directional bias" might never be fully corrected by subsequent gradient steps. Warmup gives the model a "probing period": use extremely small strides first to get a rough sense of the terrain's overall structure, establish stable gradient statistics (especially $\hat{v}_t$ for Adam), then accelerate forward.

**Cyclical learning rate.** A more aggressive strategy: instead of letting $\eta$ decay monotonically, let it rise and fall periodically within an interval. The large-$\eta$ phase helps the model escape saddle points or sharp minima; the small-$\eta$ phase lets the model fine-tune within its current basin. Cyclical learning rates exploit a deep property of SGD noise: **periodically raising the step size is equivalent to periodically "heating" the system, helping it jump out of local traps.**

A learning rate schedule is, at bottom, answering a single question: **At which stage of training does what kind of information matter most for terrain exploration?** Early on, global structure is needed—large steps. Late on, local fine-tuning is needed—small steps. In between, escaping saddle points is needed—noise and periodically large steps.

![Comparison of learning rate schedule strategies](/figures/ch04_lr_schedules.svg)

*Four common learning rate scheduling strategies. Step decay (top left): staircase decay, halving $\eta$ every N steps. Cosine annealing (top right): smooth decay along a cosine curve from $\eta_0$ to $\eta_{\min}$. Warmup + cosine (bottom left): linear warmup from a tiny value to $\eta_0$ over the first 20 steps, then cosine decay—this has become the standard configuration for Transformer training. Cyclical (bottom right): $\eta$ rises and falls periodically; the large-$\eta$ phase helps escape saddle points, the small-$\eta$ phase enables fine-tuning.*

## 4.4 Regularization: Adding Boundaries to the Terrain

We now turn to the other half of this chapter. So far we have assumed the terrain is given—the loss function is determined by the data and the task, and the model can only walk across this given terrain. But what if we could **reshape the terrain itself**?

This is the core idea of regularization. By superimposing additional "penalty terms" onto the original loss function, it directly alters the topography of parameter space. Previously flat regions may be raised, previously narrow valley floors may be broadened, previously sharp crevasses may be smoothed.

**$L_2$ regularization (weight decay).** Add the squared $L_2$ norm of the parameter vector to the original loss function:

$$L_{\text{total}}(\theta) = L(\theta) + \frac{\lambda}{2} \|\theta\|^2$$

The gradient becomes $\nabla L_{\text{total}} = \nabla L + \lambda \theta$. The update formula thus acquires an extra term:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) - \eta \lambda \theta_t$$

The last term $-\eta \lambda \theta_t$ pulls the parameters slightly back toward the origin at every step—hence the name "weight decay."
**$L_1$ regularization.** Uses the $L_1$ norm rather than the $L_2$ norm:

:::info

**Mr. Pallas's Cat: Regularization Is Pouring Cement into the Terrain**

You learned in textbooks that $L_2$ regularization is "equivalent to a Gaussian prior." You write `weight_decay=0.01` in your code. You think you are "preventing overfitting."

But what you are actually doing — from a geometric perspective — is **pouring a thin layer of cement across the bottom of the entire parameter space.**

The $L_2$ term $\frac{\lambda}{2}\|\theta\|^2$ is a quadratic bowl centered at the origin. It doesn't care what your data looks like — it imposes a uniform, origin-directed pull across the entire space. The farther from the origin, the stronger the pull.

This layer of cement has different effects in different regions. On the sharp crevices carved by training data noise — crevices typically far from the origin, because the model needs extreme parameter values to fit noise — the cement fills them in. On the broad basins supported by genuine data regularities — basins whose floors are usually not far from the origin — the cement merely raises the floor slightly, but the basin's structure remains intact.

This is why $L_2$ regularization improves generalization: **it does not make all solutions worse — it systematically suppresses the solutions that can only be reached through extreme parameter values (typically corresponding to overfitting).** Flat solutions near the origin (typically corresponding to generalization) are barely affected.

So the next time you write `weight_decay=0.01`, don't think "adding regularization." Think: you are pouring a 0.01-centimeter-thick layer of cement into parameter space. The fragile structures will crumble. The sturdy ones will remain.



:::

$$L_{\text{total}}(\theta) = L(\theta) + \lambda \|\theta\|_1 = L(\theta) + \lambda \sum_i |\theta_i|$$

The gradient of $L_1$ is constant ($\text{sign}(\theta_i)$), independent of the parameter magnitude. This means that no matter how small a parameter already is, $L_1$ pulls it toward zero with the same force. The result is a famous property: **$L_1$ regularization produces sparse solutions**—many parameters are pushed exactly to zero. With $L_2$, by contrast, the smaller the parameter, the weaker the pull toward zero (the gradient is $\lambda \theta_i$), so parameters tend to become small but not exactly zero.

The geometric distinction between $L_1$ and $L_2$ is clearest from the perspective of constrained optimization. Think of regularization as imposing a feasible-region constraint on parameter space—$L_2$ corresponds to a sphere (a circular constraint boundary), and $L_1$ corresponds to a diamond (a constraint boundary with sharp corners). The point where the loss function's contour lines (ellipses) are tangent to the constraint boundary is the regularized optimum. The $L_1$ diamond has sharp corners; loss contours are more likely to touch it at the coordinate axes—this is the geometric origin of sparsity.

You can visualize the terrain effect of data augmentation with a minimal numerical example. Consider a linear regression with two training samples: $(x_1=1, y_1=2)$ and $(x_2=2, y_2=4)$. Without augmentation, the loss function in parameter space is the intersection of two "grooves" — $w$ must simultaneously pass through $(1,2)$ and $(2,4)$; the unique solution is $w=2, b=0$, and the bowl floor is sharp.

Now apply a simple augmentation to each sample — add random jitter of $\pm 0.1$ to $x$. The effective number of samples goes from 2 to infinitely many — distributed across small regions around $x=1$ and $x=2$. The bottom of the loss bowl spreads from a single point into a region — any value of $w$ between $1.9$ and $2.1$ yields low average loss across all augmented variants.

The bowl floor has been "broadened" — transformed from a sharp minimum into a flat basin. And the model was never explicitly told "you should be insensitive to small input perturbations" — data augmentation naturally produced this effect in parameter space.


**Elastic Net.** A linear combination of $L_1$ and $L_2$:

$$L_{\text{total}}(\theta) = L(\theta) + \lambda_1 \|\theta\|_1 + \frac{\lambda_2}{2} \|\theta\|^2$$

It retains the sparsity advantage of $L_1$ while using the quadratic term of $L_2$ to smooth out the non-differentiability of $L_1$ at the origin. In practice, Elastic Net is often more stable than $L_1$ or $L_2$ alone.

![$L_1$ vs $L_2$ constraint geometry: the origin of sparsity](/figures/ch04_l1_vs_l2_geometry.svg)

*$L_2$ regularization (left): a circular constraint region tangent to elliptical loss contours. At the point of tangency, both parameters are nonzero—$L_2$ does not induce sparsity. $L_1$ regularization (right): a diamond-shaped constraint region with sharp corners. Loss contours are more likely to touch the diamond at a coordinate axis (a sharp corner)—this is the geometric root of $L_1$'s sparse solutions: at the optimum, $\theta_2 = 0$, the parameter is pushed exactly to zero. An X marks the unregularized, unconstrained optimum.*

## 4.5 Dropout: Randomly Destabilizing the Terrain

$L_1$ and $L_2$ modify the terrain directly in parameter space. But there is another class of regularization methods that do not modify the loss function itself, but instead **randomly modify the structure of the model**—so that the model cannot rely on any single feature or pathway.

**Dropout** is the most classic example. At each training iteration, a fraction $p$ of neurons are randomly "shut off" (their activations are set to zero). Forward propagation uses this randomly thinned network, and backpropagation updates only the surviving neurons. At test time, all neurons are active, but their activations are multiplied by $1-p$ to compensate for the sparsity during training.

The geometric meaning of Dropout is deep. It forces every neuron in the network not to depend on the co-presence of any other specific neuron—because it does not know which of its "colleagues" will be present next time. This drives the network to learn **redundant, distributed** representations rather than brittle, single-point-of-failure feature dependencies. From a terrain perspective, Dropout is equivalent to using a randomly perturbed loss function at each iteration—the terrain subtly deforms at every step. This persistent perturbation tends to push the model out of sharp minima (which depend on a precise configuration of weights) and guide it into broader, flatter basins (where performance does not collapse even when some neurons are deactivated).

**Stochastic Depth** is a layer-level generalization of Dropout: randomly skip entire residual blocks with a certain probability. **DropConnect** randomly masks weight connections rather than activations. The common theme of all these methods is: **by injecting randomness, prevent the model from memorizing the exact patterns of the training data, forcing it to learn more general structures.**

## 4.6 Data Augmentation: Inflating the Terrain in Data Space

Regularization need not be applied only in parameter space. Another line of thought: **do not change the model, do not change the loss function—change the data instead.**

Data augmentation artificially expands the size of the training set by randomly transforming training samples—flipping, rotating, cropping, color jittering, adding noise. A photo of a cat flipped horizontally is still a cat, but the pixel values are completely different. The model is required to perform consistently across these transformed samples, which is equivalent to injecting an "invariance prior" into the training process: the model learns to be insensitive to certain transformations.

From a terrain perspective, data augmentation imposes a kind of "smoothing effect" on the loss landscape in parameter space. Because the model cannot reduce loss by precisely memorizing the pixel positions of the original data—the same semantic content may appear tomorrow in mirrored or rotated form—sharp minima that depend on exact pixel configurations are systematically eliminated. Flat minima that depend on semantic features rather than pixel details are preserved.

This explains why data augmentation and Dropout are often used together: from different directions—data space and model structure—they jointly push the model toward flatter, more generalizable regions.


The deep conclusion of this chapter can be condensed into a single sentence: **the way you walk determines the fate of where you arrive.** Your learning rate determines how fast you can walk without falling — but its safety ceiling is set by the curvature at the steepest point of the terrain, and you will never know that curvature's exact value. Regularization pours cement into parameter space — filling in sharp crevices, preserving broad basins. Stochastic gradient noise does the same thing, but more quietly — at every step it implicitly filters for which basins are worth settling in. When you change optimizers, you are not changing "algorithms" — you are changing destiny.

## 4.7 Implicit Regularization: The Unexpected Gift of SGD Noise

Finally, we arrive at a profound observation: **you don't necessarily need explicit regularization. Gradient descent itself, especially stochastic gradient descent, comes with its own "implicit regularization."**

Recall the discussion in ch3 about sharp versus flat minima. Full-batch gradient descent (full-batch GD) uses the entire training set to compute the exact gradient; its trajectory is deterministic—the model follows the direction of steepest descent and slides precisely into whichever minimum is nearest the initial point, whether it is sharp or flat.

SGD is different. The randomness of mini-batches introduces gradient noise—each step's direction is an unbiased estimate of the true gradient, but with variance. This noise has a crucial geometric effect: **it is far more likely to "kick the model out" of sharp minima than out of flat minima.**

The intuition is this. Near a sharp minimum, the Hessian eigenvalues are large, the terrain curvature is high—a tiny perturbation in parameters causes a large increase in loss. The random gradient noise of SGD provides exactly this perturbation. The magnitude of the noise depends on the learning rate and batch size—the smaller the batch, the larger the noise. Near sharp minima, the noise easily exceeds the "escape threshold" and pushes the model out of the region. Near flat minima, the same noise magnitude causes a much smaller change in loss—the model is more likely to stay.

This means that even without any explicit $L_1$ or $L_2$ regularization, SGD itself has a tendency to select flat minima—and flat minima generally correspond to better generalization. **The larger the learning rate and the smaller the batch size, the stronger this "noise-driven flatness preference" becomes.**

This also explains why full-batch GD usually generalizes worse than SGD: full-batch GD has no noise; it naively slides into the minimum nearest the initial point, regardless of whether it is sharp or flat. The noise of SGD acts like a "terrain filter"—it screens out sharp solutions and keeps only flat ones.

This is implicit regularization. It was not designed; it emerged indirectly from the mathematical structure of stochastic gradient descent. It reminds us: **the choice of optimizer is not merely an efficiency question of "walking fast versus walking steadily"—it fundamentally determines what kind of solution you will arrive at.**

:::info

**Mr. Pallas's Cat's Position**

There is a stubborn superstition in the community: regularization is a "technique for preventing overfitting." You add $L_2$, you add Dropout, you do data augmentation—and then your validation loss stops rising. As if regularization were a "medicine" and overfitting were a "disease."

But from a geometric point of view, regularization is not "medicine" at all. It is **terrain modification**. $L_2$ superimposes a quadratic bowl centered at the origin onto parameter space—it presses the entire terrain downward, making distant regions more "expensive." Dropout randomly perturbs the terrain at every step—it smooths out sharp crevasses. Data augmentation inflates the terrain in data space—it lets countless variants of the same semantic content share the same low-loss region.

You think of these three things as three different "techniques." But they are doing the same thing: **making flat basins more likely endpoints than sharp crevasses.** Flatness is not a side effect of regularization—flatness is the entire purpose of regularization.

And SGD noise tells you something even deeper: **you don't even need explicit regularization.** Stochastic gradients themselves are a "terrain filter"—their noise is far more likely to kick the model out of a sharp minimum than out of a flat one. So SGD naturally prefers flat solutions. The larger the learning rate and the smaller the batch size, the stronger this preference.

You didn't "add regularization." You simply chose a way of walking—and that way of walking itself determines what kind of terrain you will settle on. The optimizer is not an efficiency tool. The optimizer is destiny.

:::

---

## 4.8 Chapter Summary

The theme of this chapter is "ways of walking"—but it ultimately arrives at a deeper conclusion: **walking is not merely descending along existing terrain. You can adjust the rhythm of your stride (learning rate schedules), you can reshape the ground beneath your feet (regularization), and you can even exploit the random noise inherent in walking itself to filter for better destinations (implicit regularization).**

The learning rate is the step size—it must be less than $2 / \lambda_{\max}$ to remain stable, and scheduling strategies (step decay, cosine annealing, warmup, cyclical) adapt the stride to the progress of training. $L_1$ and $L_2$ regularization directly superimpose constraints onto parameter space—the former induces sparsity (the diamond touches the ellipse at a coordinate axis), the latter uniformly shrinks all parameters. Dropout and data augmentation inject randomness from two different directions, driving the model away from fragile sharp minima into broad, generous flat basins. And the implicit regularization of SGD—the tendency of gradient noise to naturally favor flat solutions—reveals the deepest connection between optimization and generalization: **how you walk determines where you end up.**

In the next chapter, we will leave the Euclidean world and enter the non-Euclidean space of beliefs. Bregman divergences and KL divergence—these asymmetric, unconventional "distances"—will redefine what it means to be "near" or "far." There, the ways of walking will once again be overturned.

---

## Unresolved Questions

1. The stability condition for the learning rate $\eta < 2 / \lambda_{\max}$ depends on the largest eigenvalue of the Hessian. During the training of a deep network, $\lambda_{\max}$ itself changes dynamically. Can we estimate $\lambda_{\max}$ in real time during training and dynamically adjust $\eta$ to remain stable at all times?

2. Cosine annealing, warmup, and cyclical learning rates are widely used in practice, but their optimal parameters (cycle length, number of warmup steps, upper and lower bounds) are almost always determined by trial and error. Does there exist an automated scheduling strategy based on terrain geometry (e.g., the evolution of the Hessian eigenspectrum)?

3. The sparsity of $L_1$ regularization arises from the diamond constraint boundary touching the loss contours at a coordinate axis—but this requires the loss contour to "hit" the sharp corner exactly. In high-dimensional spaces, what is the probability of such an encounter? Are there scenarios where $L_1$ regularization fails to produce sparsity?

4. Dropout's "randomly thinned network" introduces noise during training, yet at test time the full network is used (multiplied by $1-p$). Does this training–testing asymmetry produce a systematic bias in parameter space? Does the training–testing equivalence of "inverted dropout" fully eliminate this bias?

5. Data augmentation constructs an "invariance prior"—the model learns to be insensitive to certain transformations. But if the augmentation method introduces an invariance that should not be there (e.g., applying vertical flips to handwritten digits, where 6 and 9 would be confused), might augmentation actually harm generalization?

6. SGD noise tends to select flat minima—is this a correlation or a causal relationship? Could there exist a sharp minimum that nevertheless generalizes well, whose generalization ability comes from some other geometric property unrelated to Hessian curvature?

7. Different optimizers (SGD vs Adam vs AdamW) rely on implicit regularization to different degrees—Adam's adaptive step size alters the noise structure. Does this partly explain why Adam generalizes worse than carefully tuned SGD on certain tasks (e.g., language modeling)?

8. If the strength of implicit regularization depends on the learning rate and batch size, is there an upper bound on "the larger the learning rate, the better the generalization"? Under what conditions does an excessively large learning rate turn from "beneficial noise" into "harmful oscillation"?

9. Weight decay ($L_2$ regularization) is not equivalent to traditional $L_2$ regularization in Adam—because Adam's adaptive step size changes the effective magnitude of the weight decay. AdamW's decoupled weight decay fixes this problem. Can this decoupling be generalized to other adaptive optimizers (e.g., RMSprop, AdaGrad)?

10. Regularization is essentially the introduction of prior knowledge—$L_2$ introduces the prior that "parameters should be small," data augmentation introduces the prior that "semantics are invariant under transformations." In what sense are these priors "helping" the model rather than "constraining" it? Does there exist a form of regularization that eliminates only harmful overfitting without restricting the model from learning genuinely useful patterns?

11. Can the "periodic heating" of cyclical learning rates be understood as a form of simulated annealing—periodically raising the temperature (large learning rate) to help the system escape local minima? If so, what should the optimal "temperature curve" look like?

12. If different optimizers have different implicit regularization effects, is "model merging" (taking a weighted average of two models trained with different optimizers) equivalent to some more complex regularization strategy? Can it combine the respective strengths of both optimizers?

---

**The core question left by this chapter is:**

**If the way of walking—step size, momentum, noise, regularization—all systematically changes the nature of the minimum ultimately reached, then should we regard "the choice of optimizer" as a prior rather than as a purely technical decision?**

:::info

**Walking is not merely descending along the gradient.** You can vary the rhythm of your stride (learning rate schedules), install guardrails along cliffs (regularization), and use the random sway of your footsteps to filter for steadier footing (implicit regularization). Once you have learned to walk, the next step is to learn to recognize the ground beneath your feet: Euclidean distance is not the only measure of distance. In belief space, the path from certainty to confusion and the path from confusion to certainty—these two roads have entirely different lengths. In the next chapter, we will enter the non-Euclidean world: Bregman Divergence and KL Divergence.

:::

## Further Reading and Related Work

**Sharpness-Aware Minimization.** Foret et al. (2021) [arXiv:2010.01412] — SAM reformulates optimization from "find the lowest point" to "find the flattest low point": a min-max objective that simultaneously minimizes loss value and Hessian sharpness, searching for basins where the worst-case loss within radius $\rho$ remains low — the most direct demonstration that "the way of walking determines where you arrive."

**Stochastic Gradient Descent as Approximate Bayesian Inference.** Mandt, Hoffman & Blei (2017) [arXiv:1704.04289] — Constant-learning-rate SGD traces a Markov chain whose stationary distribution approximates the Bayesian posterior. This connection reveals the probabilistic essence of SGD's implicit regularization: SGD is not "optimizing" but "sampling" — parameters end up in regions of high posterior probability, not at zero-loss points.

**Implicit Regularization and Generalization in Overparameterized Neural Networks.** Johannsen (2026) [arXiv:2604.07603] — Small-batch SGD systematically drives the dynamics toward flatter minima: empirical measurements show Hessian maximum eigenvalue differences of up to 11.8x. This "noise-induced flatness preference" is SGD's most fundamental implicit regularization mechanism relative to full-batch gradient descent.

**Convergence and Implicit Bias of Gradient Flow on Overparametrized Linear Networks.** Min et al. (2021) [arXiv:2105.06351] — The convergence rate of gradient flow is entirely determined by two geometric quantities: initialization imbalance and margin. The details of initialization — not the choice of optimizer — dictate "how fast" and "where to" the network moves.
