# Chapter 8: Reasoning Fields: Attractors and Verifiers

In the previous chapter we saw chains of thought—the trajectories of hidden states through representation space. But the trajectories are not random. They are guided by an invisible force field.

Imagine you are standing in a vast wilderness. You close your eyes and feel the slope beneath your feet. But this wilderness is different—it is not fixed. With every step you take, the slope beneath your feet is recalculated based on your new position. In some places, the slope pushes you in the same direction—those are **attractors**. In some places, the slope forms a shallow depression around you—you are trapped, those are **incorrect attractors**. In some places, the slope is extremely flat—you are lost, not knowing which way to go.

Every problem excites just such a wilderness in belief space—a **reasoning field**. It is a vector field that tells the model, at every point: "think this way." Some problems produce reasoning fields that are clean and crisp—a broad basin that pulls belief toward the uniquely correct answer. Other problems produce reasoning fields that are chaotic and rugged—multiple attractors competing with one another, the model possibly wandering among them.

What this chapter aims to do is to open up this reasoning field. Where does it come from? What structure does it have? Why do good answers have broad basins of attraction? How do verifiers fill in the pits of error—and how does RLHF reshape the entire terrain?

But first, let us push this metaphor far enough—far enough that it can bear the weight of the formal definitions to come.

Imagine you are standing in a vast wilderness, blindfolded. The only thing you can sense is the slope beneath your feet. You take a step—the slope tells you that leaning left means the ground rises, leaning right means it falls. You choose the descending direction. You take another step. The slope has changed—because you moved, and the ground beneath your feet is now a different patch. You choose the descending direction again.

If you walk long enough in this wilderness, you will eventually stop at a place where—no matter which way you shift—the slope pushes you back. You are standing at the **bottom of a basin**. You cannot leave—not because there is a wall, but because every direction is uphill.

Now, imagine this wilderness is not fixed. It varies by problem. You ask "What is the capital of France?"—the wilderness instantly excavates a deep, wide basin labeled "Paris." You ask "What is the meaning of life?"—the wilderness has no basin, only a plain with an impossibly gentle slope; you can walk in any direction and it makes little difference.

Imagine further—some basins are engraved with the wrong label. "What is the capital of France?"—a basin named "Romance" sits right next to the "Paris" basin, much shallower, but deep enough to capture a belief that wanders in from the wrong direction. Why does this basin exist? Because someone said that sentence in the training data. Not once. Many times.

A reasoning field is this wilderness. It is a vector field—an arrow planted at every point in belief space, telling you: "think this way." Correct answers have wide basins; wrong answers have narrow crevices; and what RLHF does, in essence, is—pour water into the correct basins to make them deeper, and shovel dirt into the incorrect crevices to make them disappear.

This is not metaphor. Below we turn it into formal definitions.

## 8.1 Formal Definition of the Reasoning Field

**Definition 1 (Reasoning Field)**. Given a model $\theta$ and a problem $x$, the **reasoning field** $F_x$ is a vector field defined on the probability simplex $\Delta^{K-1}$:

$$F_x: \Delta^{K-1} \to T\Delta^{K-1}$$

It maps each belief distribution $p \in \Delta^{K-1}$ to a tangent vector at that point—representing the model's "preferred update direction" in that belief state. Formally:

$$F_x(p) = \text{proj}_{T_p}\left(\mathbb{E}_{y \sim p_\theta(\cdot|p,x)}[\nabla_p \log p_\theta(y|p,x)]\right)$$

**Definition 2 (Orbit of the Reasoning Field)**. Given an initial belief $p_0$, the sequence $\{p_t\}$ produced by iterating Euler steps in the reasoning field is called an **orbit**:

$$p_{t+1} = \text{proj}_{\Delta}(p_t + \eta F_x(p_t))$$

where $\eta$ is the step size and $\text{proj}_{\Delta}$ is the Bregman projection onto the simplex. The implicit trajectory of the chain of thought (ch7) is precisely the process of traveling along some orbit in $F_x$.

With the definition of the field in hand, a natural question emerges: **what special positions exist in the field—those places where the model "stops in its tracks"?**

## 8.2 Classification of Fixed Points and Attractors

As with any dynamical system (ch6), the core structure of a reasoning field is determined by its fixed points and their stability.

**Definition 3 (Reasoning Fixed Point)**. A belief distribution $p^*$ satisfying $F_x(p^*) = 0$ is called a **fixed point** of the reasoning field $F_x$. At a fixed point, the model has no tendency to update its current belief—reasoning has stopped.

But not all fixed points are "good." Some are valleys—once you arrive, you can never leave. Some are peaks—you can never reach them. Some are mountain passes—you slide in along certain directions and slide out along others. To distinguish them, we need second-order information.

**Definition 4 (Stability Classification)**. At a fixed point $p^*$, consider the Jacobian matrix of $F_x$, $J(p^*) = \nabla_p F_x(p^*)$. Let its eigenvalues be $\lambda_1, \ldots, \lambda_{K-1}$:

- **Attractor**: All $\text{Re}(\lambda_i) < 0$. Nearby trajectories all converge to $p^*$. Correct attractors typically have a uniform distribution of eigenvalue magnitudes—the convergence rate is roughly the same in all directions, and the basin is approximately spherical.
- **Repeller**: All $\text{Re}(\lambda_i) > 0$.
- **Saddle point**: Some eigenvalues have positive real parts and some negative. Incorrect attractors often have extremely negative eigenvalues in certain directions (spurious "certainty") and near-zero eigenvalues in other directions (fragile, easily pushed out)—the basin is squashed flat.

**Definition 5 (Basin of Attraction)**. The **basin** $\mathcal{B}(p^*)$ of an attractor $p^*$ is the set of all initial beliefs $p_0$ such that the orbit converges to $p^*$. The wider the basin, the more initial beliefs the model can start from and still reach the correct answer—the higher the robustness.

:::info

**Pallas's Cat Professor: Why Wrong Answers Are Also "Right"**

There is a subtlety here that is easy to overlook. An attractor has no attribute of "correct" or "incorrect"—it is simply a fixed point of a vector field. Correct and incorrect are labels we humans attach after the fact.

But the model does not know which fixed point is called "the correct answer." The model only knows: the slope beneath my feet points in some direction, I follow it, I arrive at a place I cannot leave. I stop.

If that place happens to be the human-defined correct answer—we say the model "reasoned correctly." If that place happens to be a human-defined incorrect answer—we say the model "made a mistake." But from the model's perspective, the two processes are strictly symmetric in mathematics: both are gradient descents into a local energy minimum.

This is why the model confidently makes mistakes. It is not "guessing wrong"—it is walking, with exactly the same determinism, toward an incorrect fixed point. Its trajectory is mathematically just as smooth, just as continuous, just as convergent as the trajectory toward the correct fixed point. An incorrect belief is a form of **geometric correctness**—inside the wrong basin, all arrows point toward the basin center.
:::

These classifications answer "why does the model produce wrong answers"—but another question is more fundamental: **where do incorrect attractors come from?** They are not random—they are pits carved into the energy landscape by the statistical structure of the training data.

## 8.3 Why Are Wrong Answers Also Attractors?

**Theorem 1 (Existence of Incorrect Attractors)**. Let the empirical distribution of the training data be $\hat{P}_D$. For the cross-entropy loss $E(p) = \mathbb{E}_{(x,y) \sim D}[-\log p_y]$, any incorrect pattern $y_{\text{wrong}}$ that appears with sufficiently high frequency in the training data will produce a local minimum in $E(p)$—and thus an attractor in $F_x = -\nabla_p E(p)$.

**Geometric meaning**: Any systematic confusion in the training data—"Q: What is the capital of France? A: Romance."—as long as it appears with sufficiently high frequency, will carve a stable incorrect attractor into the reasoning field. The model is captured by it not because of "insufficient understanding," but because the statistical structure of the training data has dug a pit in the energy landscape. Every systematic error you write into the training data is a groove you carve into the reasoning field.

This is more than a theoretical observation—it is a geometric indictment of training data quality. If we accept this framework, then "data cleaning" is no longer an empirical operation ("remove dirty data, the model gets better"), but a precise geometric surgery: **every systematic error cleaned is one incorrect basin filled in the reasoning field.** You do not need to understand why the model makes the mistake—you only need to know where the local depression in the energy landscape comes from, then delete or correct the source data responsible for that depression. The terrain will level itself.

This raises a practically crucial follow-up question: **can we fill in these error grooves?** Verifiers are born for exactly this purpose.

## 8.4 Formalization of the Verifier: Repulsive Field Superposition

**Definition 6 (Verifier)**. A verifier is a function $V: \Delta^{K-1} \times \mathcal{X} \to \mathbb{R}$ that, given a belief state $p$ and a problem $x$, outputs a scalar assessment—the higher $V(p, x)$ is, the more "correct" the belief.

Integrating a verifier into the reasoning field is equivalent to superimposing the verifier's gradient onto the original vector field:

$$F_x^{\text{ver}}(p) = F_x(p) + \eta_v \nabla_p V(p, x)$$

**Geometric effect**: At an incorrect attractor $p_{\text{wrong}}$, the original field $F_x(p_{\text{wrong}}) = 0$ (it is a fixed point). The verifier knows that $p_{\text{wrong}}$ is wrong—$V(p_{\text{wrong}}, x)$ is low, and its gradient $\nabla_p V$ points away from $p_{\text{wrong}}$. After superposition, $F_x^{\text{ver}}(p_{\text{wrong}}) \neq 0$—the original fixed point is eliminated. The incorrect basin is filled in.

**Theorem 2 (Sufficient Condition for Verifier to Eliminate an Incorrect Attractor)**. If the verifier satisfies $\|\eta_v \nabla_p V(p_{\text{wrong}}, x)\| > \max_{\lambda \in \sigma(J)} |\text{Re}(\lambda)| \cdot \text{diam}(\mathcal{B})$, then $p_{\text{wrong}}$ is no longer a fixed point in $F_x^{\text{ver}}$.

A verifier superimposes a repulsive field at individual reasoning steps. But the verifier requires an external evaluation signal (whether the answer is correct). In many tasks—writing, translation, dialogue—there is no objective "correct answer." In these cases, we need a more fundamental way to reshape the terrain.

## 8.5 Formalization of RLHF: Energy Function Modification

**Definition 7 (RLHF-Modified Energy Function)**. Let $E_{\text{orig}}(p)$ be the energy function defined during the pretraining phase. RLHF superimposes on it a correction term derived from a human preference reward model $R(p, x)$:

$$E_{\text{RLHF}}(p) = E_{\text{orig}}(p) - \lambda \cdot R(p, x)$$

The corresponding reasoning field is $F_x^{\text{RLHF}}(p) = F_x(p) + \lambda \nabla_p R(p, x)$.

**Theorem 3 (Redistribution of Fixed Points Under RLHF)**. Let $F_x$ have a set of fixed points $\{p_1^*, \ldots, p_m^*\}$. The reasoning field after RLHF preserves the same number of fixed points (in general position), but redistributes their stability: human-preferred fixed points ($R$ high) have deeper, wider basins; human-dispreferred fixed points ($R$ low) have shallower basins or even disappear.

The key point is: **RLHF does not create new fixed points—it redistributes energy among existing fixed points.** This is a direct corollary of the Banach fixed-point theorem: under the condition that the set of fixed points remains unchanged, the only thing that can be altered is the stability of each fixed point. RLHF does not change "what the model can answer"—it changes "what the model prefers to answer."

:::info

**Pallas's Cat Professor: RLHF Is Not Education, It's Terrain Engineering**

This conclusion has an unsettling corollary. RLHF does not create new capabilities—it merely redistributes the depth of capability basins already present in the model. If the model never encountered a certain concept during pretraining, RLHF cannot "teach" it—because there is no corresponding fixed point to deepen.

This means RLHF's terrain modification has a hard ceiling: **you can only reshape existing terrain; you cannot conjure a mountain out of nothing.** A model that was never exposed to calculus in its pretraining data will never output a correct integral, no matter how much RLHF you do—because there is no "calculus" basin to deepen. A model that absorbed biased data during pretraining can have its bias basins filled in by RLHF—but RLHF cannot create a "fairness" basin that never existed.

This is not a failure of RLHF. This is Banach's honesty. The fixed-point theorem does not lie: the set of fixed points satisfying $F_x(p^*) = 0$ remains unchanged after RLHF. You can only redistribute stability and basin depth.

So the true identity of RLHF is not "making the model better"—it is **cosmetic terrain surgery**. It digs the beautiful basins deeper and fills in the ugly ones. But it does not create new basins. If you want to give the model a new basin, you must go back to pretraining—back to the data—and dig that pit with your own hands.
:::

At this point, we have a complete language for reasoning fields—the definition of the field, the classification of fixed points, the origin of incorrect attractors, and the reshaping mechanisms of verifiers and RLHF. But one question remains: **can we use this language, before deployment, to diagnose a model's reasoning capability?**

## 8.6 Geometric Diagnostics of the Reasoning Field

Three diagnostic metrics, all defined by geometric properties of $F_x$:

**Diagnostic Metric 1 (Correct Basin Coverage)**: The proportion of the simplex's total volume occupied by the basin of the correct attractor. The higher the coverage, the stronger the model's robustness to prompt variation.

**Diagnostic Metric 2 (Basin Separation)**: The KL divergence between the correct basin and the nearest incorrect basin. The larger the separation, the lower the probability that the model will "hesitate" between the two answers.

**Diagnostic Metric 3 (Field Strength Uniformity)**: The variance of the magnitude $\|F_x(p)\|$ of $F_x(p)$ across the simplex. Excessive variance means that in some regions the field strength is extremely weak (the model gets lost), while in other regions the field strength is extremely strong (unstable jumps).

These metrics are currently rarely measured in practice—but they provide a geometric assessment complementary to behavioral testing for determining "whether a model is ready."

---

## 8.7 Chapter Summary

The reasoning field $F_x$ is a vector field on belief space—it determines where the chain-of-thought trajectory starting from any initial belief will flow. Fixed points are positions where the model "no longer changes its belief." Correct attractors have broad basins and uniform convergence rates. Incorrect attractors are local energy minima carved by systematic confusions in the training data.

Verifiers superimpose repulsive fields around incorrect attractors—filling in the traps. RLHF is more fundamental—it directly modifies the energy function $E(p)$, redistributing stability among all fixed points. The two find a unified formalization within the framework of the Yonglin Limit: they do not change the fact that "reasoning necessarily converges"—they only change **which** fixed point the model converges to.

---

## Unresolved Questions

**The core question left by this chapter is:**

**If the reasoning field $F_x$ is an object that can be formally defined, numerically computed, and geometrically diagnosed in belief space—then can we establish a "reasoning topography," such that a model's reasoning behavior is no longer a black box, but an explicit landscape that can be explored, measured, and reshaped?**

:::info

**You do not choose the answer. The field chooses for you.** Every problem lays down an invisible net in belief space—fixed points, basins, saddle points, repellers. Your reasoning trajectory flows along the field's arrows until it falls into the bottom of some basin. Verifiers fill in the incorrect basins. RLHF broadens the correct basins. You think it is you who is reasoning—but in truth, you are merely moving under the field's guidance, along paths it has long since drawn, toward destinations it has long since decided. This is precisely the final teaching of the Yonglin Limit: **convergence is not hope—convergence is geometry.**

:::
