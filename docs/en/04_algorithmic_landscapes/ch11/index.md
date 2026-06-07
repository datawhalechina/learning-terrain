# Chapter 11: Geometry of Deep Architectures

In the previous chapter we looked at six classical algorithms. They share one commonality: **they operate in flat Euclidean space.** The elliptical bowl of linear regression is a quadratic form. The data cloud of PCA is a Gaussian ellipsoid. The hyperplane of SVM is linear. Their geometry is clear, elegant, analytically tractable.

In this chapter, we enter deep architectures. Here, space is no longer flat. Attention establishes **soft, data-dependent geometric connections** between tokens—not fixed distances, but gravitational pulls that change dynamically according to content. LoRA carves out a **low-rank subspace** in parameter space—the model moves within this subspace, rather than wandering freely through the entire high-dimensional parameter space.

These two architectures represent the two core geometric operations of deep learning: **bending the representation space** (Attention) and **constraining parameter motion** (LoRA). To understand their geometry is to understand the design language of modern deep architectures.

![Backpropagation chain rule](/figures/ch11_backprop_chain_graph_tikz.svg)

*Backpropagation chain rule. The gradient of loss $L$ flows downward: $\partial L/\partial y \rightarrow \partial L/\partial h \times \partial h/\partial \theta \rightarrow \nabla_\theta L \rightarrow$ update $\theta$. The chain rule is not a trick — it is the structural necessity of gradients flowing backward along the edges of the computation graph.*

![Attention: QKV Three-Axis Geometry](/figures/ch11_attention_qkv.svg)

*The Q (orange), K (blue), and V (green) axes define the projection geometry of attention in the representation space. The inner product of Query and Key vectors in the Q-K plane ($QK^T$) determines similarity—dashed lines connect each Query to its best-matched Key. Value vectors encode the "attended-to" content along the V axis. The projection matrices of the three learn, during training, a Mahalanobis metric between tokens.*

## 11.1 Attention: Soft Geometric Connections Between Tokens

Self-attention is the heart of the Transformer. Its formula—query, key, value—has been memorized countless times:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

But what is the geometry behind the formula?

Think of $Q, K, V$ as three "views" produced from the same input sequence $X$ via three different linear projections. $Q$ (Query) asks: "Whom should I attend to?" $K$ (Key) says: "I have these features—are you interested?" Their inner product $QK^T$ measures the **cosine similarity** between Query and Key in the projection space—the more aligned the directions, the stronger the attention.

Dividing by $\sqrt{d_k}$ prevents the variance of the inner product from exploding as dimension grows—a natural consequence of random vectors in high-dimensional space being nearly orthogonal (ch2, §2.4). Then softmax converts the similarities into a probability distribution—each token's attention weights over the other tokens.

Finally, $V$ (Value) provides the content being "attended to." The attention weights compute a weighted average over the rows of $V$—each token's new representation is an **attention-weighted combination** of the Values of all tokens.

The geometric perspective: **Attention builds a fully connected, weighted, data-dependent directed graph between tokens.** Tokens are nodes. Attention weights are edge strengths—pointing from Query tokens to Key tokens. This graph is recomputed at every layer, for every attention head. The projection matrices of $Q, K, V$—namely $W_Q, W_K, W_V$—define the "metric" on the graph: they determine which tokens are close to each other in the projection space.

![Softmax attention landscape](/figures/ch11_attention_geometry_tikz.svg)

*Softmax attention landscape. Sharp peaks emerge from flat inputs—the winner-take-all geometry of attention.*

![Multi-Head Attention: Parallel Subspaces](/figures/ch11_multihead_subspaces.svg)

*Three attention heads operating in parallel within three different projection subspaces. Head 1 (orange, syntactic) bends space to capture subject–predicate relations. Head 2 (green, semantic) captures synonym/antonym relations. Head 3 (red, positional) captures local dependencies between adjacent tokens. Each head is an independent geometric "viewpoint"; $W_O$ fuses all viewpoints at the end.*

## 11.2 Multi-Head Attention: Parallel Geometric Projections

A single attention head provides only one viewpoint. Transformers use **multi-head attention**—$h$ independent attention heads working in parallel, each with its own set of $Q, K, V$ projections.

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

where $\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$.

The geometric perspective: each attention head defines a **different subspace** within the representation space. Head 1 may attend to syntactic relations—connections between subjects and predicates. Head 2 may attend to semantic relations—synonyms and antonyms. Head 3 may attend to positional relations—local dependencies between adjacent tokens.

These subspaces are the column spaces of $W_Q, W_K, W_V$—each projection matrix maps the $d_{\text{model}}$-dimensional input into a $d_k$- or $d_v$-dimensional subspace (typically $d_k = d_v = d_{\text{model}} / h$). The $h$ heads compute attention simultaneously in $h$ different subspaces, then concatenate the results. This is equivalent to analyzing the same set of tokens in parallel under different geometric "viewpoints," and then fusing the multi-view information.

$W_O$ at the end maps the concatenated multi-head output back to $d_{\text{model}}$ dimensions—a "fusion" operation that reintegrates information from the different subspaces into a unified representation space.

## 11.3 Attention as a Learned Metric

Push the geometry of attention one layer deeper: **$QK^T$ defines not a fixed Euclidean inner product—but a data-dependent "similarity metric" that is learned during training.**

In Euclidean space, the similarity of two vectors is $\langle x, y \rangle = x^T y$. In attention, $x$ and $y$ are first projected into Query and Key spaces: $Qx$ and $Ky$. Their inner product in the projection space is:

$$(Qx)^T (Ky) = x^T (Q^T K) y$$

The matrix $Q^T K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ defines a **Mahalanobis metric**—a quadratic form that warps the notion of "similarity" in the original space. If $Q^T K$ were the identity matrix, we would be back to the standard Euclidean inner product. But $Q$ and $K$ are continuously adjusted during training—the model is **learning** a metric, rather than using a preset one.

This is the deep connection between attention and the Bregman divergences of ch5. Bregman divergence says: in probability space, the natural "distance" is not Euclidean—it is KL divergence. Attention says: in the space of token relations, the natural "similarity" is not the Euclidean inner product—it is the projected inner product learned by $Q$ and $K$. Both are about **abandoning a preset flat metric in favor of a curved metric shaped by data.**

![LoRA: Low-Rank Factorization](/figures/ch11_lora_factors.svg)

*Left: $A \in \mathbb{R}^{d \times r}$ (orange heatmap), $d=24$, $r=4$. Right: $B \in \mathbb{R}^{r \times d}$ (blue heatmap). Their product $AB$ is a matrix of rank at most $r$—it defines the low-rank subspace in which parameter updates reside.*

![LoRA Update Heatmap](/figures/ch11_lora_heatmap.svg)

*The full heatmap of $\Delta W = AB$. Red/blue blocks indicate positive/negative weight changes. Although there are $d \times d = 576$ elements, the matrix has rank only $r=4$—the vast majority of "degrees of freedom" are constrained within the low-rank structure. This is why LoRA is efficient.*

## 11.4 LoRA: Parameter Motion in a Low-Rank Subspace

From an entirely different direction, LoRA (Low-Rank Adaptation) also tells a story about geometry and constraint.

Fine-tuning a large model—training it further on a specific task—normally requires updating all parameters. For a 70B-parameter model, this means storing and optimizing 70B gradients. LoRA's insight is: **the parameter update matrix during fine-tuning is heavily low-rank.** You don't need to update the entire $W \in \mathbb{R}^{d \times d}$—you only need to update two small matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$, where $r \ll d$ (typically $r = 8$ or $16$):

$$W' = W + \frac{\alpha}{r} \cdot AB$$

The product $AB \in \mathbb{R}^{d \times d}$ is a matrix of rank at most $r$. It defines a **low-rank subspace** within the parameter space of $W$—parameter updates during fine-tuning can only move within this subspace.

The geometric perspective: full-parameter fine-tuning allows the model to move freely in the complete $\mathbb{R}^{d^2}$ parameter space—from one pretrained minimum toward any direction. LoRA restricts this freedom to a subspace of dimension $r(d + d) \ll d^2$. This subspace is not chosen arbitrarily—it is learned by $A$ and $B$ during fine-tuning. The columns of $A$ define the "principal directions of motion"; the rows of $B$ define the "weighted combination" of those directions.

This constraint is not a limitation—it is **implicit regularization** (ch4, §4.7). It confines parameter updates to a low-dimensional manifold, preventing the model from overfitting on the fine-tuning data while preserving the knowledge structure learned during pretraining.

## 11.5 Why LoRA Works: The Intrinsic Dimension Hypothesis

The reason LoRA works points to a deeper geometric fact: **the "intrinsic dimension" of task adaptation is far smaller than the dimension of the parameter space.**

For most downstream tasks—sentiment classification, summarization, code completion—adapting the model from its general pretrained state to a specific task only requires moving along a small number of directions in parameter space. These directions span a low-dimensional subspace—its dimension may be hundreds or thousands, not billions.

This is entirely consistent with the logic of "emergent depth" in ch9: **the terrain determines how many dimensions of motion are needed.** Simple terrain adaptation—fine-tuning to a task highly similar to pretraining—only requires fine-tuning in a few directions. Complex terrain adaptation—fine-tuning to a domain completely different from pretraining—may require more directions. LoRA's rank $r$ is a **geometric budget**: how many independent directions of parameter motion to allocate for terrain adaptation.

This also explains why LoRA's $r$ does not need to be large—on many tasks $r=8$ suffices. Not because "8 directions are always enough," but because the terrain of most fine-tuning tasks is smooth enough that only 8 independent directions are needed to walk from the pretraining basin to the task-specific basin.

## 11.6 Attention + LoRA: Unifying Bending and Constraint

Putting attention (11.1–11.3) and LoRA (11.4–11.5) together, we see a unified geometric picture.

Attention operates in **representation space**—it bends the similarity metric between tokens, building dynamic, data-dependent geometric connections. LoRA operates in **parameter space**—it constrains the directions of parameter updates, restricting fine-tuning to a low-rank subspace.

The two correspond respectively to the two core spaces of this book (ch2): attention reshapes the **field of vision**—how the model "sees" relationships between tokens. LoRA constrains the **body**—which directions the body can move in when adapting to a new task.

Their combination—Transformer + LoRA fine-tuning—is the most common paradigm in modern deep learning: use attention to build rich geometric structure in the representation space, and use LoRA for efficient, constrained adaptation in the parameter space. This is precisely the "body and field of vision" pairing from ch2, realized in engineering practice.

---

## 11.7 Chapter Summary

The geometry of deep architectures differs from that of classical algorithms. It is not flat elliptical bowls or Voronoi diagrams—it is curved, dynamic, data-dependent.

Attention establishes soft geometric connections between tokens—$QK^T$ defines not a fixed Euclidean inner product, but a Mahalanobis metric shaped by training data. Multi-head attention computes these connections simultaneously in multiple parallel subspaces. LoRA carves out a low-rank subspace in parameter space—restricting parameter motion during fine-tuning to a small number of directions, exploiting the geometric fact that "the intrinsic dimension of task adaptation is far smaller than the dimension of the parameter space."

Attention bends the field of vision. LoRA constrains the body. Together they form the geometric design language of modern deep architectures.

The next chapter—the final chapter of the book—will bring us to diffusion models. How does noise become structure? How can a process by which meaning emerges from randomness be described geometrically?

---

## Open Questions

1. $QK^T$ defines a Mahalanobis metric—but $Q$ and $K$ are learned independently, so $Q^TK$ is not necessarily symmetric. What does an asymmetric metric mean geometrically? "Similarity from A to B is not equal to similarity from B to A"—is there an intrinsic connection to the asymmetry of KL divergence?

2. Multi-head attention operates in different subspaces. Do these subspaces spontaneously "divide labor" during training—some specializing in syntax, others in semantics? Or is this division merely a human post-hoc interpretation?

3. The attention weight matrix is a stochastic matrix (each row sums to 1). How does its spectrum (eigenvalue distribution) evolve during training? Does attention become more "focused" (fewer nonzero eigenvalues) as training progresses?

4. LoRA assumes fine-tuning updates are low-rank—but does the optimal rank $r$ depend on the "distance" between the task and the pretraining data? How can the required rank be estimated before fine-tuning?

5. LoRA's parameter updates are confined to a low-rank subspace—consistent with the spirit of regularization in ch4 (constraining the directions of parameter motion). Can LoRA be understood as a form of "data-adaptive regularization"—where the subspace itself is learned from the fine-tuning data?

6. The softmax in attention outputs a probability distribution—entirely consistent with the softmax projection on the simplex in ch5. Can attention be reinterpreted within the belief-space framework (ch8)—where each attention head defines a "local vector field" within the reasoning field $F_x$ of the problem $x$?

7. In LoRA, why is $A$ typically initialized with Gaussian and $B$ with zero? What geometric position in parameter space does this initialization correspond to—fine-tuning starting from the pretraining fixed point, exploring only along the column directions of $A$?

8. The number of Transformer layers is preset (the "preset depth" of ch6). Does an "adaptive-depth Transformer" exist—one where each layer dynamically decides whether additional attention iterations are needed, analogous to the emergent depth of DEQ?

9. The $1/\sqrt{d_k}$ scaling in attention is a correction for the orthogonality of random vectors in high-dimensional spaces. Without this scaling, softmax would saturate—degenerating into one-hot attention (attending to only one token). What does this saturation correspond to geometrically—a "collapse" of the attention field?

10. Should LoRA use different ranks $r$ across different Transformer layers? Lower-layer representations are more "generic"—should their rank be smaller? Upper-layer representations are more "task-specific"—should their rank be larger?

11. Can multi-head attention and LoRA be unified—where the $Q, K, V$ projections of each attention head are fine-tuned within low-rank subspaces, and those subspaces are precisely the most sensitive "geometric directions" of that head?

12. Attention bends the representation space, LoRA constrains the parameter space. Does a unified mathematical framework exist—some kind of "parameter-space dual of attention"—that describes both in the same geometric language?

---

**The core question this chapter leaves behind is:**

**Attention learns a curved metric. LoRA discovers a low-rank subspace of motion—both are about finding low-dimensional structure in high-dimensional space. Is there a unified geometric principle that explains why nearly every effective operation in deep architectures—attention, LoRA, residual connections, layer normalization—is about "dimension reduction" or "dimension constraint"?**

:::info

**Attention bends the field of vision. LoRA constrains the body.** Classical algorithms operate in flat Euclidean space—elliptical bowls, data clouds, Voronoi diagrams. Deep architectures bend space itself—similarity is no longer a fixed inner product, but a Mahalanobis metric shaped by training data. Parameter motion is no longer free, but confined to low-rank subspaces. You thought you were designing the architecture—in truth, you are only discovering, within the geometry of parameters, those naturally existing low-dimensional structures. The final chapter of the book: diffusion models—how noise becomes structure. From randomness to order, from entropy increase to entropy decrease. There, the arc of the entire book will come to its close.

:::
