# Chapter 10: Geometry of Classical Algorithms

You've studied linear regression. You've studied PCA. You've studied SVM.

But you understood them through formulas—$\hat{\beta} = (X^TX)^{-1}X^Ty$, $\max \frac{2}{\|w\|}$, $\text{tr}(W^T C W)$. You remember these formulas, you can derive them, you used them correctly on exams.

But you've never **seen** them.

The closed-form solution of linear regression—you treat it as an algebraic result. But it is in fact a geometric fact: you are at the bottom of an elliptical bowl. The covariance matrix decomposition of PCA—you treat it as a linear algebra exercise. But it is in fact a geometric fact: you have found the longest axis of the data cloud. The maximum margin of SVM—you treat it as a constrained optimization problem. But it is in fact a geometric fact: you have found the widest gap between two classes of data.

The three chapters of Volume IV aim to transform these "algebraic formulas" into "geometric intuition." This chapter begins with the most classical algorithms—linear regression, logistic regression, PCA, SVD, SVM, K-means. The next chapter enters deep architectures—Attention and LoRA. The final chapter arrives at diffusion models—how noise becomes structure, bringing the book to a close.

## 10.1 Linear Regression: The Bottom of an Elliptical Bowl

Linear regression is the simplest learning problem—yet it already contains all the essential elements of geometry.

Given a data matrix $X \in \mathbb{R}^{n \times d}$ and a label vector $y \in \mathbb{R}^n$, we want to find weights $\theta \in \mathbb{R}^d$ such that the predictions $\hat{y} = X\theta$ are as close as possible to $y$. Minimize the mean squared error:

$$L(\theta) = \frac{1}{n} \|y - X\theta\|^2$$

Expand the quadratic form: $L(\theta) = \frac{1}{n}(\theta^T X^T X \theta - 2y^T X \theta + y^T y)$. This is a perfect quadratic function in $\theta$—it contains no higher-order terms. $L(\theta)$ defines an **elliptical bowl** in parameter space.

The shape of the bowl is determined by $X^T X$ (the Hessian). The eigenvalues of $X^T X$ give the curvature of the bowl in different directions. Large eigenvalues correspond to steep directions—along these directions, the gradient is large and convergence is fast. Small eigenvalues correspond to shallow directions—along these directions, the gradient is small and convergence is slow.

The bottom of the bowl—the global minimum—has an elegant closed-form solution: $\theta^* = (X^T X)^{-1} X^T y$. From a geometric perspective, this formula is not an "algebraic derivation"—it is **the location where the gradient of the bowl is zero**. Set $\nabla L(\theta) = \frac{2}{n} X^T (X\theta - y) = 0$, and solve directly for $\theta^*$.

Gradient descent on this elliptical bowl is a process of perfect exponential convergence. Starting from any initial point $\theta_0$, the error at each step decays by a fixed ratio:

$$\|\theta_t - \theta^*\| \leq (1 - \eta \lambda_{\min})^t \|\theta_0 - \theta^*\|$$

where $\lambda_{\min}$ is the smallest eigenvalue of $X^T X$. This is the purest instance of $L$-smoothness and $\mu$-strong convexity from ch3—the entire landscape is a uniformly curved bowl, with no saddle points, no local minima, no plateaus. Linear regression is "simple" not because it has few parameters—but because its loss landscape is a perfect elliptical bowl.

## 10.2 Logistic Regression: The Decision Boundary on the Probability Surface

Moving from regression to classification, the loss function changes from MSE to cross-entropy. The landscape transforms from an elliptical bowl into a more subtle object.

Logistic regression maps the linear output $X\theta$ through the sigmoid function $\sigma(z) = 1/(1 + e^{-z})$ to probabilities. The loss is cross-entropy:

$$L(\theta) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log \sigma(x_i^T \theta) + (1-y_i) \log(1 - \sigma(x_i^T \theta)) \right]$$

The key geometric difference from linear regression: **the loss function of logistic regression is still convex—but it is not quadratic.** Its Hessian is not a constant matrix—it depends on the current position $\theta$. In regions dense with data points, curvature is large—the terrain is steep. In regions with sparse data points, curvature is small—the terrain is shallow.

The sigmoid function creates a **probability surface** in output space. $\theta^T x = 0$ defines the decision boundary—a $(d-1)$-dimensional hyperplane. On this hyperplane, the model outputs probability 0.5—complete uncertainty. On one side of the hyperplane, the probability approaches 1; on the other side, it approaches 0. The steepness of the sigmoid is controlled by the norm of $\theta$—the larger $\|\theta\|$, the steeper the transition, the "harder" the decision boundary.

This gives the logistic regression landscape an adaptive curvature: near the decision boundary, the loss is extremely sensitive to parameter changes—a small step can flip a prediction. Far from the decision boundary, the loss is almost insensitive to parameter changes—the predictions are already "saturated." This is a geometric richness that the uniform elliptical bowl of linear regression lacks.

![PCA: Principal Axes of the Data Cloud](/figures/ch10_pca_ellipsoid.svg)

*A two-dimensional data cloud and its covariance ellipse. The two principal components align with the directions of maximum variance. The first principal component (orange) captures the greatest variance, and the second principal component (blue) is orthogonal to it.*

## 10.3 PCA: The Principal Axes of the Data Cloud

PCA involves no loss function and no labels. It is pure **unsupervised geometry**—concerned only with the shape of the data itself.

Given a centered data matrix $X \in \mathbb{R}^{n \times d}$, PCA seeks a set of orthonormal directions $v_1, v_2, \ldots, v_k$ such that the variance of the data projected onto these directions is maximized. Equivalently, PCA finds the top $k$ eigenvectors of the data covariance matrix $C = \frac{1}{n} X^T X$.

Geometric intuition: the data points form a **data cloud** in high-dimensional space. This cloud is roughly ellipsoidal. PCA finds the **principal axes** of this ellipsoid—the longest axis (first principal component), the second-longest axis (second principal component), and so on. The direction of each principal axis is given by an eigenvector of $C$, and the length (variance) of that axis is given by the corresponding eigenvalue.

In the geometric language of parameter space, PCA is equivalent to: approximating a high-dimensional data cloud with a low-dimensional hyperplane, such that the variance after projection is maximized. The normal vectors of this hyperplane are the discarded directions. From the perspective of information geometry, PCA does in Euclidean space what KL divergence does in probability space—it finds the "directions of greatest change."

PCA is the simplest form of representation learning. It involves no nonlinearity, no labels, no gradient descent—but its geometric essence runs through every chapter of this book: **find the directions of greatest variation in space, compress the data along these directions, preserve the most important structure, discard the redundant dimensions.**

![SVD: Topographic Decomposition of a Matrix](/figures/ch10_svd_decomposition.svg)

*The action of an arbitrary matrix $A$ decomposes into three steps: $V^T$ rotates the unit circle → $\Sigma$ scales along coordinate axes → $U$ rotates again. The decay rate of the singular value spectrum determines the "effective rank" of the matrix—LoRA (ch11) exploits precisely this fact.*

## 10.4 SVD: The Topographic Decomposition of Matrices

PCA is a special case of SVD. But SVD itself has richer, more independent geometric meaning.

Any matrix $A \in \mathbb{R}^{m \times n}$ can be decomposed into the product of three matrices:

$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors—an orthonormal basis for the column space of $A$. They are the eigenvectors of $AA^T$.
- $\Sigma \in \mathbb{R}^{m \times n}$: a diagonal matrix, with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ on the diagonal. The singular values are the "scaling factors" of $A$ in each direction.
- $V^T \in \mathbb{R}^{n \times n}$: right singular vectors—an orthonormal basis for the row space of $A$. They are the eigenvectors of $A^T A$.

Geometric interpretation: the action of any matrix $A$—mapping a vector $x$ to $Ax$—can be decomposed into three steps: $V^T$ rotates the coordinate system, $\Sigma$ scales along the new axes, and $U$ rotates again. **Any linear transformation, in essence, is nothing more than a set of rotations and scalings.**

The **spectrum** of singular values—$\sigma_1, \sigma_2, \ldots$—is the core geometric feature of matrix $A$. The rate at which the singular values decay determines the "effective rank" of $A$—how many independent, non-zero scaling directions there are. In deep learning, the singular value spectrum of weight matrices determines the "information capacity" of that layer's transformation. LoRA (ch11) exploits precisely this fact: the parameter update matrices during fine-tuning are severely low-rank—their singular value spectrum decays rapidly to zero after a few large values.

![SVM: Maximum Margin Geometry](/figures/ch10_svm_margin.svg)

*The maximum margin hyperplane between two classes of data. Dashed lines mark the margin boundaries. Circles mark the support vectors—the points that uniquely determine the hyperplane's position. The green band is the margin region.*

## 10.5 SVM: The Geometry of Maximum Margin

SVM finds a hyperplane to separate two classes of data—but not just any hyperplane. It seeks the **maximum margin** hyperplane—the one farthest from the nearest data points (the support vectors).

Formally, for linearly separable data, SVM solves:

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1, \forall i$$

The objective is to minimize $\|w\|^2$—this is equivalent to maximizing the margin $2/\|w\|$. The constraints ensure that every point is correctly classified and at least distance $1/\|w\|$ from the hyperplane.

Geometric perspective: the two classes of data points form two clouds in space. The hyperplane found by SVM is the **widest gap** between these two clouds—the midline of a band containing no data points. The data points that support the boundaries of this gap are the **support vectors**—they are the points closest to the hyperplane, and the only points that determine its position. All other points—no matter how you move them, as long as they don't cross the gap—do not affect the solution.

SVM is the most elegant manifestation of constrained optimization in geometry. It turns a classification problem into a **geometric gap maximization** problem. In deep learning, this "maximum margin" idea reappears in various forms—the temperature parameter in contrastive learning controls the "gap" between positive and negative samples in representation space, and triplet loss explicitly requires the distance from anchor to positive to be smaller than the distance from anchor to negative plus a margin.

![K-means: Voronoi Tessellation](/figures/ch10_kmeans_voronoi.svg)

*Four centroids (diamonds) partition space into Voronoi regions. Boundaries are the perpendicular bisectors of lines connecting adjacent centroids. Different initializations correspond to different tessellations—different local minima.*

## 10.6 K-means: The Geometry of Space Partitioning

K-means is the most straightforward clustering algorithm. Given $n$ data points and a preset number of clusters $K$, K-means alternates between two steps: assign each point to the nearest centroid; update each centroid to the mean of its assigned points.

Geometric perspective: K-means performs a **Voronoi tessellation** of the data space. Each centroid $\mu_k$ defines a Voronoi region—the set of all points closer to $\mu_k$ than to any other centroid. The boundaries between regions are the **perpendicular bisectors** (perpendicular bisecting hyperplanes in higher dimensions) of the lines connecting adjacent centroids.

K-means's loss function—the within-cluster sum of squares—is non-convex in the centroid positions. It has many local minima, corresponding to different tessellations. This is why the results of K-means depend on initialization—different initial centroids partition space into different Voronoi diagrams, and gradient descent slides along different paths into different local minima. K-means is the perfect instance of ch3's "landscape determines the destination" in unsupervised learning.

## 10.7 One Geometry, Seven Languages

Linear regression, logistic regression, PCA, SVD, SVM, K-means. Six algorithms, six different mathematical forms—but they share a single geometric soul: **in a space, find a direction or a position that extremizes some geometric quantity.**

Linear regression minimizes the sum of squared vertical distances. Logistic regression maximizes likelihood—equivalent to minimizing cross-entropy, finding the best separating boundary in probability space. PCA maximizes projection variance—finding the principal axes of the data cloud. SVD decomposes rotations and scalings—revealing the intrinsic geometric structure of a linear transformation. SVM maximizes the classification gap—finding the widest gap. K-means minimizes the within-cluster sum of squares—partitioning space into Voronoi regions.

These algorithms appeared in different eras, written in different mathematical languages. But when you draw their landscapes in parameter space or data space—elliptical bowl, probability surface, data ellipsoid, Voronoi diagram—you discover that they are all doing the same thing: **following the gradient of some geometric quantity, heading toward the position of some extremum.**

The next chapter will leap from classical algorithms into deep architectures. Attention is not "computing attention weights"—it is a soft geometric connection between tokens. LoRA is not "low-rank matrix factorization"—it is a low-rank subspace motion in parameter space.

:::info

**The Pallas's Cat Professor's Position**

You once derived $\hat{\beta} = (X^TX)^{-1}X^Ty$ on an exam and thought you understood linear regression. What did you understand? You just moved symbols from one side of the equation to the other.

You never asked: what is the $X^TX$ in this formula? It's a $d \times d$ matrix—it's the Hessian of an elliptical bowl carved into parameter space by the data cloud. What is $X^Ty$? It's the vector pointing from the center of the data cloud toward the label direction. You multiply them together—and you thought you were doing algebra. But you were actually finding the bottom coordinate of an elliptical bowl.

Machine learning textbooks have spent a hundred years teaching people to derive these formulas. But not a single one tells you: **behind every formula you derive, there is a geometric object.** $\hat{\beta}$ is not the product of a formula—$\hat{\beta}$ is the bottom of a bowl. The eigenvectors of PCA are not "the result of covariance matrix factorization"—they are the directions of the longest axes of the data cloud. The margin of SVM is not "the dual solution of constrained optimization"—it is the widest gap between two clusters of data.

Algebra lets you derive. Geometry lets you see. You've been deriving these formulas since your freshman year—four years of derivation, possibly longer—but you've never truly seen them. This is not your fault. It's the textbooks' fault. The textbooks only teach you how to calculate; they never teach you to look up.

The single task of Volume IV is this: **redraw every formula you've ever memorized as a topographic map.** You will discover that beneath all those algebraic symbols you've been grinding on for a decade, there has always been a landscape. It has always been there. You just never lifted your head to look.

:::

---

## 10.8 Chapter Summary

This chapter used six classical algorithms to demonstrate a single proposition: **behind every algebraic formula lies a geometric fact.**

The closed-form solution of linear regression is the bottom of an elliptical bowl. The decision boundary of logistic regression is the watershed on a probability surface. The principal components of PCA are the long axes of the data cloud. The singular values of SVD are the scaling factors of a linear transformation in each direction. The maximum margin of SVM is the gap between two classes of data. The centroids of K-means are the centers of a Voronoi tessellation.

Every formula you've memorized—$\hat{\beta} = (X^TX)^{-1}X^Ty$, $\max \frac{2}{\|w\|}$—corresponds to a geometric object in parameter space or data space. Learning these algorithms should not be about remembering their formulas—it should be about seeing their landscapes.

---

## Unresolved Questions

1. The elliptical bowl of linear regression is perfect—but real-world data is almost never linear. After a nonlinear feature transformation, does the loss landscape still preserve some kind of "approximate bowl shape"—or does it become a rugged non-convex terrain?

2. The Hessian of logistic regression depends on the current parameter position $\theta$. During optimization, how does the eigenspectrum of the Hessian evolve? Does an "optimal path" exist that keeps the curvature within a favorable range at all times?

3. PCA assumes that principal components are orthogonal. Under what conditions is this assumption geometrically justified? If the data manifold is curved, do the linear principal axes of PCA still carry meaning—or is nonlinear dimensionality reduction (such as autoencoders) necessary to "unfold" the manifold?

4. The decay rate of SVD singular values determines the "effective rank" of a matrix. In the weight matrices of deep networks, how does the singular value spectrum change during training? As training progresses, do singular values concentrate into a few large values—indicating that the network has "learned a more compact representation"?

5. The maximum margin of SVM is determined solely by the support vectors. In deep representation space, do analogous "support representations" exist—where a few critical hidden states determine the classification boundary?

6. The Voronoi tessellation of K-means depends on Euclidean distance. If the intrinsic metric of the data is not Euclidean (e.g., cosine similarity in text embeddings)—what does the Voronoi diagram become? Does a "Bregman K-means" exist—using KL divergence in place of Euclidean distance?

7. All of these classical algorithms have closed-form solutions or convex optimization guarantees—they "guarantee convergence to the global optimum." Deep networks have lost these guarantees. Is it possible to build a bridge between deep networks and classical algorithms—for example, proving that certain deep architectures are equivalent, under specific conditions, to the nonlinear generalization of some classical algorithm?

8. Classical algorithms work well on low-dimensional data but often fail on high-dimensional data (the curse of dimensionality). Deep networks "reduce dimensionality" through representation learning—mapping high-dimensional raw data onto low-dimensional manifolds. From a geometric perspective, can deep networks be understood as a cascade of "first do nonlinear PCA, then do linear classification"?

9. PCA and autoencoders are both dimensionality reduction methods—but PCA is linear, while autoencoders are nonlinear. What is the essential geometric difference between their "latent spaces"? Under what conditions does the latent space manifold of an autoencoder degenerate into the linear subspace of PCA?

10. The kernel trick of SVM maps data into a high-dimensional space, where a linear separation is sought. How is this fundamentally different from the "representation learning" of deep networks? Do deep networks learn a "data-adaptive kernel"?

11. K-means assumes that each cluster is spherical. In representation space, classes may form arbitrary manifolds. Does a "manifold K-means" exist—using geodesic distance in place of Euclidean distance to define "nearest centroid"?

12. If we draw all the geometries of classical algorithms on a single map of parameter space—elliptical bowl, probability surface, data ellipsoid, Voronoi diagram—do there exist some kind of "geometric duality" relations among them? For example, is there an intrinsic mathematical connection between the principal axes of PCA and the maximum margin direction of SVM?

---

**The core question left by this chapter is:**

**The geometry of classical algorithms is clear—elliptical bowl, data cloud, Voronoi diagram. The geometry of deep networks is complex—non-convex, high-dimensional, filled with saddle points and basins. Can the chasm between the two be bridged by a single geometric language?**

:::info

**Every formula you've ever learned is a contour line of a landscape.** The $\hat{\beta} = (X^TX)^{-1}X^Ty$ of linear regression is the coordinate of the bottom of an elliptical bowl. The eigenvectors of PCA are the directions of the longest axes of the data cloud. The maximum margin of SVM is the widest gap between two classes of data. When you look at formulas, you see algebraic symbols. When you lift your head—you see mountains, basins, gaps, and axes. This is geometry. Volume IV has just begun. Next chapter: Attention and LoRA—the geometry of deep architectures.

:::

## Further Reading and Related Work

This chapter's geometric perspective places classical algorithms back into the space where they naturally belong—the terrain of parameter space and data space. The following works extend this viewpoint in different directions.

**Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes.** Novak et al. (2019) [arXiv:1810.05148]——Infinitely wide CNNs are equivalent to Gaussian processes in distribution. This result builds a bridge between classical kernel methods and deep networks: when you push width to infinity, the uncertainty quantification of your network degenerates back to a kernel method you have known for a hundred years.

**Neural Tangent Kernel: A Survey.** Golikov et al. (2022) [arXiv:2208.13614]——A comprehensive survey of the Neural Tangent Kernel. In the infinite-width limit, the training dynamics of deep networks are entirely described by the NTK—gradient descent in function space becomes kernel ridge regression. Classical algorithms are not replaced by deep networks; they are embedded as limiting cases within deep network theory.

**Batch Normalization.** Ioffe & Szegedy (2015) [arXiv:1502.03167]——Layer-wise normalization redefines the coordinate scale in parameter space. From the geometric perspective of this chapter, BN keeps the condition number of the Hessian manageable—preventing the terrain from becoming too rugged or too flat during training. It is not an optimization trick; it is a curvature regulator.

**Explaining and Harnessing Adversarial Examples.** Goodfellow, Shlens & Szegedy (2015) [arXiv:1412.6572]——The linear explanation of adversarial examples. The local linearity of deep networks means that along directions where parameter space is locally too flat, a tiny perturbation in input space can cause a huge change in output. Adversarial vulnerability is the flip side of smoothness—the smoother you are, the easier you are to mislead.
