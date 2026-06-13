# Volume IV: Algorithmic Landscapes

Every formula you've ever learned is a contour line of a terrain.

$\hat{\beta} = (X^TX)^{-1}X^Ty$ for linear regression—you thought it was an algebraic derivation. It's not. It's the bottom coordinate of an elliptical bowl. The eigenvectors of PCA—you thought they were the result of covariance matrix factorization. They're not. They're the directions of the longest axes of the data cloud. The maximum margin of SVM—you thought it was the dual solution of constrained optimization. It's not. It's the widest gap between two clusters of data.

The three chapters of Volume IV take the entire geometric language you've built over the previous nine chapters and apply it back to the algorithms you know best—and you'll discover you've never truly seen them.

---

Chapter 10 re-examines six classical algorithms—linear regression, logistic regression, PCA, SVD, SVM, K-means. Behind every formula you memorized lies a geometric object. Linear regression slides to the bottom of an elliptical bowl. PCA finds the longest axis of the data cloud. SVM cleaves the widest gap between two clusters. K-means partitions space into a Voronoi tessellation. Six algorithms, six different mathematical forms—one soul: **in a space, along the gradient of some geometric quantity, walk toward an extremum.**

Chapter 11 enters deep architectures. Classical algorithms operate in flat Euclidean space. Deep architectures bend space itself. Attention is not "computing attention weights"—$Q^T K$ defines a Mahalanobis metric learned during training. The "similarity" between every pair of tokens is not discovered—it is carved by backpropagation. LoRA is not a "memory-saving trick"—it reveals a geometric fact you're unwilling to admit: adapting to a new task only requires moving along 8 low-rank directions. The vast majority of the 175 billion parameters you spent millions training are maintaining the smoothness of the terrain.

Chapter 12—the book's destination. Diffusion models. Forward: structure dissolves into noise (entropy increase). Reverse: noise converges to structure (entropy decrease). The score function is a vector field in data space—it points toward "where there is more data." Reverse diffusion is an Euler-step trajectory in this vector field. Generation is the trajectory converging to the data manifold. **337 years. From $F=ma$ to diffusion models. The same arc: Force → Energy → Terrain → Motion → Fixed Point → Structure.**

---

Volume IV is not "application." Volume IV is **coming home**. You've walked nine chapters, building an entire geometric language. Now, use that language to re-examine every algorithm you thought you understood—and discover they have never been clearer.

---

**Volume IV: Algorithmic Landscapes**
- [Chapter 10: Geometry of Classical Algorithms](./ch10/)
- [Chapter 11: Geometry of Deep Architectures](./ch11/)
- [Chapter 12: Diffusion and Convergence](./ch12/)
