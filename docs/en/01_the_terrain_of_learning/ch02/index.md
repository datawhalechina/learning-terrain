# Chapter 2: Body and Vision: Parameter Space and Representation Space

At the end of the previous chapter, we left a question hanging: **if learning is a kind of motion, then what is it that moves? And what does it see?**

These two questions point to two sides of the same thing. A deep learning model is not a ghost floating in a void — it has a body (its parameters), and this body occupies a position in space; it also has vision (its representations), and this vision determines how it "sees" and "understands" the world that enters as input.

The body answers "what the model is." Vision answers "how the model understands."

What this chapter sets out to do is to open up each of these two sides separately, and then see clearly the relationship between them — because the most central logical chain of the entire book is tied together at this very link.

## 2.1 Body: A Model Is a Point in Parameter Space

Let us begin from the simplest place.

Consider a model with only two parameters:

$$f(x) = w_1 x + w_2$$

What it does is exceedingly simple: take an input $x$, multiply it by $w_1$, add $w_2$, and output the result. But note — $(1, 0)$ is one model ($f(x)=x$), $(2,3)$ is another model ($f(x)=2x+3$), and $(-0.5, 7)$ is yet another. They give different outputs for the same input; they occupy different positions in space.

**Every parameter combination is a concrete, behaviorally determinate model.** However many parameters you have, that is the dimensionality of the space you inhabit. Two parameters — a two-dimensional plane. $N$ parameters — $N$-dimensional space. This is parameter space.

![Parameter space of linear regression](/figures/ch02_linear_regression_space.svg)

*Left: data space — the scattered points are noisy observations, and the red, orange, and green lines correspond respectively to the three models $(1, 0)$, $(2, 3)$, $(-0.5, 7)$. Right: the parameter space $(w_1, w_2)$ plane — every point is a complete model. Changing the position of a point changes the entire behavioral pattern of the model.*

Extending this to real neural networks, the principle is exactly the same. Take all the weight matrices, bias vectors, embedding tables, and attention matrices, flatten them, and line them up:

$$\theta = (\theta_1, \theta_2, \ldots, \theta_N)$$

This $\theta$ is the model's coordinate in $N$-dimensional parameter space. A small network for MNIST handwritten digit recognition has about a hundred thousand parameters; BERT-base has 110 million; GPT-3 has 175 billion. No matter how large the scale, the fundamental logic remains unchanged: **the body of a model is a point in parameter space.**

This body has several properties worth noting. First, it is continuous — a parameter can change from $0.01$ to $0.02$, and from $0.02$ to $0.0217$; learning is continuous motion, not jumping. Second, it has symmetry — swap two neurons of a hidden layer and simultaneously swap the corresponding incoming and outgoing weights, and the network's function remains completely unchanged, yet the parameter coordinates have undergone a massive shift. This means that different points in parameter space can correspond to the same function — the body's "posture" differs, but the "capability" is the same.

## 2.2 Training: The Body Moves through a Landscape

Parameter space itself is merely a container — an $N$-dimensional empty room. A container alone produces nothing at all.

What brings this space "alive" is the loss function. The loss function $L(\theta)$ assigns a numerical value to every position in parameter space: how poorly the model at that position performs. High loss, bad model; low loss, good model. And so, the originally flat, undifferentiated, featureless parameter space suddenly acquires rises and falls:

$$\theta \mapsto L(\theta)$$

This is no longer an empty room. This is a landscape.

This transformation is a perfect parallel to the introduction of the potential energy function $V(x)$ in physics. Before potential energy exists, space is homogeneous — an object placed anywhere is the same. But once there is gravitational potential, electric potential, elastic potential, space is no longer flat. An object in a region of high potential energy is "uncomfortable" and will naturally slide toward regions of lower potential. A ball on a hillside rolls down; water in a valley converges at the lowest point.

The loss function in deep learning plays exactly the role of potential energy. It turns a homogeneous parameter plain into an energy landscape of mountains, basins, cliffs, and rivers. **Parameters are position; loss is the elevation; the gradient is the direction of the slope; the optimizer is the manner of walking.** Training is not a model consciously "searching for answers" — training is a physical process: the model is pulled by the gradient, sliding downhill along the terrain, sliding into deeper and deeper, more and more stable low-loss regions.

Every gradient update:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

genuinely moves the model in parameter space. From the initial position $\theta_0$ to the final position $\theta_T$, the model traces out a trajectory in parameter space. This trajectory is the training process.

![Loss landscape: parameter space acquires rises and falls](/figures/ch02_loss_landscape_concept.svg)

*Every point $(\theta_1, \theta_2)$ in parameter space is assigned an elevation $L(\theta)$ by the loss function. Blue regions are low-loss areas (good models); red regions are high-loss areas (poor models). A training run begins from some random high-loss position and descends step by step along the terrain. Two different initial points may slide into entirely different valley floors.*

We can thus understand all the fundamental elements of deep learning as follows:

- **Initialization** is the model being placed at some random position in parameter space at birth.
- **Training** is its walking through space — step by step, gradually moving toward better places.
- **The optimizer** is its way of walking — large strides or small, straight ahead or with momentum.
- **Data** is the force that shapes this space — it determines which positions are "good" and which are "bad."
- **Generalization** is whether it has reached a stable and broad region, rather than getting stuck by chance in a narrow lucky spot.

## 2.3 Good Models Are Not Isolated Points, but Basins

Low-dimensional intuition can easily lead us into an error: thinking that a good model is a solitary, precise point in parameter space — you must stand exactly on that coordinate; the slightest deviation ruins everything.

But in deep learning, the situation is entirely different. Good solutions — models with very low loss and excellent generalization — are often not isolated points, but entire regions. Perhaps a broad basin, perhaps a connected valley, perhaps an entire connected region of low loss.

This implies two things. First, **good models are not unique.** Parameter space contains vast numbers of different positions that can all produce models with excellent performance. Starting from different random initializations, taking different training paths, one may arrive at different but equally good endpoints. Second, **there may be paths between two good models.** Two equally excellent models may be far apart in Euclidean distance in parameter space — yet there may exist a "low-loss corridor" between them, along which one can walk from one good model to the other without ever entering high-loss territory.

"Model merging" — taking a weighted average of the parameters of two fine-tuned models — sometimes produces a hybrid model that possesses the capabilities of both. This itself is indirect evidence for the existence of low-loss corridors: along the interpolation path between two good positions, the loss does not spike dramatically.

Flatness is also crucial. A sharp minimum — loss extremely low but performance collapses at the slightest perturbation — may generalize poorly. A flat minimum — loss slightly higher but surrounded by low-loss territory — may generalize better. **Generalization ability often depends more on how wide and how stable the basin the model occupies is, rather than on how deep the valley floor is.**

:::info

**Lyapunov Functions: Knowing a System Will Converge without Watching the Endpoint**

Here is a deeply counterintuitive question. To determine whether a system will ultimately stabilize, do we need to stare at it continuously until it actually comes to rest?

The Russian mathematician Lyapunov gave a beautiful answer in 1892: **No. You only need to find an "energy function."** For a dynamical system, if you can find a scalar function $V(x)$ satisfying three conditions: $V(x) > 0$ for all non-equilibrium points, $V(x^*) = 0$ at the equilibrium point, and $\dot{V}(x) < 0$ strictly decreasing along trajectories — then without having to watch the trajectory run its full course, you can assert that the system must converge.

Under an appropriate step size, the loss function $L(\theta)$ behaves exactly like a Lyapunov function: monotonically decreasing, bounded below, continuing to descend unless it halts at a critical point. **When we say "the loss is decreasing," we are essentially saying: we have found a Lyapunov function that is doing its job. The decrease of energy is the proof of convergence.**

This idea will recur throughout the latter half of this book — in Volume II on dynamical systems, and in Volume III on belief fixed points. To judge whether a reasoning process is stable, you do not necessarily need to read through its entire chain of thought; as long as you can define a reasonable "belief energy" and prove that it decreases monotonically at each reasoning step, you can assert that the system will converge long before it actually stops.

:::

![Good models occupy basins, not isolated points](/figures/ch02_connected_basins.svg)

*Left: sharp minimum — loss is extremely low, but the slightest perturbation of parameters causes performance to collapse catastrophically. Right: a broad, flat basin — the loss value may not be the absolute lowest, but a large surrounding region maintains good performance. The dashed "low-loss corridor" connects two different basins: you can walk smoothly from one good model to another without ever entering high-loss territory.*

## 2.4 Vision: When a Model "Looks" at an Image

Now let us turn the lens from parameter space to the interior of the model. The preceding sections discussed the model's body — where it is, how it moves, and what kind of landscape it walks into. But a body standing there — what does it see?

Consider a photograph of a cat. To a human, this is a cat — a furry animal with ears and whiskers. But to a model, the input is not a cat. The input is a matrix of numbers — for example, $28 \times 28 = 784$ pixel values, each between 0 and 255. The model does not "see" a cat; it merely receives a string of 784 numbers.

This string of numbers enters the first layer of the network. The neurons of the first layer are sensitive to simple local patterns — certain neurons activate at horizontal edges, certain at vertical edges, certain in response to particular contrasts of light and dark. And so, the output of the first layer is no longer a string of pixels, but a "map of edges" — where are the boundaries, where are the flat surfaces, where are the corners.

This edge map enters the second layer. The neurons of the second layer combine edge information to recognize more complex local structures — the contours of circles, the shapes of corners, repeating patterns of texture. The output becomes a "map of parts" — here might be an eye, there might be an ear, another region might be the texture of fur.

The third layer continues the combination. Eyes, ears, nose, whiskers — these parts are assembled together. By the time we reach just before the final layer, the interior of the network has formed a highly abstract representation: it is not a copy of any specific photograph of a cat, but rather the "cat-ness" extracted from all cats it has ever seen — a vector in a high-dimensional space, far from the representation vector of "dog," close to the representation vector of "tiger" but not overlapping with it.

This process — **from pixels to edges, from edges to parts, from parts to concepts** — is representation learning. The activation values of each layer's neurons constitute that layer's "representation." Every transformation an input undergoes within the network is a projection of a raw signal into a new representation space.

This is the model's vision. It does not passively receive the world; it actively translates the world into its own internal language.

![Layer-by-layer representation transformation: from pixels to concepts](/figures/ch02_layer_transformation.svg)

*An input image undergoes layer-by-layer transformations, with each layer producing representations at different levels of abstraction. Layer 1: edge detection (local directional features); Layer 2: part combinations (eye, ear, nose regions); Layer 3: object-level activation (holistic cat-shaped response); Output: classification probabilities ("cat" wins at 82%). The essence of representation is the gradual translation of raw pixels into increasingly abstract semantic coordinates.*

## 2.5 Representation Space: The Model's Map of the World

If parameter space answers "where is the model's body," then representation space answers "how does the model understand what it sees."

Representation space is the way the interior of the model encodes input data. A photograph of a cat, after being transformed layer by layer, ultimately becomes a vector in a hidden layer — this vector is the model's understanding of "this cat." A photograph of a dog also becomes a vector, and under normal circumstances, this vector will land in space near the cat's vector (both are pets, both are furry four-legged animals), yet also distinctly separated (different species).

This is the fundamental property of representation space: **similar inputs produce nearby representations.** The model does not need to be explicitly told that "cats and dogs are both animals" — if, in the training data, cats and dogs frequently appear in similar contexts, their representation vectors will naturally gravitate toward each other. The distances in representation space encode the model's understanding of the world's structure.

But what does "similar" mean? This is the core question of all representation learning. The model is not doing "genuine understanding" — what it is doing is: mapping raw inputs (pixels, text, sound) into a vector space, such that within this space, similarities useful for the task are amplified, and differences irrelevant to the task are compressed.

For instance, a model trained for cat-vs-dog classification will compress away differences in fur color (a black cat and a white cat are close in its representation space), but will amplify differences in ear shape (cat ears and dog ears are far apart in its representation space). A model trained for facial recognition does the opposite — it amplifies subtle differences between different individuals, while compressing the differences caused by different expressions, different angles, and different lighting for the same person.

**Representation space is a map, but this map is drawn by the model itself.** Different training tasks, different data distributions, different loss functions will sculpt utterly different representation spaces. The same model architecture, trained on ImageNet versus trained on medical images, will develop entirely different "worldviews."

## 2.6 The Manifold Hypothesis: Data Curls Up in Low-Dimensional Surfaces

Why is representation learning possible? Why can a model extract meaningful low-dimensional structure from a million-dimensional pixel space?

The answer points to a profound observation known as the **manifold hypothesis**: real-world data does not fill the high-dimensional space it resides in — it curls up on a low-dimensional manifold.

What is a "manifold"? Intuitively speaking, a manifold is something that locally looks like Euclidean space. The surface of the Earth is a two-dimensional manifold — locally it looks flat (standing on the ground, you cannot see the curvature), but globally it is a sphere. A curved line is a one-dimensional manifold; a twisted surface is a two-dimensional manifold. The key property of a manifold is: its intrinsic dimension is far lower than the dimension of the ambient space it sits in.

Now consider all possible photographs of cats. These photographs exist in pixel space — if each photo is a $256 \times 256$ RGB image, then they exist in a space of $256 \times 256 \times 3 = 196{,}608$ dimensions. But "all photographs of cats" are not scattered randomly throughout this two-hundred-thousand-dimensional space — they are densely curled up on a manifold of far lower dimension.

Why? Because cat photographs are not arbitrary combinations of pixels. Cats have eyes, ears, noses, fur — these structures impose extremely strong constraints. You cannot randomly generate a string of pixels and obtain a photograph of a cat. Cat photographs occupy only an extremely narrow, highly structured subregion of pixel space. The intrinsic dimension of this subregion is determined by the number of factors of variation — cat breed, angle, lighting, posture, age, fur color — likely only a few dozen or a few hundred dimensions, far smaller than two hundred thousand.

The same logic applies to all natural data: human faces, speech, text, protein structures. They all curl up on low-dimensional manifolds. **The essence of representation learning is to "unfold" these manifolds — to find the low-dimensional structure curled up in high-dimensional space, smooth it out, and map it into a well-structured representation space.**

In representation space, moving along the directions of the manifold should correspond to meaningful semantic changes. For example, in a face representation space, moving along one direction might gradually change the age of the face; moving along another direction might change the expression from happy to sad. These directions are the "coordinate axes" of the manifold — they capture the intrinsic dimensions of data variation.

![The manifold hypothesis: data curls up on low-dimensional manifolds](/figures/ch02_manifold_hypothesis.svg)

*Left: Swiss Roll — a two-dimensional manifold in three-dimensional space. The data points appear to be distributed within a three-dimensional volume, but in fact they occupy only a rolled-up two-dimensional surface (color varies smoothly along the manifold directions). Middle: the unfolded manifold — representation learning "irons out" the manifold so that the intrinsic coordinates along the manifold (rotation angle, height) become Euclidean coordinates. Right: a one-dimensional manifold embedded in two-dimensional space — even if there is only a curve, there is noise in the orthogonal direction around it (short gray lines); the goal of representation learning is to separate manifold directions from noise directions.*

![Nonlinear Manifold Projection](/figures/ch02_manifold_projection_tikz.svg)

*Nonlinear manifold projection from parameter space to representation space. The mesh surface shows how points in the parameter plane ($\theta_1,\theta_2$) are mapped through a nonlinear transformation $f_\theta$ into structured representations. Different parameter positions (red $\theta_a$, blue $\theta_b$, orange $\theta_c$) produce different representational geometries — changing the body's position changes the projection.*

## 2.7 Word Embeddings: When Semantics Becomes Geometry

The most intuitive, and also most striking, example of representation space comes from word embeddings in natural language processing.

In 2013, Mikolov et al. trained a simple language model and discovered an unexpected phenomenon: stable vector arithmetic relationships existed among the trained word vectors. The most famous example is:

$$\text{king} - \text{man} + \text{woman} \approx \text{queen}$$

This is not magic, nor was it designed. The model was never explicitly told that "the relationship between king and queen is analogous to the relationship between man and woman." Yet it automatically learned this structure in representation space.

How? The training of word embeddings is based on a simple idea: **the meaning of a word is determined by the words around it.** If "king" and "queen" frequently appear in similar contexts (royal, throne, crown, reign...), their vectors will be trained to be near each other. But "king" more often co-occurs with "man" and "he," while "queen" more often co-occurs with "woman" and "she," so a direction opens up between them — and this direction happens to correspond to gender.

Even more astonishing, "king - man + woman ≈ queen" shows that this direction not only exists, but is **translatable**. The "royalty" direction (from ordinary person to royal) and the "gender" direction (from male to female) form approximately orthogonal axes in the vector space. By moving along these axes, you can navigate in semantic space: starting from "king," subtract the "male" direction, add the "female" direction, and arrive near "queen."

This means that geometric structure in representation space carries semantic relationships. **Vector arithmetic becomes semantic reasoning.** If representation space is flat and well-structured, then "France - Paris + Tokyo" should be close to "Japan"; "walk - walked + run" should be close to "ran" (the direction of tense change is consistent across different verbs).

This is not merely a trick. It reveals a fundamental fact: **meaning can be encoded as geometry.** In a sufficiently good representation space, "understanding" a concept means placing it in the correct geometric relationships with other concepts. Distance encodes similarity, direction encodes type of relation, subspaces encode category boundaries.

![The geometry of word embeddings: king - man + woman ≈ queen](/figures/ch02_word_embedding_geometry.svg)

*Semantic arithmetic in word vector space. The blue arrow (man → king) = the royalty direction; the pink arrow (man → woman) = the gender direction. Translating king along the gender direction yields queen; translating woman along the royalty direction also yields queen. The equation king - man + woman ≈ queen holds because "gender" and "royalty" form approximately orthogonal semantic coordinate axes in representation space. Vector arithmetic becomes semantic reasoning.*

## 2.8 The Sharpening of Vision: Before and After Training

Now that we understand what representation space is, the most natural follow-up question is: **how does training change it?**

A randomly initialized model and a trained model, faced with the same photograph of a cat, have entirely different internal representations.

At random initialization, the weights of all layers are random values. After a cat photograph enters the network, the activations of every layer are also random — no structure, no clustering, no meaningful distances. The representation vectors of all categories of images are mixed together in space, like a pot of stirred soup. The cat's vector and the dog's vector may be close or may be far — but either way, there is no regularity to it.

Training changes everything. As the loss function pushes the model's body to move through parameter space, the weights of each layer are gradually adjusted. The cumulative effect of this adjustment manifests in representation space as the emergence of structure:

The representation vectors of cat images begin to cluster together; the representation vectors of dog images also begin to cluster together — but the cat cluster and the dog cluster separate from each other. Within cats, different breeds (Siamese, orange tabby, black cat) form subclusters; within dogs, different breeds also aggregate. A broad chasm opens up between "animals" and "vehicles." The distance between "cat" and "tiger" is far smaller than the distance between "cat" and "car," but somewhat larger than the distance between "cat" and "dog."

This structure is not explicitly programmed. It is the natural product of training. The model's body moves through parameter space; each gradient update subtly adjusts the weight matrices of all layers. The cumulative effect of these adjustments is the structuration of representation space — the model learns to project the world into a meaningful internal coordinate system.

**The movement of the body is the sharpening of vision.** Every step in parameter space leaves a trace in representation space: certain directions are stretched (differences important for the task are amplified), certain directions are compressed (differences irrelevant to the task are ignored), certain regions are curved (nonlinear transformations make linearly inseparable data separable).

![The emergence of representation space: before vs. after training](/figures/ch02_representation_emergence.svg)

*Left: before training — the representation vectors of five categories of data (cat, dog, bird, car, fish) are randomly intermingled in space, with no structure, no clustering, no meaningful distance relations whatsoever. Right: after training — samples of the same class automatically aggregate, and clear boundaries form between different classes. The emergence of structure is not explicitly programmed, but is the natural result of the body's movement in parameter space: the body's position changes, and vision sharpens accordingly.*

## 2.9 Body and Vision: The Central Logical Chain of the Book

Now we can look at parameter space and representation space together. The relationship between them is the pivot of the entire book's theoretical main line.

**Parameter space answers "what the model is." Representation space answers "how the model understands."**

The relationship between them is as follows: every position in parameter space corresponds to a particular structure of representation space. Changing position in parameter space (training) changes the structure of representation space. The model's body standing in different places leads to different vision — its coordinate system for viewing the world, what differences it amplifies, what differences it compresses, along what dimensions it establishes similarity, all change accordingly.

And so, the following chain is the most central logic of the entire book:

- **Parameter space is the body** — it records what the model is; it is the "subject" of training.
- **The loss landscape is the slope** — it tells the body which way to go.
- **Gradient descent is the walking** — the body moves through the slope.
- **Representation space is vision** — it shows how, after the body moves, the model re-understands the world.
- **Capability does not appear out of nowhere** — when the body's position changes, the structure of vision changes along with it. The structure of vision is the geometric foundation of capability.

This logical chain means that we cannot understand any single link in isolation. You cannot simply say "the loss went down" — you must ask: when the loss went down, how far did the body travel? What kind of terrain did it stop on? What does the vision corresponding to this position look like? Is the structure of vision stable? Does it generalize? Is it interpretable?

Conversely, you cannot simply say "the model learned" — you must ask: what change in the body does "learned" mean? What trajectory did this change trace out in parameter space? What terrain did the trajectory pass through? Why is this endpoint better than the starting point — better how? Because it stands in a deeper valley floor, or because it stands in a wider basin?

This is the power of the geometric perspective: it transforms "learning" from a black-box term into a structural process that can be traced, measured, and analyzed in space.

:::info

**Mr. Pallas's Cat's Position**

Some say that parameter space and representation space are two different things — one is "the internal structure of the model," the other is "the world the model sees." They are trapped by terminology.

Body and vision are not two things. They are two projections of the same thing. You move the body in parameter space, and the structure of representation space follows. You see clusters, directions, boundaries in representation space — those are all geometric shadows of the body's position.

Many people ask: "Does this model truly understand language?" And then they stare back and forth at the output text. You are looking in the wrong place. Text is the shadow of a shadow. Genuine "understanding" — if this word has any meaning — happens in representation space. It is a set of geometric relations among vectors. You do not need to ask whether the model "understands" — you need only look at whether, in its representation space, the vectors of cat and dog have each clustered into their own groups, and whether there is an edge between king and queen that is parallel to the edge from man to woman.

Geometry does not ask about "understanding." Geometry looks only at position.

:::

---

## 2.10 Chapter Summary

This chapter has established the most crucial pairing of the entire book: **body and vision.**

The body of a model is its parameters — an $N$-dimensional coordinate composed of weights, biases, and embedding tables. Every point in parameter space is a behaviorally determinate, complete model. Training is not the pouring in of knowledge, but the movement of this body: step by step through the loss landscape, from a random high place to a stable low place. Good models are often not isolated minima, but broad basins — generalization depends on how stably one stands, not how deeply.

The vision of a model is its representations — the internal vectors into which input data is encoded through layer-by-layer transformations. Representation space is the model's map of the world: similar inputs cluster together, different inputs separate, relations become directions, analogies become vector arithmetic. The manifold hypothesis tells us that natural data curls up on low-dimensional manifolds; the essence of representation learning is to unfold these manifolds. The astonishing geometric properties of word embeddings (king - man + woman ≈ queen) prove: meaning can be encoded as geometry.

The body determines vision. Every movement in parameter space reshapes the structure of representation space. Every step of training is not just the loss decreasing — it is the model's body moving, and its vision sharpening in turn.

In the next chapter, we shall descend into the very ground beneath the body's feet: **the loss landscape and gradient motion.** How does the loss function carve mountains and basins in parameter space? How does the gradient indicate direction? How does the Euler step turn direction into displacement? Why are some minima flat and others sharp? The geometry of this terrain will determine how far the model can walk, and where it comes to rest.

---

## Unresolved Questions

1. If different points in parameter space can correspond to the same function (due to neuron permutation symmetry), then is "the position of the model's body" an objective fact, or a convention dependent on how we label the neurons?

2. The loss function is human-defined (cross-entropy, mean squared error, contrastive loss...). Do different loss functions draw entirely different landscapes in parameter space? Or are they different projections of a shared "true terrain"?

3. If good models tend to occupy broad basins rather than sharp minima, what is the causal relationship between flatness and generalization ability? Does flatness cause generalization, does generalization cause flatness, or are both the result of some more fundamental cause?

4. Is "distance" in representation space — two inputs being judged "similar" by the model — objective? Or does every definition of distance (Euclidean, cosine, geodesic) implicitly embody a stance on "what counts as similar"?

5. The manifold hypothesis says natural data curls up on low-dimensional manifolds. Is the dimensionality and topological structure of this manifold determined by the data itself, or is it co-sculpted by the model architecture and the training objective? In representation space, are the manifold of a cat and the manifold of an airplane connected?

6. Vector arithmetic in word embeddings (king - man + woman ≈ queen) depends on the approximately linear structure of the representation space. Is this linearity a natural outcome of training, or a constraint imposed by some inductive bias (such as the linear layers of the model)? In deeper representation spaces with stronger nonlinearity, do analogous geometric operations still hold?

7. Before training, the random representation space has no structure whatsoever; after training, the representation space exhibits clear geometric regularities. Is the emergence of this structure gradual or abrupt? During training, does the representation space undergo a "phase transition" — suddenly switching from chaos to order at some critical point?

8. The same model architecture, trained on different data, develops entirely different representation spaces. Does this mean that data "sculpts" the model's vision? If so, are biases in the training data (race, gender, culture) directly etched into the geometric structure of representation space?

9. "Model merging" suggests the existence of low-loss corridors in parameter space. What relationship do these corridors have with the structure of representation space? When walking from one model to another along a low-loss corridor, does representation space deform continuously, or does it undergo abrupt reorganization?

10. If we regard the patterns of neural activity in the human brain as a kind of "representation space" as well, what are the similarities and differences, in geometric properties, between representation learning in biological neural systems and in deep networks?

11. Are "directions" in representation space (such as the gender direction in word embeddings) genuine geometric entities, or projections imposed by us in post-hoc interpretation? Can we define, in representation space, "natural coordinate axes" that do not depend on human interpretation?

12. "The body determines vision" — but what about the reverse: can we infer the body from the vision? That is, given the structure of a model's representation space, can we deduce its position in parameter space, or even infer its training data and training process?

---

**The core question left by this chapter is:**

**Is the relationship between parameter space and representation space — between body and vision — a one-way "body determines vision," or a two-way mutual shaping?**

:::info

**The body moves through parameter space; vision sharpens in representation space.** But why does the body walk in this direction rather than that direction? Because the terrain underfoot has a slope. In the next chapter, we shall focus on that terrain itself — the geometry of the loss function: how the gradient points the way, how the Euler step takes a stride, and how valleys, saddle points, and minima shape all the possibilities of learning. Where the body stands determines vision — but where the body can walk to is determined by the terrain.

:::
