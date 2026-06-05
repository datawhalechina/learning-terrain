---
layout: home

hero:
  name: "The Terrain of Learning"
  text: "Geometry of Reasoning and Dynamics of Intelligence"
  tagline: From space to terrain, from terrain to motion, from motion to convergence, from convergence to reasoning
  image:
    src: /cover.png
    alt: The Terrain of Learning
  actions:
    - theme: brand
      text: Start Reading
      link: /en/01_the_terrain_of_learning/ch01
    - theme: alt
      text: "Vol.I: The Terrain of Learning"
      link: /en/01_the_terrain_of_learning/ch01
    - theme: alt
      text: "Vol.II: Walking and Belief"
      link: /en/02_dynamics_of_intelligence/ch04
    - theme: alt
      text: "Vol.III: The Geometry of Reasoning"
      link: /en/03_geometry_of_reasoning/ch07
    - theme: alt
      text: "Vol.IV: Algorithmic Landscapes"
      link: /en/04_algorithmic_landscapes/ch10

features:
  - title: "Vol.I: The Terrain of Learning (3 chapters)"
    details: Building geometric intuition. Parameter space, representation space, loss terrain, gradient fields, Euler steps — where the model's body lives, how its vision forms, how the terrain guides motion.
  - title: "Vol.II: Walking and Belief (3 chapters)"
    details: Optimizers as ways of walking, non-Euclidean distance, dynamical systems and fixed points. ResNet = explicit Euler, GPT autoregression = Euler in hidden state space, DEQ = fixed point iteration.
  - title: "Vol.III: The Geometry of Reasoning (3 chapters)"
    details: Chain of thought is not reasoning itself, but the visible projection of the reasoning trajectory. Attractors, verifiers, RLHF — how the reasoning field is reshaped by training.
  - title: "Vol.IV: Algorithmic Landscapes (3 chapters)"
    details: Classical algorithms geometrized. Linear regression, PCA, SVM, Attention, LoRA, Diffusion — one figure, one formula, one experiment per chapter.
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://www.github.com/lizixi-0x2F.png',
    name: 'Li Zixi (Mr. Pallas\'s Cat)',
    title: 'Author',
    links: [
      { icon: 'github', link: 'https://github.com/lizixi-0x2F' },
    ]
  },
]
</script>

<h2 align="center">Author</h2>
<VPTeamMembers size="small" :members />
