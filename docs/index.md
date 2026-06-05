---
layout: home

hero:
  name: "学习的地形"
  text: "推理几何与智能动力学导论"
  tagline: 从空间到地形，从地形到运动，从运动到收敛，从收敛到推理
  image:
    src: /cover.png
    alt: 学习的地形
  actions:
    - theme: brand
      text: 开始阅读
      link: /01_the_terrain_of_learning/ch01
    - theme: alt
      text: 卷一：学习的地形
      link: /01_the_terrain_of_learning/ch01
    - theme: alt
      text: 卷二：行走与信念
      link: /02_dynamics_of_intelligence/ch04
    - theme: alt
      text: 卷三：推理的几何
      link: /03_geometry_of_reasoning/ch08
    - theme: alt
      text: 卷四：算法的地貌学
      link: /04_algorithmic_landscapes/ch11

features:
  - title: 卷一：学习的地形（3章）
    details: 建立几何直觉。参数空间、表示空间、损失地形、梯度场、欧拉步——模型的身体在哪里，视野如何形成，地形如何引导运动。
  - title: 卷二：行走与信念（4章）
    details: 优化器作为行走方式，非欧距离，推理轨迹，动力系统与不动点。ResNet=显式欧拉，GPT自回归=隐状态欧拉迭代，DEQ=不动点。
  - title: 卷三：推理的几何（3章）
    details: 思维链不是推理本身，而是推理轨迹的可见投影。吸引子、验证器、RLHF——推理场如何被训练重塑。
  - title: 卷四：算法的地貌学（3章）
    details: 经典算法几何化。线性回归、PCA、SVM、Attention、LoRA、Diffusion——每章一图一公式一实验。
---

<script setup>
import { VPTeamMembers } from 'vitepress/theme'

const members = [
  {
    avatar: 'https://www.github.com/lizixi-0x2F.png',
    name: '李籽溪（兔狲）',
    title: '作者',
    links: [
      { icon: 'github', link: 'https://github.com/lizixi-0x2F' },
    ]
  },
]
</script>

<h2 align="center">Team</h2>
<VPTeamMembers size="small" :members />
