import { defineConfig } from 'vitepress'
import container from 'markdown-it-container'

const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/learning-terrain/'

const mdConfig = (md: any) => {
  const defaultRender = md.renderer.rules.html_inline || function(tokens: any, idx: any, options: any, env: any, self: any) {
    return self.renderToken(tokens, idx, options)
  }
  md.renderer.rules.html_inline = function(tokens: any, idx: any, options: any, env: any, self: any) {
    const token = tokens[idx]
    if (token.content.startsWith('<') && token.content.endsWith('>')) {
      return token.content.replace(/</g, '&lt;').replace(/>/g, '&gt;')
    }
    return defaultRender(tokens, idx, options, env, self)
  }
  md.use(container, 'info', {
    validate: (p: string) => p.trim() === 'info',
    render: (tokens: any[], idx: number) => tokens[idx].nesting === 1 ? '<div class="info custom-block">\n' : '</div>\n'
  })
  md.use(container, 'detail', {
    validate: (p: string) => p.trim() === 'detail',
    render: (tokens: any[], idx: number) => tokens[idx].nesting === 1 ? '<div class="detail custom-block">\n' : '</div>\n'
  })
  md.use(container, 'details', {
    validate: (p: string) => !!p.trim().match(/^details(::|\s|$)/),
    render: (tokens: any[], idx: number) => {
      if (tokens[idx].nesting === 1) {
        const title = tokens[idx].info.trim().replace(/^details\s*:?\s*/, '')
        return title
          ? `<div class="details custom-block"><p class="details-summary">${title}</p>\n`
          : '<div class="details custom-block">\n'
      }
      return '</div>\n'
    }
  })
}

export default defineConfig({
  base: baseConfig,
  appearance: false,

  locales: {
    root: {
      label: '中文',
      lang: 'zh-CN',
      title: '学习的地形',
      description: '推理几何与智能动力学导论',
      themeConfig: {
        logo: '/datawhale-logo.png',
        nav: [
          { text: '序言', link: '/preface' },
          { text: '卷一：学习的地形', link: '/01_the_terrain_of_learning/ch01' },
          { text: '卷二：行走与信念', link: '/02_dynamics_of_intelligence/ch04' },
          { text: '卷三：推理的几何', link: '/03_geometry_of_reasoning/ch07' },
          { text: '卷四：算法的地貌学', link: '/04_algorithmic_landscapes/ch10' },
        ],
        search: {
          provider: 'local',
          shortcut: {
            search: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' },
            open: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' }
          }
        },
        sidebar: {
          '/01_the_terrain_of_learning/': [
            {
              text: '卷一：学习的地形',
              items: [
                { text: 'ch1 为什么学习需要几何', link: '/01_the_terrain_of_learning/ch01' },
                { text: 'ch2 身体与视野：参数空间与表示空间', link: '/01_the_terrain_of_learning/ch02' },
                { text: 'ch3 损失地形与梯度运动', link: '/01_the_terrain_of_learning/ch03' },
              ]
            }
          ],
          '/02_dynamics_of_intelligence/': [
            {
              text: '卷二：行走与信念',
              items: [
                { text: 'ch4 行走的方式：优化器与正则化', link: '/02_dynamics_of_intelligence/ch04' },
                { text: 'ch5 非欧距离：Bregman与KL', link: '/02_dynamics_of_intelligence/ch05' },
                { text: 'ch6 不动点：从ResNet到GPT到DEQ', link: '/02_dynamics_of_intelligence/ch06' },
              ]
            }
          ],
          '/03_geometry_of_reasoning/': [
            {
              text: '卷三：推理的几何',
              items: [
                { text: 'ch7 思维链：推理轨迹的投影', link: '/03_geometry_of_reasoning/ch07' },
                { text: 'ch8 推理场：吸引子与验证器', link: '/03_geometry_of_reasoning/ch08' },
                { text: 'ch9 长推理与地貌重塑', link: '/03_geometry_of_reasoning/ch09' },
              ]
            }
          ],
          '/04_algorithmic_landscapes/': [
            {
              text: '卷四：算法的地貌学',
              items: [
                { text: 'ch10 经典算法的几何', link: '/04_algorithmic_landscapes/ch10' },
                { text: 'ch11 深度架构的几何', link: '/04_algorithmic_landscapes/ch11' },
                { text: 'ch12 扩散与收敛', link: '/04_algorithmic_landscapes/ch12' },
              ]
            }
          ],
          '/': [
            {
              items: [
                { text: '卷一：学习的地形 →', link: '/01_the_terrain_of_learning/ch01' },
                { text: '卷二：行走与信念 →', link: '/02_dynamics_of_intelligence/ch04' },
                { text: '卷三：推理的几何 →', link: '/03_geometry_of_reasoning/ch07' },
                { text: '卷四：算法的地貌学 →', link: '/04_algorithmic_landscapes/ch10' },
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/lizixi-0x2F/learning-terrain/' }
        ],
        editLink: {
          pattern: 'https://github.com/lizixi-0x2F/learning-terrain/blob/main/docs/:path'
        },
        footer: {
          message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
          copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'The Terrain of Learning',
      description: 'An Introduction to the Geometry of Reasoning and the Dynamics of Intelligence',
      themeConfig: {
        logo: '/datawhale-logo.png',
        nav: [
          { text: 'Preface', link: '/en/preface' },
          { text: 'Vol.I: The Terrain of Learning', link: '/en/01_the_terrain_of_learning/ch01' },
          { text: 'Vol.II: Walking and Belief', link: '/en/02_dynamics_of_intelligence/ch04' },
          { text: 'Vol.III: The Geometry of Reasoning', link: '/en/03_geometry_of_reasoning/ch07' },
          { text: 'Vol.IV: Algorithmic Landscapes', link: '/en/04_algorithmic_landscapes/ch10' },
        ],
        search: {
          provider: 'local',
          shortcut: {
            search: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' },
            open: { macos: 'Cmd+K', windows: 'Ctrl+K', linux: 'Ctrl+K' }
          }
        },
        sidebar: {
          '/en/01_the_terrain_of_learning/': [
            {
              text: 'Vol.I: The Terrain of Learning',
              items: [
                { text: 'ch1 Why Learning Needs Geometry', link: '/en/01_the_terrain_of_learning/ch01' },
                { text: 'ch2 Body and Vision: Parameter Space and Representation Space', link: '/en/01_the_terrain_of_learning/ch02' },
                { text: 'ch3 Loss Terrain and Gradient Motion', link: '/en/01_the_terrain_of_learning/ch03' },
              ]
            }
          ],
          '/en/02_dynamics_of_intelligence/': [
            {
              text: 'Vol.II: Walking and Belief',
              items: [
                { text: 'ch4 Ways of Walking: Optimizers and Regularization', link: '/en/02_dynamics_of_intelligence/ch04' },
                { text: 'ch5 The Non-Euclidean World: Bregman and KL', link: '/en/02_dynamics_of_intelligence/ch05' },
                { text: 'ch6 Fixed Points: From ResNet to GPT to DEQ', link: '/en/02_dynamics_of_intelligence/ch06' },
              ]
            }
          ],
          '/en/03_geometry_of_reasoning/': [
            {
              text: 'Vol.III: The Geometry of Reasoning',
              items: [
                { text: 'ch7 Chain of Thought: Projection of the Reasoning Trajectory', link: '/en/03_geometry_of_reasoning/ch07' },
                { text: 'ch8 Reasoning Fields: Attractors and Verifiers', link: '/en/03_geometry_of_reasoning/ch08' },
                { text: 'ch9 Long Reasoning and Landscape Reshaping', link: '/en/03_geometry_of_reasoning/ch09' },
              ]
            }
          ],
          '/en/04_algorithmic_landscapes/': [
            {
              text: 'Vol.IV: Algorithmic Landscapes',
              items: [
                { text: 'ch10 Geometry of Classical Algorithms', link: '/en/04_algorithmic_landscapes/ch10' },
                { text: 'ch11 Geometry of Deep Architectures', link: '/en/04_algorithmic_landscapes/ch11' },
                { text: 'ch12 Diffusion and Convergence', link: '/en/04_algorithmic_landscapes/ch12' },
              ]
            }
          ],
          '/en/': [
            {
              items: [
                { text: 'Vol.I: The Terrain of Learning →', link: '/en/01_the_terrain_of_learning/ch01' },
                { text: 'Vol.II: Walking and Belief →', link: '/en/02_dynamics_of_intelligence/ch04' },
                { text: 'Vol.III: The Geometry of Reasoning →', link: '/en/03_geometry_of_reasoning/ch07' },
                { text: 'Vol.IV: Algorithmic Landscapes →', link: '/en/04_algorithmic_landscapes/ch10' },
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/lizixi-0x2F/learning-terrain/' }
        ],
        editLink: {
          pattern: 'https://github.com/lizixi-0x2F/learning-terrain/blob/main/docs/:path'
        },
        footer: {
          copyright: 'Licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">CC BY-NC-SA 4.0</a>'
        }
      }
    },
  },

  markdown: {
    math: true,
    vue: { enabled: false },
    config: mdConfig
  }
})
