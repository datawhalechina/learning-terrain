import DefaultTheme from 'vitepress/theme'
import './swiss-minimal.css'
import './custom.css'
import type { Theme } from 'vitepress'
import 'viewerjs/dist/viewer.min.css';
import imageViewer from 'vitepress-plugin-image-viewer';
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue';
import { useRoute } from 'vitepress';

export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        app.component('vImageViewer', vImageViewer);
    },
    setup() {
        const route = useRoute();
        imageViewer(route);
    }
} satisfies Theme
