import { vlyPlugin } from "@vly-ai/integrations";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";
import { clipforgeBackend } from "./vite-plugin-clipforge-backend";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), vlyPlugin(), tailwindcss(), clipforgeBackend()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    // Force a single copy of React across all packages (including vlyPlugin).
    // Without this, @vly-ai/integrations can resolve its own React copy, which
    // triggers "Invalid hook call" errors at runtime.
    dedupe: ["react", "react/jsx-runtime", "react-dom", "react-dom/client"],
  },
  build: {
    // Enable source maps for better debugging (disable in production if needed)
    sourcemap: false,
    // Optimize chunk splitting
    rollupOptions: {
      output: {
        // Manual chunk splitting for better caching and lazy loading
        manualChunks: {
          // Vendor chunks for large libraries
          'react-vendor': ['react', 'react-dom', 'react-router'],
          // Large UI library chunks
          'radix-ui': [
            '@radix-ui/react-accordion',
            '@radix-ui/react-alert-dialog',
            '@radix-ui/react-avatar',
            '@radix-ui/react-checkbox',
            '@radix-ui/react-collapsible',
            '@radix-ui/react-context-menu',
            '@radix-ui/react-dialog',
            '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-hover-card',
            '@radix-ui/react-label',
            '@radix-ui/react-menubar',
            '@radix-ui/react-navigation-menu',
            '@radix-ui/react-popover',
            '@radix-ui/react-progress',
            '@radix-ui/react-radio-group',
            '@radix-ui/react-scroll-area',
            '@radix-ui/react-select',
            '@radix-ui/react-separator',
            '@radix-ui/react-slider',
            '@radix-ui/react-switch',
            '@radix-ui/react-tabs',
            '@radix-ui/react-toggle',
            '@radix-ui/react-toggle-group',
            '@radix-ui/react-tooltip',
          ],
          // Heavy optional libraries - separate chunks for better lazy loading
          'framer-motion': ['framer-motion'],
          'charts': ['recharts'],
          'forms': ['react-hook-form', '@hookform/resolvers', 'zod'],
        },
        // Optimize chunk size
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
    },
    // Increase chunk size warning limit for better chunking
    chunkSizeWarningLimit: 1000,
    // Target modern browsers for better optimization
    target: 'esnext',
    // Minify options - using esbuild (faster than terser)
    minify: 'esbuild',
  },
  // Optimize dependencies
  optimizeDeps: {
    // Only scan the app entry HTML; avoids crawling unrelated *.html files
    // if a legacy snapshot accidentally contains leaked package folders.
    entries: ['index.html'],
    include: [
      'react',
      'react/jsx-runtime',
      'react-dom',
      'react-dom/client',
      'react-router',
      'framer-motion',
    ],
  },
  // Performance hints
  server: {
    // Bind to all interfaces so the preview proxy (and WebContainer's
    // server-ready event) can reach the dev server from outside the sandbox.
    host: "0.0.0.0",
    port: 5173,
    // The preview is served from a generated *.freebuff.dev domain, so the
    // Host header never matches localhost. Accept it instead of 403-ing.
    allowedHosts: true,
    // Freebuff manages the web process; keep HMR disabled for this project.
    hmr: false,
    // The pipeline writes uploads, sources, transcripts, rendered clips and the
    // job store as it runs. Those writes used to land inside the watched project
    // root, so every frame ffmpeg produced was a file-system event in the
    // watcher. Ignore them outright — the dev server has no business reacting
    // to gigabytes of media, and the media now lives outside this tree anyway
    // (see CLIPFORGE_STORAGE_DIR in app/config.py); the symlinks are kept out
    // of the crawl so a stale path can never drag them back in.
    watch: {
      ignored: [
        "**/downloads/**",
        "**/outputs/**",
        "**/data/**",
        "**/.reliability-backup/**",
        "**/.rel2-backup/**",
        "**/*.mp4",
        "**/*.mp3",
        "**/*.wav",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.png",
        "**/*.srt",
        "**/*.vtt",
        "**/*.part",
      ],
    },
    proxy: {
      // local pipeline API + rendered media
      "/api": "http://127.0.0.1:8000",
      "/clips": "http://127.0.0.1:8000",
    },
  },
});
