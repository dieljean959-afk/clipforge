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
            '@radix-ui/react-slot',
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
  // Keep the dep-optimizer cache OUT of node_modules.
  //
  // The default cacheDir is `node_modules/.vite`, and the platform recreated
  // that tree on each sandbox restart (24 restarts on 2026-10-04 alone). A cold
  // cacheDir means a full esbuild re-optimization on the FIRST page load, and
  // that work is synchronous with the module-graph request. Measured from the
  // toolbox proxy log across a real boot: `GET /src/main.tsx` blocked for
  // 16.262s before returning 200, and `/src/index.css` took 3.58s. That is
  // the preview panel sitting at "Project connected, booting up preview… 97%"
  // — the origin is answering, it simply cannot serve the entry module until
  // esbuild finishes, so the bar has nothing to advance against.
  //
  // A cacheDir at the project root is part of the synced workspace, so the
  // optimizer output survives a sandbox restart and later boots start warm.
  cacheDir: '.vite-cache',

  // Optimize dependencies
  optimizeDeps: {
    // Only scan the app entry HTML; avoids crawling unrelated *.html files
    // if a legacy snapshot accidentally contains leaked package folders.
    entries: ['index.html'],
    // Every bare specifier the app actually imports.
    //
    // This list used to hold only 6 entries while `src/` imports roughly 40.
    // Vite therefore discovered the remainder LAZILY, mid-request, which forces
    // an extra optimization round while the page is already loading — and with
    // `hmr: false` the browser never receives the reload notice, so the boot
    // stalls rather than recovering. Declaring the full set up front keeps the
    // optimizer's work deterministic and confined to server startup.
    include: [
      'react',
      'react/jsx-runtime',
      'react-dom',
      'react-dom/client',
      'react-router',
      'framer-motion',
      'lucide-react',
      'sonner',
      'next-themes',
      'cmdk',
      'class-variance-authority',
      'clsx',
      'tailwind-merge',
      'vaul',
      'embla-carousel-react',
      'input-otp',
      'react-day-picker',
      'react-hook-form',
      'react-resizable-panels',
      'recharts',
      '@vly-ai/integrations',
      '@radix-ui/react-accordion',
      '@radix-ui/react-alert-dialog',
      '@radix-ui/react-aspect-ratio',
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
      '@radix-ui/react-slot',
      '@radix-ui/react-switch',
      '@radix-ui/react-tabs',
      '@radix-ui/react-toggle',
      '@radix-ui/react-toggle-group',
      '@radix-ui/react-tooltip',
    ],
    // Do not hold the optimizer back until the entry crawl finishes. With the
    // crawl held, any module requested during startup queues behind it, which
    // is what pushed the entry-module response into double-digit seconds.
    holdUntilCrawlEnd: false,
    // Discovery stays ON. With `include` complete above it should find nothing
    // new, but if it ever does, a re-optimize is recoverable here (the server
    // restarts) rather than leaving a half-applied dep graph in the browser.
    noDiscovery: false,
  },
  // Performance hints
  server: {
    // Bind DUAL-STACK, not IPv4-only.
    //
    // This used to be "0.0.0.0", which listens on IPv4 only. The platform's
    // preview proxy does not dial 127.0.0.1 — it dials the IPv6 loopback,
    // `[::1]:5173`, and on an IPv4-only socket that dial is REFUSED. The
    // sandbox daemon log recorded 4,719 `502 bad gateway` responses, virtually
    // all of them the same error:
    //
    //   toolbox proxy upstream error path=/ error="dial tcp [::1]:5173:
    //   connect: connection refused"
    //
    // So every preview boot raced the proxy against a dial it could never
    // satisfy, and the preview sat at "Project connected, booting up preview"
    // (97%) retrying a connection that was guaranteed to fail until it fell
    // back to IPv4. That is the wait the user was seeing.
    //
    // "::" with net.ipv6.bindv6only=0 (confirmed 0 on this host) is a
    // dual-stack socket: it accepts IPv6 loopback AND the IPv4 connections
    // that already worked, so the proxy's preferred dial succeeds on the first
    // try instead of after a failed attempt. IPv4 reachability is preserved.
    host: "::",
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
        // The dep-optimizer cache now lives in the project root. It churns
        // hundreds of files while pre-bundling; without this, each one is a
        // watcher event in the middle of a page load.
        ".vite-cache/**",
      ],
    },
    proxy: {
      // local pipeline API + rendered media
      "/api": "http://127.0.0.1:8000",
      "/clips": "http://127.0.0.1:8000",
      // Public share links must reach FastAPI on the SAME origin they were
      // copied from. Without this, Vite's SPA fallback served index.html for
      // /s/<token> and the recipient/social crawler never saw the clip. Use
      // "/s/" (not "/s") so Vite's module path /src is never captured.
      "/s/": "http://127.0.0.1:8000",
    },
  },
});
