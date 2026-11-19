import path from "node:path";
import fs from "node:fs";
import { defineConfig, mergeConfig, type Plugin } from "vite";
import base from "./vite.config";

/**
 * Production build entry.
 *
 * The dev `index.html` is a boot loader which appends `/src/main.tsx` at
 * runtime, so Vite cannot see the real app entry and emits only the loader.
 * This config uses a normal HTML entry instead, then renames its output to
 * `index.html` so `webui/` is a drop-in static app.
 */
const renameIndex: Plugin = {
  name: "clipforge-rename-prod-index",
  closeBundle() {
    const out = path.resolve(__dirname, "dist/index.html");
    const built = path.resolve(__dirname, "dist/index.prod.html");
    if (fs.existsSync(built)) fs.renameSync(built, out);
  },
};

export default mergeConfig(
  base,
  defineConfig({
    plugins: [renameIndex],
    build: {
      outDir: "dist",
      emptyOutDir: true,
      rollupOptions: {
        input: path.resolve(__dirname, "index.prod.html"),
      },
    },
  }),
);
