import { defineConfig } from "vite";
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8097",
      "/metabolic-api": {
        target: "http://127.0.0.1:8098",
        rewrite: (path) => path.replace(/^\/metabolic-api/, "/api"),
      },
    },
  },
  build: {
    sourcemap: false,
    rollupOptions: {
      preserveEntrySignatures: 'strict',
      input: {index: 'index.html', 'product-wearable': 'src/platform/productWearable.tsx'},
      output: {entryFileNames: entry => entry.name === 'product-wearable' ? 'assets/product-wearable.js' : 'assets/[name]-[hash].js'},
    },
  },
});
