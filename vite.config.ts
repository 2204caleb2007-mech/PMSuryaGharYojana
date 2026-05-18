import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from "path"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    // Proxy API calls to the FastAPI backend so `/api/*` routes work during dev
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/samples': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/sample': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
