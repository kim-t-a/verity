import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

/**
 * The browser talks to the Vite dev server, and the dev server forwards /api
 * to FastAPI. That keeps the frontend origin-agnostic (no CORS in the normal
 * path) and means the API base can stay empty.
 *
 * `VITE_PROXY_TARGET` is read at server start, not baked into the bundle: the
 * container sets it to http://backend:8000, a laptop leaves it at localhost.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://localhost:8000";
  const proxy = {
    "/api": {
      target,
      changeOrigin: true,
    },
  };

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: { proxy },
    preview: { proxy },
  };
});
