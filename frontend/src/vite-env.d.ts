/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API origin. Empty in development: Vite proxies /api. */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
