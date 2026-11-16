import type { NextConfig } from "next";

// Static export: `next build` writes plain files to out/. There is no server;
// Python runs in the visitor's browser (public/pyodide-worker.js).
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
