import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The UI is static vanilla HTML/JS under public/ (no React pages); Next
  // doesn't map "/" to public/index.html on its own, so rewrite it.
  async rewrites() {
    return [{ source: "/", destination: "/index.html" }];
  },
};

export default nextConfig;
