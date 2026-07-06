import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Basic-auth gate, active only when SMARTROOM_PASS is set (so local dev stays
// open, but a public tunnel is protected). The browser caches the credential and
// sends it on every same-origin request — including the <img> stream proxy — so
// one prompt covers the whole app.
export function middleware(req: NextRequest) {
  const pass = process.env.SMARTROOM_PASS;
  if (!pass) return NextResponse.next();

  const header = req.headers.get("authorization");
  if (header?.startsWith("Basic ")) {
    try {
      const [user, pw] = atob(header.slice(6)).split(":");
      const userOk = !process.env.SMARTROOM_USER || user === process.env.SMARTROOM_USER;
      if (userOk && pw === pass) return NextResponse.next();
    } catch {
      /* malformed header */
    }
  }
  return new NextResponse("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Smartroom", charset="UTF-8"' },
  });
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
