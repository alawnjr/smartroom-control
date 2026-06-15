import * as React from "react";

import { cn } from "@/lib/utils";

export function Input({ className, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      className={cn(
        "h-10 w-full rounded-lg border border-neutral-700 bg-neutral-900 px-3 text-sm text-neutral-100 outline-none transition-colors placeholder:text-neutral-500 focus-visible:ring-2 focus-visible:ring-emerald-500/50 disabled:opacity-50",
        className
      )}
      {...props}
    />
  );
}
