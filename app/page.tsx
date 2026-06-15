import { Panel } from "@/components/panel";
import { NODES } from "@/lib/nodes";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Smartroom Control</h1>
        <p className="text-sm text-neutral-500">
          Live preview and synchronized recording across {NODES.length} camera nodes.
        </p>
      </header>
      <Panel nodes={NODES} />
    </main>
  );
}
