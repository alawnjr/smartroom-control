import { Dashboard } from "@/components/dashboard";
import { NODES } from "@/lib/nodes";

export default function Home() {
  return <Dashboard nodes={NODES} />;
}
