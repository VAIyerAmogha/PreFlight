import type { Metadata } from "next";
import PageEffects from "@/components/PageEffects";
import Topbar from "@/components/Topbar";
import TryStudio from "@/components/try/TryStudio";

export const metadata: Metadata = {
  title: "Playground · PreFlight",
  description: "Run the real pypreflight on your own CSV, in your browser. Guess the layout and types, answer its questions, and see every change it makes.",
};

export default function TryPage() {
  return (
    <>
      <PageEffects />
      <Topbar page="try" />
      <TryStudio />
    </>
  );
}
