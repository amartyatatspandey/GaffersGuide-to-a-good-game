import { Navigation } from "@/components/Navigation";
import { OpeningSequence } from "@/components/OpeningSequence";
import { Hero } from "@/components/Hero";
import { TransformationStages } from "@/components/TransformationStages";
import { VirtualCoachTerminal } from "@/components/VirtualCoachTerminal";
import { PrivacyMoat } from "@/components/PrivacyMoat";
import { DownloadCTA } from "@/components/DownloadCTA";

export default function Home() {
  return (
    <main className="min-h-screen bg-pitch text-chalk selection:bg-neon/30 selection:text-neon overflow-x-hidden">
      <Navigation />
      <OpeningSequence />
      <Hero />
      <TransformationStages />
      <VirtualCoachTerminal />
      <PrivacyMoat />
      <DownloadCTA />
    </main>
  );
}
