import type { Metadata } from "next";
import { Barlow, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const barlow = Barlow({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
});

const jetBrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Gaffer's Guide | Local-First AI Football Engine",
  description: "Your Tactics. Decoded. AI Coaching Engine v1.0",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${barlow.variable} ${jetBrainsMono.variable} antialiased dark`}>
      <body className="bg-pitch text-chalk min-h-screen flex flex-col font-sans selection:bg-neon/30 selection:text-neon relative">
        {/* Fixed background grid */}
        <div className="fixed inset-0 pointer-events-none z-[-1] bg-grid opacity-50"></div>
        {children}
      </body>
    </html>
  );
}
