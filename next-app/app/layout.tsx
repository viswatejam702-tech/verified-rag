import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Verified RAG | Zero-Hallucination Claim-Level AI",
  description: "Enterprise Retrieval-Augmented Generation with Cross-Encoder Reranking & Claim-Level NLI Verification",
  keywords: ["RAG", "Verified RAG", "NLI", "Hallucination Detection", "pgvector", "Cross-Encoder", "Supabase"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-mesh min-h-screen text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
