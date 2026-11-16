import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import "./globals.css";

const serif = Source_Serif_4({ subsets: ["latin"], style: ["normal", "italic"], variable: "--font-serif", display: "swap" });
const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "PreFlight: model-ready data, every choice written down",
  description:
    "PreFlight turns a messy CSV into a typed table, an unfitted scikit-learn pipeline and a report, with every decision explained. Documentation and a live playground that runs the real package in your browser.",
  icons: {
    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' rx='6' fill='%23161c1f'/%3E%3Cpath d='M6 15.5 18 8l-4.5 10-2-4-5.5 1.5Z' fill='white'/%3E%3C/svg%3E",
  },
  openGraph: {
    title: "PreFlight",
    description: "Turn a messy CSV into model-ready data, with every choice written down. Try it live in your browser.",
    type: "website",
  },
};

export const viewport: Viewport = {
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0f11" },
  ],
};

// Applies a saved theme before first paint, so there is no flash.
// Saved choice wins; otherwise follow the OS. The palette's dark theme is the .dark class.
const themeScript = `try{var t=localStorage.getItem("pf-theme");var d=t?t==="dark":matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.classList.toggle("dark",d)}catch(e){}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="" />
      </head>
      <body>{children}</body>
    </html>
  );
}
