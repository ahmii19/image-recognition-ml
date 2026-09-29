import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Image Recognition & Object Detection",
  description:
    "Production-grade neural vision interface powered by MobileNetV2 ImageNet-1K and SSD-MobileNetV2 COCO-80.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased selection:bg-sky-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
