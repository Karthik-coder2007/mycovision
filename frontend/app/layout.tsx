import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "MycoVision | Mushroom AI", description: "Mushroom image classification and visual explanations." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}