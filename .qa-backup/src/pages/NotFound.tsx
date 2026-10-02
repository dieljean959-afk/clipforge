import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LogoWordmark } from "@/components/clipforge/LogoMark";

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background text-foreground">
      <LogoWordmark />
      <h1 className="text-4xl font-black tracking-tight">404</h1>
      <p className="text-sm text-muted-foreground">
        This page never rendered — wrong turn.
      </p>
      <Button asChild variant="outline" className="cursor-pointer gap-2">
        <a href="/">
          <ArrowLeft className="size-4" />
          Back to the landing page
        </a>
      </Button>
    </main>
  );
}
