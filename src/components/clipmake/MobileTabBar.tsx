import { Clapperboard, Film, Home, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Phone-style bottom navigation for the app shell.
 *
 * The dashboard is a single long page, so these are in-page anchors rather
 * than routes. The raised center action matches the mobile app language while
 * keeping "new batch" one tap away from anywhere on the page.
 */
export function MobileTabBar({ className }: { className?: string }) {
  return (
    <nav
      aria-label="App navigation"
      className={cn(
        "fixed inset-x-0 bottom-0 z-30 border-t border-border/80 bg-[#101216]/95 backdrop-blur md:hidden",
        className,
      )}
      style={{ paddingBottom: "max(0.5rem, env(safe-area-inset-bottom))" }}
    >
      <div className="mx-auto grid h-16 max-w-md grid-cols-4 items-end px-5 pb-1">
        <a
          href="/"
          className="flex flex-col items-center gap-1 text-[10px] font-medium text-muted-foreground transition hover:text-foreground"
        >
          <Home className="size-5" />
          Home
        </a>

        <a
          href="#new-batch"
          className="flex flex-col items-center gap-1 text-[10px] font-semibold text-primary"
        >
          <span className="-mt-6 flex size-12 items-center justify-center rounded-2xl border border-primary/30 bg-primary text-primary-foreground shadow-[0_8px_24px_rgba(0,0,0,0.4)] transition active:scale-95">
            <Plus className="size-6" strokeWidth={2.5} />
          </span>
          New
        </a>

        <a
          href="#batches"
          className="flex flex-col items-center gap-1 text-[10px] font-medium text-muted-foreground transition hover:text-foreground"
        >
          <Clapperboard className="size-5" />
          Batches
        </a>

        <a
          href="#clips"
          className="flex flex-col items-center gap-1 text-[10px] font-medium text-muted-foreground transition hover:text-foreground"
        >
          <Film className="size-5" />
          Clips
        </a>
      </div>
    </nav>
  );
}
