import { ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Rights-safety notice, required across the app:
 * "Only clip content you own or have permission to use."
 */
export function RightsNotice({
  className,
  compact = false,
}: {
  className?: string;
  compact?: boolean;
}) {
  if (compact) {
    return (
      <p
        className={cn(
          "flex items-center gap-1.5 text-xs text-muted-foreground",
          className,
        )}
      >
        <ShieldAlert className="size-3.5 shrink-0 text-primary/70" />
        Only clip content you own or have permission to use.
      </p>
    );
  }
  return (
    <div
      className={cn(
        "rounded-lg border border-primary/25 bg-primary/5 px-4 py-3",
        className,
      )}
    >
      <p className="flex items-start gap-2 text-sm">
        <ShieldAlert className="mt-0.5 size-4 shrink-0 text-primary" />
        <span>
          <span className="font-semibold text-foreground">
            Only clip content you own or have permission to use.
          </span>{" "}
          <span className="text-muted-foreground">
            Clipping other creators&apos; videos may violate platform terms of
            service and copyright law.
          </span>
        </span>
      </p>
    </div>
  );
}
