import { cn } from "@/lib/utils";

/** ClipForge logo mark, drawn from scratch: two blades on a pivot, a handle
 * base and two thumb holes. No shared-library glyph, no external trademark. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("size-8", className)}
      aria-hidden="true"
    >
      <rect width="64" height="64" rx="15" fill="#0B120C" />
      <circle cx="32" cy="17" r="8" fill="#0B120C" />
      <circle cx="32" cy="17" r="3" fill="#39FF14" />
      <line
        x1="26"
        y1="17"
        x2="12"
        y2="46"
        stroke="#39FF14"
        strokeWidth="9"
        strokeLinecap="round"
      />
      <line
        x1="38"
        y1="17"
        x2="52"
        y2="46"
        stroke="#39FF14"
        strokeWidth="9"
        strokeLinecap="round"
      />
      <circle cx="25" cy="7" r="5" fill="#39FF14" />
      <circle cx="39" cy="7" r="5" fill="#39FF14" />
      <circle cx="32" cy="48" r="8" fill="#0B120C" />
      <circle cx="32" cy="48" r="3" fill="#39FF14" />
    </svg>
  );
}

export function LogoWordmark({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2", className)}>
      <LogoMark />
      <span className="text-lg font-black tracking-tight text-white">
        Clip<span className="text-primary">Make</span>
      </span>
    </span>
  );
}
