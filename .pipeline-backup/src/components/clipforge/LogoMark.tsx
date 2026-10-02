import { cn } from "@/lib/utils";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("size-8", className)}
      aria-hidden="true"
    >
      <rect width="64" height="64" rx="14" className="fill-black" />
      <rect x="20" y="8" width="24" height="48" rx="7" stroke="#39FF14" strokeWidth="4" />
      <path d="M29 25.5 38 31l-9 5.5v-11Z" fill="#39FF14" />
      <rect x="24.5" y="42" width="15" height="4.5" rx="2.25" fill="#39FF14" />
      <rect x="26.5" y="49" width="11" height="4.5" rx="2.25" fill="#39FF14" opacity="0.55" />
    </svg>
  );
}

export function LogoWordmark({ className }: { className?: string }) {
  return (
    <span className={cn("flex items-center gap-2", className)}>
      <LogoMark />
      <span className="text-lg font-black tracking-tight">
        Clip<span className="text-primary">Forge</span>
      </span>
    </span>
  );
}
