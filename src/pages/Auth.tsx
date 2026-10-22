import { useNavigate } from "react-router";
import { Sparkles, Mail, Lock, User } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Loader2 } from "lucide-react";
import { useSession, signIn, signUp } from "@/lib/auth";

interface AuthPageProps {
  returnTo?: string;
}

export function AuthPage({ returnTo = "/dashboard" }: AuthPageProps) {
  const navigate = useNavigate();
  const { signingIn, signingUp, weakError } = useSession();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const canSignup =
    mode === "signup" &&
    email.trim().length > 0 &&
    name.trim().length > 0 &&
    password.length >= 8 &&
    !signingUp;
  const canSignin =
    mode === "signin" &&
    email.trim().length > 0 &&
    password.length > 0 &&
    !signingIn;

  const doAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError(null);
    try {
      if (mode === "signup") {
        await signUp(email.trim(), password, name.trim());
      } else {
        await signIn(email.trim(), password);
      }
      toast.success(
        mode === "signup" ? "Account created — signed in." : "Signed in.",
      );
      setEmail("");
      setName("");
      setPassword("");
      navigate(returnTo);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Could not sign you in.";
      setLocalError(msg);
      toast.error(msg);
    }
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <div className="w-full max-w-md">
          <div className="mb-6 flex flex-col items-center gap-2 text-center">
            <div className="flex size-12 items-center justify-center rounded-xl border border-primary/30 bg-primary/10">
              <Sparkles className="size-6 text-primary" />
            </div>
            <p className="text-sm font-semibold tracking-tight">Clipmake</p>
            <p className="text-xs text-muted-foreground">
              Sign in to use the hosted pipeline
            </p>
          </div>

          <div className="rounded-xl border border-border bg-card p-6">
            <div className="mb-4 flex items-center gap-2">
              <Button
                type="button"
                variant={mode === "signin" ? "default" : "ghost"}
                size="sm"
                className="cursor-pointer"
                onClick={() => {
                  setMode("signin");
                  setLocalError(null);
                }}
              >
                Sign in
              </Button>
              <Button
                type="button"
                variant={mode === "signup" ? "default" : "ghost"}
                size="sm"
                className="cursor-pointer"
                onClick={() => {
                  setMode("signup");
                  setLocalError(null);
                }}
              >
                Create account
              </Button>
            </div>

            {mode === "signup" && (
              <p className="text-xs text-muted-foreground">
                Free to start. You own what you clip; the server deletes source
                files after export.
              </p>
            )}

            {weakError && (
              <p className="mb-4 text-xs text-amber-300">{weakError}</p>
            )}

            <form onSubmit={doAuth} className="flex flex-col gap-4">
              {mode === "signup" && (
                <div className="grid gap-2">
                  <Label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    Name
                  </Label>
                  <div className="flex items-center gap-2">
                    <User className="size-4 shrink-0 text-muted-foreground" />
                    <Input
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="You can be anonymous"
                      className="h-10"
                      required
                    />
                  </div>
                </div>
              )}

              <div className="grid gap-2">
                <Label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Email
                </Label>
                <div className="flex items-center gap-2">
                  <Mail className="size-4 shrink-0 text-muted-foreground" />
                  <Input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="h-10"
                    autoComplete="email"
                    required
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Password
                </Label>
                <div className="flex items-center gap-2">
                  <Lock className="size-4 shrink-0 text-muted-foreground" />
                  <Input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={mode === "signup" ? "8 characters or more" : "••••••••"}
                    className="h-10"
                    required
                    minLength={mode === "signup" ? 8 : 1}
                  />
                </div>
                {mode === "signup" && password.length > 0 && password.length < 8 && (
                  <p className="text-[11px] text-amber-300">
                    Use at least 8 characters.
                  </p>
                )}
              </div>

              {localError && <p className="text-sm text-destructive">{localError}</p>}

              <Button
                type="submit"
                size="lg"
                disabled={!canSignup && !canSignin}
                className="cursor-pointer gap-2 font-bold"
              >
                {mode === "signup" && signingUp ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Sparkles className="size-4" />
                )}
                {mode === "signup" ? "Create account" : "Sign in"}
              </Button>
            </form>

            <Separator className="my-5" />

            <p className="text-[11px] text-muted-foreground text-center">
              By continuing, you agree you own or have permission to clip the
              videos you submit.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
