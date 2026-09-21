// OciSetupWizard — first-run setup for the Oracle Cloud worker-VM control.
//
// Shown when the power switch would otherwise render a dead "state unknown"
// toggle. All 6 values are pasted by the user, tested against the real OCI
// API (a genuine GetInstance call — no faked success), and saved server-side
// only. The browser never reads the secrets back after save.

import { useState } from "react";
import { toast } from "sonner";
import { useAction } from "convex/react";
import {
  ChevronDown,
  CircleCheck,
  CircleX,
  ExternalLink,
  Loader2,
} from "lucide-react";
import { api } from "@/convex/_generated/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const FIELDS = [
  {
    key: "tenancyOcid" as const,
    label: "Tenancy OCID",
    placeholder: "ocid1.tenancy.oc1..aaaa…",
    hint: "Profile page (top-right icon) → Tenancy: <name>",
    textarea: false,
  },
  {
    key: "userOcid" as const,
    label: "User OCID",
    placeholder: "ocid1.user.oc1..aaaa…",
    hint: "Same profile page → your username → OCID",
    textarea: false,
  },
  {
    key: "fingerprint" as const,
    label: "API Key Fingerprint",
    placeholder: "aa:bb:cc:11:22:…:ff",
    hint: "Profile → API keys → the fingerprint shown next to the key you added",
    textarea: false,
  },
  {
    key: "privateKeyPem" as const,
    label: "Private Key (PEM)",
    placeholder: "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----",
    hint: "The .pem file downloaded when you added the API key — paste the full contents",
    textarea: true,
  },
  {
    key: "instanceOcid" as const,
    label: "Instance OCID",
    placeholder: "ocid1.instance.oc1…",
    hint: "Compute → Instances → your worker VM → copy the OCID",
    textarea: false,
  },
  {
    key: "region" as const,
    label: "Region",
    placeholder: "us-ashburn-1",
    hint: "Shown on the instance details page (e.g. us-phoenix-1)",
    textarea: false,
  },
];

type FieldKey = (typeof FIELDS)[number]["key"];
type FormState = Record<FieldKey, string>;

const EMPTY: FormState = {
  tenancyOcid: "",
  userOcid: "",
  fingerprint: "",
  privateKeyPem: "",
  instanceOcid: "",
  region: "",
};

type TestResult = { ok: boolean; state: string | null; error: string | null };

export function OciSetupWizard({
  className,
  onDone,
}: {
  className?: string;
  /** Called after Save or Skip so the parent can re-poll the real state. */
  onDone: () => void;
}) {
  const testConnection = useAction(api.vm.vmTestConnection);
  const saveConfig = useAction(api.vm.vmSaveConfig);
  const skipWizard = useAction(api.vm.vmSkipWizard);

  const [form, setForm] = useState<FormState>(EMPTY);
  const [showHelper, setShowHelper] = useState(false);
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<TestResult | null>(null);
  const [saving, setSaving] = useState(false);

  const complete = FIELDS.every((f) => form[f.key].trim().length > 0);

  const set = (key: FieldKey, value: string) => {
    setForm((f) => ({ ...f, [key]: value }));
    setTest(null); // a new value invalidates the previous test result
  };

  const runTest = async () => {
    setTesting(true);
    setTest(null);
    try {
      const res = (await testConnection(form)) as TestResult;
      setTest(res);
      if (res.ok) {
        toast.success(`Connected — VM is ${res.state}`);
      }
    } catch (err) {
      setTest({
        ok: false,
        state: null,
        error: err instanceof Error ? err.message : "Connection test failed.",
      });
    } finally {
      setTesting(false);
    }
  };

  const runSave = async () => {
    if (!complete) {
      toast.error("Fill in all 6 fields first.");
      return;
    }
    setSaving(true);
    try {
      await saveConfig(form);
      toast.success("Oracle VM connected — credentials stored server-side.");
      onDone();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not save credentials.");
    } finally {
      setSaving(false);
    }
  };

  const runSkip = async () => {
    try {
      await skipWizard({});
      onDone();
    } catch {
      toast.error("Could not skip — try again.");
    }
  };

  return (
    <div
      className={cn(
        "rounded-xl border border-primary/30 bg-card p-5 neon-glow-sm",
        className,
      )}
    >
      <p className="text-sm font-black tracking-tight">Connect your Oracle VM</p>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
        The ON/OFF switch drives the Oracle Cloud API, which needs 6
        credentials. They are stored server-side only — after saving, the
        browser never sees them again.
      </p>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {FIELDS.map((f) => (
          <div key={f.key} className={cn("space-y-1.5", f.textarea && "sm:col-span-2")}>
            <Label className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
              {f.label}
            </Label>
            {f.textarea ? (
              <Textarea
                value={form[f.key]}
                onChange={(e) => set(f.key, e.target.value)}
                placeholder={f.placeholder}
                rows={5}
                className="resize-y font-mono text-xs"
                spellCheck={false}
              />
            ) : (
              <Input
                value={form[f.key]}
                onChange={(e) => set(f.key, e.target.value)}
                placeholder={f.placeholder}
                className="font-mono text-xs"
                spellCheck={false}
                autoComplete="off"
              />
            )}
          </div>
        ))}
      </div>

      {/* collapsible "Where do I get these?" */}
      <button
        type="button"
        onClick={() => setShowHelper((s) => !s)}
        className="mt-3 flex cursor-pointer items-center gap-1.5 text-xs font-semibold text-primary hover:underline"
      >
        Where do I get these?
        <ChevronDown
          className={cn("size-3.5 transition-transform", showHelper && "rotate-180")}
        />
      </button>
      {showHelper && (
        <ol className="mt-2 list-decimal space-y-1.5 rounded-lg border border-border bg-muted/40 p-3 pl-7 text-xs leading-relaxed text-muted-foreground">
          <li>
            Open the{" "}
            <a
              href="https://cloud.oracle.com/"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-0.5 font-medium text-primary hover:underline"
            >
              Oracle Cloud console <ExternalLink className="size-3" />
            </a>{" "}
            and click the profile icon (top-right).
          </li>
          <li>
            My profile → <b>API keys</b> → <b>Add API Key</b> → choose
            “Generate API Key Pair” and <b>download the private key file</b>{" "}
            (that is the PEM above).
          </li>
          <li>
            Copy the <b>fingerprint</b> and your <b>User OCID</b> shown in the
            same dialog (or on the profile page afterwards).
          </li>
          <li>
            <b>Tenancy OCID</b> is on the same profile menu → Tenancy:{" "}
            {"<your-tenancy>"}.
          </li>
          <li>
            <b>Instance OCID</b>: menu ☰ → Compute → Instances → click your
            worker VM → copy the OCID.
          </li>
          <li>
            <b>Region</b> is on the instance details page, e.g.{" "}
            <code className="font-mono">us-phoenix-1</code>.
          </li>
          <li>
            Give this app permission to manage <b>only your worker VM</b>:
            Identity → <b>Policies</b> → Create Policy (create it in the{" "}
            <b>compartment that contains the VM</b>) and paste this minimal
            policy:
            <pre className="mt-2 overflow-x-auto rounded-md border border-border bg-background/80 p-2.5 font-mono text-[10px] leading-relaxed">
{`Allow group clipforge-workers to manage instance in compartment <compartment-name>
  where request.instance.id = '<instance-ocid>'`}
            </pre>
            <span className="mt-1.5 block">
              Create the <code className="font-mono">clipforge-workers</code>{" "}
              group and add your API-key user to it. That single statement
              grants START/STOP/read on this one instance — nothing else in
              the tenancy. If “Test connection” fails with{" "}
              <code className="font-mono">NotAuthorizedOrNotFound</code>, the
              policy is missing or was created in the wrong compartment.
            </span>
          </li>
        </ol>
      )}

      {/* test result — always the EXACT OCI error, never a silent failure */}
      {test !== null &&
        (test.ok ? (
          <p className="mt-3 flex items-center gap-1.5 rounded-lg border border-primary/40 bg-primary/10 px-3 py-2 text-xs font-semibold text-primary">
            <CircleCheck className="size-4 shrink-0" />
            Connected — VM is {test.state}
          </p>
        ) : (
          <p className="mt-3 flex items-start gap-1.5 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs font-semibold text-destructive">
            <CircleX className="mt-0.5 size-4 shrink-0" />
            {test.error}
          </p>
        ))}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={testing || !complete}
          className="cursor-pointer gap-1.5 text-xs"
          onClick={() => void runTest()}
        >
          {testing ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <CircleCheck className="size-3.5" />
          )}
          Test connection
        </Button>
        <Button
          size="sm"
          disabled={saving || !complete}
          className="cursor-pointer gap-1.5 text-xs"
          onClick={() => void runSave()}
        >
          {saving ? (
            <Loader2 className="size-3.5 animate-spin" />
          ) : (
            <CircleCheck className="size-3.5" />
          )}
          Save
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto cursor-pointer text-xs text-muted-foreground"
          onClick={() => void runSkip()}
        >
          Skip for now
        </Button>
      </div>
    </div>
  );
}
