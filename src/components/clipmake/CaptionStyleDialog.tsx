import { useState } from "react";
import { toast } from "sonner";
import { Captions, Loader2, RotateCcw, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { api, type CaptionStyleDto, type CaptionStyleUpdate } from "@/lib/api";

/** Enough words to show the rhythm the settings produce, and nothing more. */
const SAMPLE = "we get a hook".split(" ");

function burst(sample: string[], target: number, cap: number): string[][] {
  const out: string[][] = [];
  for (let i = 0; i < sample.length; i += Math.max(1, target)) {
    out.push(sample.slice(i, i + Math.max(1, Math.min(target, cap))));
  }
  return out;
}

/**
 * Caption rhythm and highlight colour, without opening config.toml.
 *
 * These are the only knobs the dashboard exposes, and they are the ones a
 * person judging a clip actually wants to try several of: how long a burst is,
 * how much silence ends it, and what colour the word being spoken lights up.
 * Saving does not re-encode anything — the captions are burned into the clip's
 * pixels — so the dialog says so plainly and points at the button that does
 * apply it.
 */
export function CaptionStyleDialog({ onApplied }: { onApplied?: () => void }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [current, setCurrent] = useState<CaptionStyleDto | null>(null);
  const [draft, setDraft] = useState<CaptionStyleDto | null>(null);

  // Loaded when the dialog OPENS, not in an effect: the values are fetched from
  // the server, and an effect that fires a setState on mount re-renders the
  // whole dashboard just to show a form that is not visible yet. Opening is
  // the event that makes the data relevant, so it is where the fetch belongs.
  const load = async () => {
    setLoading(true);
    try {
      const next = await api.captionStyle();
      setCurrent(next);
      setDraft(next);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not load caption settings.");
    } finally {
      setLoading(false);
    }
  };

  const save = async (values: CaptionStyleUpdate) => {
    setSaving(true);
    try {
      const next = await api.saveCaptionStyle(values);
      setCurrent(next);
      setDraft(next);
      toast.success("Saved. Re-render captions to apply it to a batch.");
      onApplied?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not save caption settings.");
    } finally {
      setSaving(false);
    }
  };

  const dirty =
    !!current &&
    !!draft &&
    (current.words_per_cue !== draft.words_per_cue ||
      current.words_per_cue_max !== draft.words_per_cue_max ||
      current.pause_sec !== draft.pause_sec ||
      (current.size_scale ?? 1) !== (draft.size_scale ?? 1) ||
      current.highlight_color !== draft.highlight_color ||
      // The preset chip and the emoji switch are settings too: without these
      // two comparisons the Save button stayed disabled after picking a look.
      (current.presetKey ?? "") !== (draft.presetKey ?? "") ||
      (current.emojiEnabled ?? false) !== (draft.emojiEnabled ?? false) ||
      (current.emojiDensity ?? "medium") !== (draft.emojiDensity ?? "medium"));

  const preview = draft ?? current;
  const accent = preview?.highlight_color || "#39FF14";
  const selected = draft?.presetKey ?? current?.presetKey ?? "bold_stack";
  // "custom" badge: the set the BACKEND resolved (preset defaults vs sliders),
  // not a local guess, so the badge matches what the render will use.
  const custom = new Set(preview?.custom ?? []);
  const emojiOn = preview?.emojiEnabled ?? false;
  const emojiDensity = preview?.emojiDensity ?? "medium";

  // Preset keys in the order they should appear in the UI. The labels below
  // are known locally so the dialog can render the selector without importing
  // the backend package that also ships the CaptionPreset class.
  const CAPTION_PRESETS_ORDERED = [
    "bold_stack",
    "viral",
    "vertical_916",
    "bold_pop",
    "karaoke",
    "lower_third",
    "boxed",
  ] as const;

  type CaptionPresetKey = (typeof CAPTION_PRESETS_ORDERED)[number];

  const CAPTION_PRESET_LABELS: Record<string, string> = {
    bold_stack: "Bold stack (reference look)",
    viral: "Ssemble-style viral",
    vertical_916: "Vertical 9:16 big captions",
    bold_pop: "Bold pop (default)",
    karaoke: "Karaoke fill",
    lower_third: "Clean lower-third",
    boxed: "Boxed",
  };

  function presetLabel(key: string | undefined): string {
    if (!key) return "Caption style";
    return CAPTION_PRESET_LABELS[key] ?? "Caption style";
  }

  const presetOptions = CAPTION_PRESETS_ORDERED.map((key) => ({
    key,
    label: CAPTION_PRESET_LABELS[key] ?? "Caption style",
  }));

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) void load();
      }}
    >
      <DialogTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          className="cursor-pointer gap-1.5 text-xs"
          disabled={loading}
        >
          <Captions className="size-3.5 text-primary" />
          Caption style
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Caption style</DialogTitle>
          <DialogDescription>
            How long each caption burst is, where a burst can break, and the colour the
            spoken word lights up. Applies to every clip in a batch — never one at a time.
          </DialogDescription>
        </DialogHeader>

        {loading || !preview ? (
          <div className="flex items-center gap-2 py-6 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading…
          </div>
        ) : (
          <div className="space-y-5">
            {/* live preview — the only way to judge a rhythm is to see it */}
            <div className="rounded-lg border border-border/60 bg-black/40 p-4">
              <div className="space-y-1.5">
                {burst(SAMPLE, preview.words_per_cue, preview.words_per_cue_max).map(
                  (group, i) => {
                    // Same shape the renderer burns: one colour per chunk from
                    // the resolved palette (the preset's cycle, or the picked
                    // highlight), the rest white, and 2 stacked lines when the
                    // preset says so.
                    const chunkColor = preview.highlight_color || accent;
                    const lines =
                      preview.lines === 2 && group.length > 2
                        ? [group.slice(0, Math.ceil(group.length / 2)), group.slice(Math.ceil(group.length / 2))]
                        : [group];
                    return (
                      <p
                        key={i}
                        className="text-center font-black leading-tight"
                        style={{
                          color: "#fff",
                          fontSize: `${Math.max(12, Math.min(34, (preview.fontSize ?? 54) * 0.42))}px`,
                          letterSpacing: "0.01em",
                          textShadow: `0 0 6px ${chunkColor}`,
                          textTransform: preview.uppercase ? "uppercase" : undefined,
                        }}
                      >
                        {lines.map((line, li) => (
                          <span key={li} className="block">
                            {line.map((word, j) => (
                              <span
                                key={j}
                                style={
                                  (i % 2 === 0 ? li === 1 : li === 0)
                                    ? { color: chunkColor }
                                    : undefined
                                }
                              >
                                {word}
                                {j < line.length - 1 ? " " : ""}
                              </span>
                            ))}
                          </span>
                        ))}
                      </p>
                    );
                  },
                )}
              </div>
              <p className="mt-3 text-center text-[11px] text-muted-foreground">
                Preview only — the real thing is timed to the audio.
              </p>
            </div>

            <Field
              label="Burst length"
              hint="Words per caption. A pause or the cap still ends one sooner."
              value={`${preview.words_per_cue} word${preview.words_per_cue === 1 ? "" : "s"}`}
              pinned={custom.has("words_per_cue")}
            >
              <Slider
                min={preview.limits.words_per_cue?.min ?? 1}
                max={preview.limits.words_per_cue?.max ?? 4}
                step={1}
                value={[preview.words_per_cue]}
                onValueChange={([v]) =>
                  setDraft({ ...preview, words_per_cue: Math.min(v, preview.words_per_cue_max) })
                }
              />
            </Field>

            <Field
              label="Hard cap"
              hint="The most words a caption may ever hold."
              value={`${preview.words_per_cue_max} words`}
              pinned={custom.has("words_per_cue_max")}
            >
              <Slider
                min={preview.limits.words_per_cue_max?.min ?? 1}
                max={preview.limits.words_per_cue_max?.max ?? 6}
                step={1}
                value={[preview.words_per_cue_max]}
                onValueChange={([v]) =>
                  setDraft({ ...preview, words_per_cue_max: Math.max(v, preview.words_per_cue) })
                }
              />
            </Field>

            <Field
              label="Pause threshold"
              hint="A silence this long always ends a caption, whatever the word count."
              value={`${preview.pause_sec.toFixed(2)} s`}
              pinned={custom.has("pause_sec")}
            >
              <Slider
                min={preview.limits.pause_sec?.min ?? 0}
                max={preview.limits.pause_sec?.max ?? 1.5}
                step={preview.limits.pause_sec?.step ?? 0.05}
                value={[preview.pause_sec]}
                onValueChange={([v]) => setDraft({ ...preview, pause_sec: v })}
              />
            </Field>

            <Field
              label="Caption size"
              hint="Scales the bold-stack text, its outline and the line gap together."
              value={`${Math.round((preview.size_scale ?? 1) * 100)} %`}
              pinned={custom.has("size_scale")}
            >
              <Slider
                min={preview.limits.size_scale?.min ?? 0.8}
                max={preview.limits.size_scale?.max ?? 1.4}
                step={preview.limits.size_scale?.step ?? 0.05}
                value={[preview.size_scale ?? 1]}
                onValueChange={([v]) => setDraft({ ...preview, size_scale: v })}
              />
            </Field>

            <Field
              label="Highlight colour"
              hint="The word being spoken. Empty keeps the style's own colour."
              value={preview.highlight_color || "Style default"}
              pinned={custom.has("highlight_color")}
            >
              <div className="flex flex-wrap items-center gap-2">
                {["#39FF14", "#FFD400", "#FF5C39", "#FF2E88", "#4DA3FF", "#FFFFFF"].map((hex) => (
                  <button
                    key={hex}
                    type="button"
                    aria-label={`Highlight colour ${hex}`}
                    onClick={() => setDraft({ ...preview, highlight_color: hex })}
                    className={
                      preview.highlight_color === hex
                        ? "size-7 cursor-pointer rounded-full border-2 border-foreground scale-110"
                        : "size-7 cursor-pointer rounded-full border-2 border-border/60 hover:border-foreground/60"
                    }
                    style={{ backgroundColor: hex }}
                  />
                ))}
                <input
                  type="color"
                  aria-label="Pick a highlight colour"
                  value={/^#[0-9a-f]{6}$/i.test(preview.highlight_color) ? preview.highlight_color : "#39FF14"}
                  onChange={(e) => setDraft({ ...preview, highlight_color: e.target.value })}
                  className="size-7 cursor-pointer rounded-full border-2 border-border/60 bg-transparent p-0"
                />
                {preview.highlight_color !== "" && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 cursor-pointer px-2 text-xs"
                    onClick={() => setDraft({ ...preview, highlight_color: "" })}
                  >
                    Default
                  </Button>
                )}
              </div>
            </Field>

            <Field
              label="Emoji per caption"
              hint="Twemoji PNGs burned in one ffmpeg pass, above the caption block. Low 15% of chunks, medium 35%, high 60%."
              value={emojiOn ? emojiDensity : "Off"}
              pinned={false}
            >
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  aria-label="Emoji off"
                  onClick={() => setDraft({ ...preview, emojiEnabled: false })}
                  className={
                    !emojiOn
                      ? "cursor-pointer rounded-lg border border-foreground/30 bg-accent/40 px-3 py-1.5 text-xs font-medium text-foreground"
                      : "cursor-pointer rounded-lg border border-border/60 px-3 py-1.5 text-xs text-muted-foreground hover:border-foreground/40"
                  }
                >
                  Off
                </button>
                {(["low", "medium", "high"] as const).map((level) => (
                  <button
                    key={level}
                    type="button"
                    aria-label={`Emoji density ${level}`}
                    onClick={() => setDraft({ ...preview, emojiEnabled: true, emojiDensity: level })}
                    className={
                      emojiOn && emojiDensity === level
                        ? "cursor-pointer rounded-lg border border-foreground/30 bg-accent/40 px-3 py-1.5 text-xs font-medium capitalize text-foreground"
                        : "cursor-pointer rounded-lg border border-border/60 px-3 py-1.5 text-xs capitalize text-muted-foreground hover:border-foreground/40"
                    }
                  >
                    {level}
                  </button>
                ))}
              </div>
            </Field>

            <div className="space-y-2">
              <Label className="text-sm font-medium">Caption style</Label>
              <p className="text-xs leading-relaxed text-muted-foreground">
                The look of every clip in this batch — size, outline, position,
                and the colour the word you are speaking lights up.
              </p>
              <div className="flex flex-wrap gap-2">
                {presetOptions.map((presetOption) => {
                  const isSelected = selected === presetOption.key;
                  return (
                    <button
                      key={presetOption.key}
                      type="button"
                      aria-label={`Use ${presetOption.label}`}
                      onClick={() => setDraft({ ...preview, presetKey: presetOption.key })}
                      className={
                        isSelected
                          ? "flex items-center gap-2 rounded-lg border border-foreground/30 bg-accent/40 px-3 py-1.5 text-xs font-medium text-foreground"
                          : "flex items-center gap-2 rounded-lg border border-border/60 bg-background px-3 py-1.5 text-xs text-muted-foreground hover:border-foreground/40 hover:text-foreground"
                      }
                    >
                      <Captions className="size-3.5" />
                      {presetOption.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <p className="rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
              Captions are burned into each clip when it renders. Saving changes the next
              render — use <span className="font-medium text-foreground">Re-render captions</span>{" "}
              on a batch to apply it to clips that already exist. No restart needed.
            </p>

            <div className="flex items-center justify-between gap-2">
              <Button
                size="sm"
                variant="ghost"
                className="cursor-pointer gap-1.5 text-xs"
                disabled={saving || (preview?.custom ?? []).length === 0}
                onClick={() =>
                  void save({
                    presetKey: selected,
                    emojiEnabled: emojiOn,
                    emojiDensity,
                    reset: true,
                  })
                }
              >
                <RotateCcw className="size-3.5" />
                Reset to preset
              </Button>
              <Button
                size="sm"
                className="cursor-pointer gap-1.5 text-xs"
                disabled={saving || !dirty}
                onClick={() =>
                  void save({
                    words_per_cue: draft?.words_per_cue,
                    words_per_cue_max: draft?.words_per_cue_max,
                    pause_sec: draft?.pause_sec,
                    size_scale: draft?.size_scale,
                    highlight_color: draft?.highlight_color,
                    // Without these two the chip was a picture of a preset:
                    // the save dropped the key and every render used the default.
                    presetKey: selected,
                    emojiEnabled: emojiOn,
                    emojiDensity,
                  })
                }
              >
                {saving ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Save className="size-3.5" />
                )}
                Save
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Field({
  label,
  hint,
  value,
  pinned,
  children,
}: {
  label: string;
  hint: string;
  value: string;
  pinned: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <Label className="text-sm font-medium">{label}</Label>
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          {value}
          {pinned && (
            <span className="rounded-full border border-primary/50 bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              custom
            </span>
          )}
        </span>
      </div>
      {children}
      <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>
    </div>
  );
}
