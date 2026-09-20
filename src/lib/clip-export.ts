// Clip export helpers: per-clip SRT download, caption+hashtag clipboard,
// captions.txt rendering, and the batch ZIP assembly for "Export all 9:16".
// The ZIP carries the REAL rendered MP4s — no metadata-only exports.

import { buildZip, downloadBlob, downloadText } from "@/lib/zip";
import { fmtSrtTime } from "@/lib/clipforge";
import type { ClipDoc, JobDoc } from "@/lib/clipforge";

/** Build and download the SRT for one clip. */
export function buildClipSrt(clip: ClipDoc) {
  const srt = clip.captionCues
    .map((c, i) => `${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`)
    .join("\n");
  downloadText(srt, `clip-${clip.order + 1}.srt`);
}

/** Copy "title + hook + transcript + hashtags" metadata to the clipboard. */
export async function copyClipMeta(clip: ClipDoc) {
  const text = [
    clip.title,
    "",
    clip.hookText,
    "",
    clip.transcript,
    "",
    clip.hashtags.join(" "),
  ].join("\n");
  await navigator.clipboard.writeText(text);
}

/** Render the whole job's captions.txt (one SRT block per clip). */
export function buildCaptionsTxt(job: JobDoc, clips: ClipDoc[]): string {
  return clips
    .map(
      (clip, i) =>
        `=== Clip ${i + 1}: ${clip.title} (score ${clip.score}) ===\n` +
        clip.captionCues
          .map(
            (c, j) =>
              `${j + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`,
          )
          .join("\n"),
    )
    .join("\n");
}

/** Filename stem for a clip inside export archives. */
export function clipFileStem(clip: ClipDoc): string {
  const safe = clip.title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return `${String(clip.order + 1).padStart(2, "0")}-${safe || "clip"}`;
}

export function clipFileName(clip: ClipDoc): string {
  return `${clipFileStem(clip)}.mp4`;
}

/**
 * Download one REAL rendered clip. Caller must pass the fetched Blob of the
 * stored MP4 — a clip without media has nothing to download and we say so.
 */
export function downloadClipMp4(blob: Blob, clip: ClipDoc) {
  downloadBlob(blob, clipFileName(clip));
}

/**
 * Assemble the batch export ZIP: real MP4s (bytes passed in by the caller —
 * only clips that actually have media are included) + per-clip SRT +
 * captions.txt + README manifest.
 */
export function buildExportZip(
  job: JobDoc,
  clips: ClipDoc[],
  media: { clip: ClipDoc; bytes: Uint8Array }[],
): { blob: Blob; included: number; skipped: number } {
  const files: { name: string; text?: string; bytes?: Uint8Array }[] = [];
  for (const { clip, bytes } of media) {
    files.push({ name: `clips/${clipFileName(clip)}`, bytes });
  }
  files.push({
    name: "captions.txt",
    text: buildCaptionsTxt(job, clips),
  });
  files.push({
    name: "README.txt",
    text: [
      "ClipForge export — 9:16 1080x1920, H.264/AAC (ffmpeg.wasm)",
      `Source: ${job.source}`,
      `Clips included: ${media.length} of ${clips.length}`,
      "",
      "Only clip content you own or have permission to use.",
      "Clipping other creators' videos may violate platform ToS and copyright.",
      "",
      "Clip list:",
      ...clips.map((c) => {
        const has = media.some((m) => m.clip._id === c._id);
        return `  ${clipFileStem(c)} — ${c.start.toFixed(1)}s-${c.end.toFixed(1)}s, score ${c.score}${has ? "" : " (no rendered file)"}`;
      }),
    ].join("\n"),
  });
  return {
    blob: buildZip(files),
    included: media.length,
    skipped: clips.length - media.length,
  };
}
