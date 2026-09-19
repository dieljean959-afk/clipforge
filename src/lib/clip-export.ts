// Clip export helpers: per-clip SRT download, caption+hashtag clipboard,
// captions.txt rendering, and the batch ZIP assembly for "Export all 9:16".

import { buildZip, downloadBlob, downloadText } from "@/lib/zip";
import { fmtSrtTime } from "@/lib/clipforge";
import type { ClipDoc, JobDoc } from "@/lib/clipforge";

/** Build and download the SRT for one clip. */
export function buildClipSrt(clip: ClipDoc) {
  const srt = clip.captionCues
    .map(
      (c, i) =>
        `${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`,
    )
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

/**
 * Assemble the batch export ZIP: per-clip SRT + captions.txt + README
 * manifest. (Video payloads are produced by the render worker; the hosted
 * build ships the caption sidecars.)
 */
export function buildExportZip(job: JobDoc, clips: ClipDoc[]): Blob {
  const files = [
    ...clips.map((clip) => ({
      name: `clips/${clipFileStem(clip)}.srt`,
      text: clip.captionCues
        .map(
          (c, i) =>
            `${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`,
        )
        .join("\n"),
    })),
    {
      name: "captions.txt",
      text: buildCaptionsTxt(job, clips),
    },
    {
      name: "README.txt",
      text: [
        "ClipForge export — 9:16 1080x1920, libx264/AAC",
        `Source: ${job.source}`,
        `Clips: ${clips.length}`,
        "",
        "Only clip content you own or have permission to use.",
        "Clipping other creators' videos may violate platform ToS and copyright.",
        "",
        "Clip list:",
        ...clips.map(
          (c) =>
            `  ${clipFileStem(c)} — ${c.start.toFixed(1)}s-${c.end.toFixed(1)}s, score ${c.score}, ${c.hashtags.join(" ")}`,
        ),
      ].join("\n"),
    },
  ];
  return buildZip(files);
}
