// Clip export helpers: per-clip SRT download, caption+hashtag clipboard,
// captions.txt rendering, and the batch ZIP assembly for "Export all 9:16".
// The ZIP carries the REAL rendered MP4s — no metadata-only exports.

import { downloadBlob, downloadText } from "@/lib/zip";
import { fmtSrtTime } from "@/lib/clipforge";
import type { ClipDto, JobDto } from "@/lib/api";

/** Build and download the SRT for one clip. */
export function buildClipSrt(clip: ClipDto) {
  const srt = clip.captionCues
    .map((c, i) => `${i + 1}\n${fmtSrtTime(c.start)} --> ${fmtSrtTime(c.end)}\n${c.text}\n`)
    .join("\n");
  downloadText(srt, `clip-${clip.order + 1}.srt`);
}

/** Copy "title + hook + transcript + hashtags" metadata to the clipboard. */
export async function copyClipMeta(clip: ClipDto) {
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
export function buildCaptionsTxt(job: JobDto, clips: ClipDto[]): string {
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
export function clipFileStem(clip: ClipDto): string {
  const safe = clip.title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return `${String(clip.order + 1).padStart(2, "0")}-${safe || "clip"}`;
}

export function clipFileName(clip: ClipDto): string {
  return `${clipFileStem(clip)}.mp4`;
}

/**
 * Download one REAL rendered clip via the backend streaming endpoint
 * (Content-Disposition: attachment). A clip without media has nothing to
 * download and we say so.
 */
export async function downloadClipMp4(clip: ClipDto): Promise<void> {
  if (!clip.videoPath) {
    throw new Error("No rendered MP4 exists for this clip yet.");
  }
  const url = `/api/clips/${clip._id}/download`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (HTTP ${res.status})`);
  const blob = await res.blob();
  downloadBlob(blob, clipFileName(clip));
}
