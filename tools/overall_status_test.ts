// Tests for the whole-pipeline status indicator (src/lib/clipforge.ts).
//
// The property that matters: a job must read as DONE only when the server
// says stage === "complete", and must read as FAILED (never "stuck"/running)
// when a stage errors. A job sitting at 99% in the final stage is still
// running and must never be shown as finished.
//
// Run: bun tools/overall_status_test.ts

import {
  PIPELINE_STAGES,
  QUIET_THRESHOLD_MS,
  FALLBACK_STALL_TIMEOUT_SEC,
  STALL_WARN_FRACTION,
  STALL_WARN_MS,
  fmtAgo,
  isActiveJob,
  overallStatus,
  silenceMs,
  type JobDoc,
} from "../src/lib/clipforge";

let failures: string[] = [];

function check(name: string, ok: boolean, detail = ""): void {
  console.log(`   ${ok ? "OK " : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures.push(name);
}

function job(over: Partial<JobDoc>): JobDoc {
  return {
    _id: "j1",
    sourceType: "url",
    source: "https://example.com/v",
    isUpload: false,
    stage: "queued",
    progress: 0,
    logs: [],
    ...over,
  } as JobDoc;
}

const TOTAL = PIPELINE_STAGES.length; // 4

console.log("== 1. queued");
{
  const s = overallStatus(job({ stage: "queued", progress: 0 }));
  check("queued state", s.state === "queued", s.label);
  check("queued is 0% overall", s.percent === 0, `${s.percent}%`);
}

console.log();
console.log("== 2. every running stage maps to the right position");
{
  PIPELINE_STAGES.forEach((stage, i) => {
    const s = overallStatus(job({ stage, progress: 0 }));
    check(`${stage} is stage ${i + 1} of ${TOTAL}`, s.stageNumber === i + 1, s.label);
    check(`${stage} is running`, s.state === "running");
  });
}

console.log();
console.log("== 3. overall progress is monotonic and bounded");
{
  let prev = -1;
  for (let stage = 0; stage < PIPELINE_STAGES.length; stage++) {
    for (const p of [0, 25, 50, 75, 100]) {
      const s = overallStatus(
        job({ stage: PIPELINE_STAGES[stage], progress: p }),
      );
      check(
        `overall progress never goes backwards (${PIPELINE_STAGES[stage]} @${p}%)`,
        s.percent >= prev,
        `${s.percent}% >= ${prev}%`,
      );
      prev = s.percent;
    }
  }
  check(
    "a running job never reports 100% overall",
    prev < 100,
    `max was ${prev}%`,
  );
}

console.log();
console.log("== 4. DONE requires stage === 'complete'");
{
  const done = overallStatus(
    job({ stage: "complete", progress: 100, clipCount: 7 }),
  );
  check("complete reads as done", done.state === "done", done.label);
  check("done is 100%", done.percent === 100);
  check("done label is unambiguous", done.label === "Done — clips ready", done.label);

  // The critical regression: 99% of the FINAL stage is still not done.
  const almost = overallStatus(
    job({ stage: "rendering", progress: 99, clipCount: 7 }),
  );
  check("99% of the last stage is NOT done", almost.state === "running", almost.label);
  check("99% of the last stage is capped below 100", almost.percent === 99, `${almost.percent}%`);

  const mid = overallStatus(job({ stage: "transcribing", progress: 100 }));
  check("100% of a mid stage is still not done", mid.state === "running", mid.label);
}

console.log();
console.log("== 5. FAILED never reads as stuck or running");
{
  for (const stage of [
    "grabbing",
    "transcribing",
    "finding",
    "rendering",
  ] as const) {
    const s = overallStatus(
      job({ stage: "failed", progress: 42, error: "boom" }),
    );
    check(`failure is never 'running' (from ${stage})`, s.state === "failed", s.label);
    check(
      `failure is never 'done' (from ${stage})`,
      s.state !== "done",
    );
  }
  const s = overallStatus(
    job({ stage: "failed", progress: 99, error: "renderer crashed" }),
  );
  check("a 99% failure is still Failed, not Done", s.state === "failed", s.label);
  check("failure label never leaks the literal stage name 'Failed'",
    !s.label.includes("Failed during Failed"), s.label);
}

console.log();
console.log("== 5b. the failure names the stage that broke");
{
  const cases: [string, string][] = [
    ["grabbing", "Fetching video"],
    ["transcribing", "Transcribing"],
    ["finding", "AI clip detection"],
    ["rendering", "Rendering 9:16"],
  ];
  for (const [failedStage, label] of cases) {
    const s = overallStatus(
      job({ stage: "failed", failedStage, progress: 10, error: "boom" }),
    );
    check(`names the stage: ${failedStage}`, s.label === `Failed during ${label}`, s.label);
  }
  // Older jobs persisted before `failedStage` existed must still read clearly.
  const legacy = overallStatus(job({ stage: "failed", progress: 10 }));
  check("job without failedStage falls back to 'Failed'", legacy.label === "Failed", legacy.label);
  check("legacy failure is still not done/running",
    legacy.state === "failed" && legacy.label !== "Done — clips ready");
}

console.log();
console.log("== 6. stage coverage");
{
  check("pipeline exposes 4 real stages", TOTAL === 4, `${TOTAL}`);
  check(
    "pipeline stages are the ones the stepper shows",
    PIPELINE_STAGES.every((s) => s !== "queued" && s !== "complete" && s !== "failed"),
    PIPELINE_STAGES.join(" -> "),
  );
}

console.log("\n== 7. paused is its own state, never 'running' or 'failed'");
{
  const p = overallStatus(
    job({ stage: "paused" as never, progress: 62, pausedStage: "finding" as never }),
  );
  check("a paused job does not read as running", p.state !== "running", p.state);
  check("a paused job does not read as failed", p.state !== "failed", p.state);
  check("a paused job does not read as done", p.state !== "done", p.state);
  check("it is reported as paused", p.state === "paused", p.state);
  check("it says WHERE it stopped", /finding|AI clip detection/i.test(p.label), p.label);
  check(
    "it keeps the progress it had reached",
    p.percent > 0 && p.percent <= 99,
    String(p.percent),
  );
  const noStage = overallStatus(job({ stage: "paused" as never }));
  check("a pause with no recorded stage still reads as paused", noStage.state === "paused");
}

console.log("\n== 8. liveness helpers");
{
  const now = 1_000_000_000;
  const asJob = (o: Record<string, unknown>) => o as unknown as JobDoc;
  check(
    "an in-progress job is active",
    isActiveJob(asJob({ stage: "rendering", lastProgressAt: now })),
  );
  check("a paused job is not active", !isActiveJob(asJob({ stage: "paused" })));
  check("a complete job is not active", !isActiveJob(asJob({ stage: "complete" })));
  check("a failed job is not active", !isActiveJob(asJob({ stage: "failed" })));
  check(
    "silence falls back through the stage clock",
    silenceMs(asJob({ stageStartedAt: now - 20_000 }), now) === 20_000,
  );
  check(
    "silence prefers lastProgressAt",
    silenceMs(asJob({ lastProgressAt: now - 90_000, stageStartedAt: now }), now) === 90_000,
  );
  check(
    "the quiet hint fires before the stall warning does",
    QUIET_THRESHOLD_MS < STALL_WARN_MS,
    `${QUIET_THRESHOLD_MS / 1000}s quiet vs ${STALL_WARN_MS / 1000}s warn`,
  );
  // The stall budget belongs to the server; this is only the cold-load
  // fallback, so it must still order correctly against it. The old check
  // compared QUIET to a hardcoded 90s and asserted 75s < 30s, which has been
  // false since the two constants were introduced -- and nothing ran this
  // file, so it rotted unnoticed.
  check(
    "the stall warning fires before the server's budget runs out",
    STALL_WARN_MS < FALLBACK_STALL_TIMEOUT_SEC * 1000,
    `${STALL_WARN_MS / 1000}s warn vs ${FALLBACK_STALL_TIMEOUT_SEC}s budget`,
  );
  check(
    "the warning fraction stays below 1 so the warning precedes the kill",
    STALL_WARN_FRACTION > 0 && STALL_WARN_FRACTION < 1,
    `${STALL_WARN_FRACTION}`,
  );
  check(
    "the quiet threshold sits in the 60-90s band",
    QUIET_THRESHOLD_MS >= 60_000 && QUIET_THRESHOLD_MS <= 90_000,
  );
  check("a fresh stamp reads 'just now'", fmtAgo(0) === "just now", fmtAgo(0));
  check("seconds are shown", fmtAgo(12_000) === "12s ago", fmtAgo(12_000));
  check("minutes are shown", fmtAgo(4 * 60_000) === "4m ago", fmtAgo(4 * 60_000));
  check("hours are shown", fmtAgo(2 * 3_600_000) === "2h ago", fmtAgo(2 * 3_600_000));
}

console.log();
if (failures.length) {
  console.log(`OVERALL STATUS: ${failures.length} FAILURE(S): ${failures.join(", ")}`);
  process.exit(1);
}
console.log("OVERALL STATUS: ALL CHECKS PASSED");
