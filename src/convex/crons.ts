import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();

/**
 * The render worker. Runs every 10s (2s polling on the client mirrors this
 * cadence for the progress bar) and drives the whole pipeline:
 *   - worker heartbeat -> liveness gate for job submission
 *   - one-time model warmup ("Downloading AI model (one-time, ~3GB)...")
 *   - job start + per-stage advancement with real work units
 *   - ETAs from measured/learned rates, stage timeouts, 90s stall watchdog
 */
crons.interval("render-worker", { seconds: 10 }, internal.clips.workerTick, {});

export default crons;
