import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();

/**
 * The render worker. Runs every 10s as a SAFETY RE-ARM: the worker itself is
 * a self-scheduling mutation chain firing every 2s (each tick schedules the
 * next; a claim gate dedupes cron/client kicks into exactly ONE chain).
 * This is the serverless equivalent of the Celery worker process: same
 * deployment, DB-as-broker, no separate process to forget to start.
 *   - boot handshake logs queue + broker identity on (re)arm
 *   - one-time model warmup ("Downloading AI model (one-time, ~3GB)...")
 *   - job start + per-stage advancement with real work units
 *   - ETAs from measured/learned rates, stage timeouts, 90s stall watchdog
 */
crons.interval("render-worker", { seconds: 10 }, internal.clips.workerTick, {});

export default crons;
