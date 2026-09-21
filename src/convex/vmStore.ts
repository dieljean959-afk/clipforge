// vmStore — internal DB layer for vm.ts (pure DB ops, no node runtime).
//
// The workerVm singleton caches:
//   - the last REAL OCI lifecycle state + when it was checked (the power
//     switch re-queries OCI instead of trusting the last click)
//   - the worker VM's heartbeat (freshness = worker online)
//   - a pending STOP intent deferred until the worker drains its queue

import { v } from "convex/values";
import { internalMutation, internalQuery } from "./_generated/server";

const nullableStr = v.union(v.string(), v.null());

export const getState = internalQuery({
  args: {},
  handler: async (ctx) => {
    return await ctx.db
      .query("workerVm")
      .withIndex("singleton", (q) => q.eq("singleton", "vm"))
      .unique();
  },
});

/** Jobs currently in an active pipeline stage (drives the busy-stop guard). */
export const countActiveJobs = internalQuery({
  args: {},
  handler: async (ctx) => {
    const stages = ["grabbing", "transcribing", "finding", "rendering"];
    let count = 0;
    for (const stage of stages) {
      const rows = await ctx.db
        .query("jobs")
        .withIndex("by_stage", (q) => q.eq("stage", stage))
        .collect();
      count += rows.length;
    }
    return count;
  },
});

/** Record the last REAL OCI lifecycle state (or the exact API error). */
export const cacheOciState = internalMutation({
  args: {
    ociState: v.union(v.string(), v.null()),
    ociError: nullableStr,
    ociCheckedAt: v.number(),
  },
  handler: async (ctx, { ociState, ociError, ociCheckedAt }) => {
    const row = await ctx.db
      .query("workerVm")
      .withIndex("singleton", (q) => q.eq("singleton", "vm"))
      .unique();
    const base = {
      singleton: "vm" as const,
      ociState,
      ociError,
      ociCheckedAt,
      // a successful OCI read clears any stale error from before
      ...(ociState !== null ? { ociError: null } : {}),
    };
    if (row === null) {
      await ctx.db.insert("workerVm", base);
      return;
    }
    await ctx.db.patch(row._id, base);
  },
});

/** Worker heartbeat from the VM (HTTP route in http.ts calls the action). */
export const cacheHeartbeat = internalMutation({
  args: {
    modelLoaded: v.boolean(),
    pendingAction: v.optional(v.union(v.literal("START"), v.literal("STOP"))),
  },
  handler: async (ctx, { modelLoaded, pendingAction }) => {
    const now = Date.now();
    const row = await ctx.db
      .query("workerVm")
      .withIndex("singleton", (q) => q.eq("singleton", "vm"))
      .unique();
    const base = {
      singleton: "vm" as const,
      workerOk: true,
      workerLastSeenAt: now,
      workerLastSeenServerMs: now,
      modelLoaded,
    };
    if (row === null) {
      await ctx.db.insert("workerVm", {
        ...base,
        ociState: null,
        ociError: null,
        ociCheckedAt: now,
        pendingAction,
      });
      return;
    }
    await ctx.db.patch(row._id, {
      ...base,
      // START intent is cleared on first heartbeat (VM is up); STOP intent is
      // kept until vm.ts executes it after the worker reports idle.
      pendingAction: pendingAction ?? (row.pendingAction === "STOP" ? "STOP" : undefined),
    });
  },
});

/** Set/clear the deferred power intent (START handled by OCI immediately). */
export const cacheIntent = internalMutation({
  args: { action: v.union(v.literal("START"), v.literal("STOP")) },
  handler: async (ctx, { action }) => {
    const row = await ctx.db
      .query("workerVm")
      .withIndex("singleton", (q) => q.eq("singleton", "vm"))
      .unique();
    if (row === null) {
      await ctx.db.insert("workerVm", {
        singleton: "vm",
        ociState: null,
        ociError: null,
        ociCheckedAt: Date.now(),
        pendingAction: action === "STOP" ? "STOP" : undefined,
      });
      return;
    }
    await ctx.db.patch(row._id, {
      pendingAction: action === "STOP" ? "STOP" : undefined,
    });
  },
});
