# Pipeline Monitoring Report

**Started:** 2026-03-08 09:32 UTC
**Duration:** 60 minutes


### Check #6 — 09:37 UTC
- Runs: 1 | Fills: 0
0 | Rejections: 1
- Errors: 0
0 | PerfWarns: 377 | LogErrs: 0
0
- Decisions: 1

### Check #12 — 09:43 UTC
- Runs: 2 | Fills: 0
0 | Rejections: 2
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #18 — 09:49 UTC
- Runs: 3 | Fills: 0
0 | Rejections: 3
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #24 — 09:55 UTC
- Runs: 3 | Fills: 0
0 | Rejections: 3
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #30 — 10:01 UTC
- Runs: 4 | Fills: 0
0 | Rejections: 4
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #36 — 10:07 UTC
- Runs: 4 | Fills: 0
0 | Rejections: 4
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #42 — 10:13 UTC
- Runs: 5 | Fills: 0
0 | Rejections: 5
- Errors: 0
0 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #48 — 10:19 UTC
- Runs: 6 | Fills: 1 | Rejections: 5
- Errors: 3 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

## ALERT: Pipeline stopped at 2026-03-08 10:24 UTC

### Check #54 — 10:25 UTC
- Runs: 6 | Fills: 1 | Rejections: 5
- Errors: 3 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

### Check #60 — 10:31 UTC
- Runs: 7 | Fills: 2 | Rejections: 5
- Errors: 3 | PerfWarns: 578 | LogErrs: 0
0
- Decisions: 1

## Final Summary
**Ended:** 2026-03-08 10:32 UTC

| Metric | Count |
|--------|-------|
| Pipeline runs | 7 |
| Fills | 2 |
| Rejections | 5 |
| Errors | 3 |
| PerformanceWarnings | 578 |
| Logging errors | 0
0 |

### Unique Errors
```
2026-03-08 12:18:40 [execution.dydx_executor] ERROR: SL order failed — position is UNPROTECTED: Execution value DEFAULT not supported for STOP_MARKET or TAKE_PROFIT_MARKET
2026-03-08 12:18:40 [execution.dydx_executor] ERROR: TP order failed: Execution value DEFAULT not supported for STOP_MARKET or TAKE_PROFIT_MARKET
2026-03-08 12:18:41 [execution.dydx_executor] ERROR: CRITICAL: Emergency close FAILED — manual intervention required: Reduce-only orders cannot increase the position size
```
