# ADR-005: FlowWorks Datasource Client Design

**Status:** Accepted  
**Date:** 2025-10

## Context

Lark needs to ingest real sensor data from the FlowWorks API to feed the
SARIMAX models and threshold classifier. The FlowWorks Web API v2 has two
constraints that shaped the client design:

1. **Token expiry:** JWT tokens expire after 60 minutes. Long-running
   processes (the Flock runtime) need transparent token refresh.
2. **Response size cap:** ~1.5 million data points per response. 2 years
   of 5-minute data = ~210,240 points per channel — well under the cap per
   channel, but pagination is still needed for multi-year ranges.

## Decisions

**Auto token refresh with margin:** The client checks token expiry before
every request, refreshing 5 minutes early (`tokenRefreshMargin`). A 401
response also triggers a single re-auth + retry. This handles both clock
skew and edge cases where the token expires mid-batch.

**90-day pagination chunks:** When a `DateRange` query spans more than 90
days, `paginatedFetch` splits it into 90-day chunks automatically.
90 days × 288 readings/day = 25,920 points per chunk — far under the 1.5M
cap even for high-resolution data. The caller gets a single merged slice.

**`MultiChannelData` uses goroutines:** Fetching depth + rainfall + float
for both Peel sites = 6 serial requests. Concurrently fetching each channel
via a goroutine reduces wall time by ~6x. Errors are collected into
`MultiError` rather than failing fast, so a single bad channel doesn't
block the others.

**`DataValue` is a string in the API:** The FlowWorks API returns all values
as strings. Non-numeric values (sensor errors, "N/A") are preserved in the
`Raw` field and the `Value` field is set to `math.NaN()`. Callers must
check `math.IsNaN(pt.Value)` before using values in model fitting.

**No retry logic:** Transient HTTP errors are returned to the caller.
Retry/backoff belongs in the Flock handler or a higher-level ingestion
coordinator, not in the HTTP client. This keeps the client simple and
testable.

## Known Peel site/channel IDs

| Site | SiteID | Channel | ChannelID |
|---|---|---|---|
| Cavendish Cr (OVF) | 241 | Depth | 36843 |
| Cavendish Cr (OVF) | 241 | Rainfall | 21881 |
| Cavendish Cr (OVF) | 241 | Float (overflow) | 36451 |
| Clarkson GO Weir (OVF1) | 255 | Depth | 36930 |
| Clarkson GO Weir (OVF1) | 255 | Rainfall | 36503 |
| Clarkson GO Weir (OVF1) | 255 | Float (overflow) | 36493 |

Note: Float channels are binary (0/1) mechanical float switches. Known to
get stuck closed. Treat as secondary validation only — depth channel is
the primary signal for threshold classification.
