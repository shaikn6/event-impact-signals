# event-impact-signals

A research tool that classifies real-world events (war, natural disaster,
pandemic, trade dispute, monetary policy, energy shock, labor strike,
cyberattack) from live news, and maps each one to the sectors it has
historically tended to help or hurt — with a direction, confidence, and a
one-line rationale for every impact it reports.

**This is a signal/research surface, not investment advice.** Every output
is a documented hypothesis ("war raises defense spending and oil-supply
risk, so defense and energy tend up") based on historical patterns, not a
calibrated probability or a recommendation to buy or sell anything. It
never places trades, never executes anything, and the API is read-only.

## The idea

One event rarely affects the whole market the same way — it reshuffles
winners and losers. A war is bad for airlines and tourism but tends to be
good for defense contractors and energy producers. A pandemic hurts travel
but helps pharma and remote-work software. A rate hike helps bank margins
but hurts real estate and growth-tech valuations. This project surfaces
that reshuffling as structured data instead of leaving it as something
you have to reason through by hand for every headline.

```json
{
  "title": "US Reinforcing Military Presence In Middle East As Iran Tensions Rise",
  "event_type": "war_conflict",
  "event_confidence": 0.67,
  "sector_impacts": [
    {"sector": "defense", "direction": "up", "confidence": 0.8,
     "rationale": "Military spending rises during active conflicts."},
    {"sector": "energy_oil_gas", "direction": "up", "confidence": 0.7,
     "rationale": "Supply-disruption risk pushes oil/gas prices up."},
    {"sector": "airlines", "direction": "down", "confidence": 0.7,
     "rationale": "Route disruption and fuel-cost spikes hurt margins."}
  ]
}
```

## How it works

```
  ingest/           classify/            impact/
  gdelt.py    ──┐    event_type.py  ──┐   sector_mapping.py
  rss.py      ──┼──▶ (keyword rules,  │   (documented event→sector
  reddit.py   ──┘     closed category │    hypothesis table)
                       set)            │
                            │           ▼
                            └──▶ pipeline.py ──▶ store.py (SQLite)
                                                       │
                                                       ▼
                                              api.py (FastAPI, read-only)
```

- **Ingest** — pulls live articles from [GDELT](https://blog.gdeltproject.org/gdelt-doc-2-0-api/)
  (free, global news search, no API key), a curated RSS list (BBC,
  MarketWatch, Fox Business), and optionally Reddit (best-effort — see
  caveat below).
- **Classify** — a deterministic keyword-rule classifier
  (`classify/event_type.py`), not a trained model. Every category is an
  explicit, auditable list of trigger phrases, scored by phrase
  specificity so e.g. "trade war" correctly outweighs the bare word "war"
  it contains. This trades some recall for the thing a research tool
  needs most: every output traces back to *why* it fired.
- **Impact** — `impact/sector_mapping.py` is a hand-curated lookup table,
  not a trained model. It's deliberately data, not code, so the mapping
  itself can be read, debated, and extended without touching any logic.
- **Store + API** — new signals are deduplicated by URL into SQLite;
  FastAPI serves them read-only at `/signals`.

## Known limitations (read before trusting any output)

- **The event classifier is keyword-based, not NLP.** It will miss events
  phrased without any of its trigger words, and a sarcastic or
  speculative headline ("could a war save the auto industry?") can fire
  the same as a real one.
- **The sector-impact table encodes historical patterns, not guarantees.**
  A given real event can break any rule in it — a short, contained
  conflict with no supply disruption doesn't move oil the way a
  prolonged one does. Confidence reflects how reliably the pattern has
  held historically, not the probability of a specific outcome.
- **Reddit ingestion is unreliable.** Reddit's public `.json` endpoints
  reject unauthenticated requests from some networks (observed: HTTP
  403). It's off by default (`--include-reddit` to try it) and a failed
  fetch is treated as "no new posts," never as an error. For reliable
  Reddit data, register a free API app and switch `ingest/reddit.py` to
  OAuth — that's a documented upgrade path, not implemented here.
- **A poll cycle takes roughly a minute, dominated by GDELT's own
  response latency, not sleep overhead.** GDELT rate-limits to one
  request per ~5 seconds (the scheduler spaces its 5 queries accordingly
  — don't lower `_GDELT_REQUEST_SPACING_SECONDS` without checking GDELT's
  current terms), but measured against the live API, GDELT's own
  response time regularly runs ~12-14s per request even when it
  succeeds — that's the real bottleneck, and the required spacing forces
  those 5 requests to stay sequential against the same host. The
  independent RSS fetch runs concurrently with all of that instead of
  after it, and the 5 GDELT queries share one `httpx.Client` connection
  to avoid a repeated TLS handshake — real but modest savings (RSS
  typically finishes in well under a second), not a fix for GDELT's own
  latency. `fetch_articles`'s default timeout is 25s, not the more usual
  10s, specifically because a 10s timeout was measured silently
  discarding good, slow-but-successful GDELT responses as failures.

## Quick start

```bash
pip install -e ".[dev]"

# One ingest/classify/store cycle, then exit:
eis poll

# Live: background poller (every 15 min, matching GDELT's update cadence)
# + a read-only API on :8000:
eis serve
curl http://localhost:8000/signals | python3 -m json.tool
```

## Tests

```bash
pytest tests/ -q
```

All network calls are mocked in tests (`respx` for HTTP, `monkeypatch` for
`feedparser`) — the suite runs in well under a second and needs no live
network access or API keys.

## License

MIT.
