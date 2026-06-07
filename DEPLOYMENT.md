# Deploying the self-learning worker on Railway

This runs the strategy as an **always-on worker** that loops forever:

```
collect live markets ─► persist ─► settle resolved trades ─► (every N cycles)
retrain a challenger & promote if it beats the champion ─► trade (paper/live)
```

It is connected to the live **Kalshi** API and (optionally) a **news/sentiment**
API, persists everything to **Postgres** + a **Volume**, and exposes a
`/health` endpoint Railway uses for health checks.

> **Money safety:** the worker is **paper by default**. It only sends real
> orders when `LIVE_TRADING=1` *and* Kalshi credentials are present, and even
> then every order passes hard risk caps and a daily-loss kill-switch.

---

## Architecture on Railway

```
┌─────────────────────────────────────────────────────────────┐
│ Railway project                                               │
│                                                               │
│  ┌──────────────┐   DATABASE_URL   ┌────────────────────────┐ │
│  │  worker      │ ───────────────► │  Postgres (plugin)     │ │
│  │  (this repo) │                  │  snapshots, trades,    │ │
│  │  Dockerfile  │                  │  resolutions, models,  │ │
│  │  :$PORT      │                  │  equity, state         │ │
│  └──────┬───────┘                  └────────────────────────┘ │
│         │ Volume @ /data (model blobs survive redeploys)      │
│         │ outbound HTTPS → Kalshi API, News API               │
└─────────────────────────────────────────────────────────────┘
```

A single worker is intentional: it's the cheapest, simplest topology and all
state lives in Postgres/Volume, so a redeploy or crash resumes cleanly.

---

## One-time setup

### 1. Create the project from this repo
- Railway → **New Project → Deploy from GitHub repo** → pick
  `tonycmooz/Claude-Kalshi-Algo`, branch `claude/kalshi-prediction-markets-oZEtf`.
- Railway detects the **Dockerfile** and `railway.toml` automatically.

### 2. Add Postgres
- In the project: **New → Database → Add PostgreSQL**.
- Railway injects `DATABASE_URL` into the worker automatically (the code
  rewrites it to the `postgresql+psycopg2` dialect). Tables are created on boot.

### 3. Add a Volume for model artifacts
- Select the worker service → **Settings → Volumes → Add Volume**.
- Mount path: **`/data`** (matches `MODEL_DIR=/data/models`).

### 4. Get Kalshi API credentials
- In the Kalshi dashboard, create an **API key**. You receive a **key id** and
  download an **RSA private key** (a `.pem` file).
- Base64-encode the PEM so it fits cleanly in an env var:
  ```bash
  base64 -w0 kalshi_private_key.pem    # Linux
  base64 -i kalshi_private_key.pem     # macOS
  ```

### 5. Set environment variables
Service → **Variables**. Start in paper mode:

| Variable | Value | Notes |
|----------|-------|-------|
| `DATA_SOURCE` | `kalshi` | use `sim` for a no-creds smoke test |
| `KALSHI_API_KEY_ID` | *(your key id)* | |
| `KALSHI_PRIVATE_KEY_B64` | *(base64 PEM from step 4)* | |
| `LIVE_TRADING` | `0` | **keep 0 until you've reviewed paper results** |
| `MIN_LABELS_TO_TRAIN` | `800` | outcomes to collect before first model |
| `CYCLE_SECONDS` | `300` | loop cadence |
| `RISK_MAX_POSITION_DOLLARS` | `50` | per-market cap |
| `RISK_DAILY_LOSS_LIMIT` | `100` | live kill-switch |
| `NEWS_API_URL` / `NEWS_API_KEY` | *(optional)* | adds sentiment features |

See `.env.example` for the full list with defaults.

### 6. Deploy
Railway builds the image and starts `python -m kalshi_algo.service.worker`.
Watch **Deploy logs**; you should see `bootstrap`, then `cycle {...}` lines.

---

## Verifying it's working

- **Health:** Railway shows the service healthy once `/health` returns 200.
- **Status JSON:** open the service URL root `/` (or `/metrics`):
  ```json
  {"cycle": 42, "bankroll": 10012.4, "opened": 2, "settled": 3,
   "open_positions": 7, "labels": 1840, "has_model": true, "live": false}
  ```
- **Cold start is expected:** for the first cycles `has_model` is `false` and no
  trades happen — the worker is accumulating resolved markets until
  `MIN_LABELS_TO_TRAIN`. Then it trains the first champion and begins paper
  trading. (To see the whole loop immediately without waiting, deploy once with
  `DATA_SOURCE=sim` — it replays a market timeline and trades within a minute.)
- **Inspect data:** connect to the Postgres plugin and query:
  ```sql
  select count(*) from snapshots;
  select count(*) filter (where status='settled'), sum(pnl) from trades;
  select id, score, active, n_train from models order by created_ts desc;
  select ts, bankroll, open_positions from equity order by ts desc limit 20;
  ```

---

## Going live (after reviewing paper performance)

1. Fund your Kalshi account.
2. Tighten risk caps to amounts you can afford to lose.
3. Set `LIVE_TRADING=1` and redeploy. The worker now sends **real** orders for
   signals that pass the model **and** the risk caps. `TRADING_ENABLED=0` is an
   instant global stop; `RISK_DAILY_LOSS_LIMIT` auto-halts live entries after a
   bad day (paper learning continues).

---

## How the continual learning works

- Every `LEARN_EVERY_CYCLES` cycles the worker runs a **bounded** self-learning
  search over *all accumulated real outcomes*, walk-forward validates each
  candidate, and trains a **challenger**.
- The challenger is **promoted to champion only if** it clears minimum quality
  gates (positive out-of-sample ROI/Sharpe, Brier < 0.25) **and** beats the
  incumbent's score. Otherwise the current champion keeps trading.
- This means the deployed strategy *improves itself over time* as more live
  data arrives — the "loop and iterate" requirement, running autonomously in
  the cloud.

---

## Cost / scaling notes

- One small worker + the Postgres plugin + a tiny volume is enough; the model is
  lightweight (gradient-boosted trees) and trains in seconds.
- To scale the search, raise `LEARN_EVERY_CYCLES` cost by editing the worker's
  `SelfLearningLoop(max_iters=...)`, or split collector/learner/trader into
  separate Railway services sharing the same Postgres (the code is already
  layered for that).
- Outbound network access to `api.elections.kalshi.com` (and your news API) must
  be permitted by your environment's egress policy.
