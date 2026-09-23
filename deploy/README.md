# systemd units for the signals container

These are the unit files the signals service actually runs under, copied out of
`/etc/systemd/system/` on the deployment container so that a rebuild does not lose them.

Until 2026-08-21 none of them were in the repo. They existed only on the container, which meant the
entire schedule — the nightly scan, the macro refresh, the sandbox tick — was one `pct destroy` away
from having to be reconstructed from memory. Everything except `signals-market-scan.*` is captured
here verbatim as it was found; the market-scan pair was authored alongside SWT-1.

Nothing in this directory contains a credential. The units reference `/opt/signals/.env` by path,
and every `curl` in them targets loopback — keep it that way, since this repository is public.

## Installing

These files are **not** synced automatically. The rsync deploy only touches `/opt/signals`, so a
change here has to be applied deliberately:

```bash
# NODE is whichever host currently runs the container — it moves, so discover it rather than
# hardcoding it:  ls /etc/pve/nodes/*/lxc/<CTID>.conf
NODE=<node>
CT=<ctid>

for u in signals-market-scan.service signals-market-scan.timer; do
  ssh "root@$NODE" "pct exec $CT -- bash -c 'cat > /etc/systemd/system/$u'" < "deploy/$u"
done

ssh "root@$NODE" "pct exec $CT -- systemctl daemon-reload"
ssh "root@$NODE" "pct exec $CT -- systemctl enable --now signals-market-scan.timer"
ssh "root@$NODE" "pct exec $CT -- systemctl list-timers 'signals-*' --no-pager"
```

Two files here do **not** live in `/etc/systemd/system` — `signals-monday-check.sh` (invoked by
`signals-check-open` / `signals-check-tick`) and `signals-swt-check.sh` (invoked by
`signals-swt-check`). Both install to `/usr/local/bin/` and must be copied there separately.

The container runs on `America/Chicago`, so every `OnCalendar` below is Central wall-clock time.

## The schedule, and why it is ordered this way

| Time (CT) | Unit | What it does |
|---|---|---|
| 05:45 | `signals-market-scan` | LLM-free mechanical scan of the whole tradeable universe (SWT-1) |
| 06:15 | `signals-macro` | macro / geopolitical catalyst read |
| 06:30 | `signals-scan` | nightly watchlist scan — the only nightly job that spends model tokens |
| 07:05, 07:40 | `signals-daily-pick` | Daily Pick (DP-3) — one deep-model call; 07:40 retries only a failed run |
| 10:00, 14:20 | `signals-macro` | intraday refreshes |
| 14:35 | `signals-sandbox` | paper-trading tick |
| 14:40 | `signals-parked` | sweep parked orders |
| Mon+Tue 06:05 | `signals-swt-check` | post-wave verification → `/var/log/signals-swt.log` |

The ordering is load-bearing at the top. Market breadth is derived from the market scan, so anything
that reads a regime gate before 05:45 is reading yesterday's market. Macro and the watchlist scan
both sit behind it deliberately, so each consumes a cross-section that is minutes old rather than a
day old.

## Deploying the code itself

**Use rsync into `/opt/signals`.** `POST /api/update` used to exist and ran `git reset --hard
origin/main` — it was removed (OPS-6, 2026-09-18) rather than fixed, because it could never have
worked here: it also had no authentication at all, and it would have rolled the container backwards
past every commit the laptop repo is ahead of `origin/main`. The container's own git checkout has
been stale since `7297e19` for exactly this reason: deploys have always been file syncs, and its
working tree is current even though its `HEAD` is not.

Verify a deploy by hitting a route that only exists in the new code. `GET /api/version` reports the
new SHA some 60–90 seconds before uvicorn is actually serving it, so it will tell you the deploy
succeeded before it has.
