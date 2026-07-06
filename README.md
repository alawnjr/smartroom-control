# smartroom-control

A laptop control panel for the two Smartroom camera Pis. Starts/stops a recording
on **both nodes with one button** and shows each node's live status.

It talks to each Pi's `smartroom_video_page.py` server (`:8000`): control calls
(`/record`, `/record/cancel`, `/record/status`) are proxied through this app's
own `app/api/*` routes server-side (the Pi sets no CORS headers).

**Save All to Laptop** mirrors every recording from both nodes to disk. It reads
each node's `/recordings` listing and streams the videos into
`<SMARTROOM_SAVE_DIR>/<nodeId>/<day>/<rec>/streams/...` (default
`~/Videos/Smartroom Recordings`), skipping files already present at the same
size — so re-running only fetches new recordings.

> "Record All" fires both POSTs in one tick to minimize start skew, but the
> clap-at-t0 marker is still the fine-sync mechanism — don't treat the two
> streams as frame-locked.

## Configure

Nodes come from `SMARTROOM_NODES` in `.env.local` — comma-separated `id|name|host`
triples (host = bare host or IP). Prefer **`.local` hostnames** so DHCP IP changes
don't break the panel; this needs working IPv4 mDNS on the laptop (avahi-daemon
running + `mdns4_minimal [NOTFOUND=return]` in `/etc/nsswitch.conf`'s `hosts:`
line). Fall back to raw IPs if mDNS isn't set up. See `.env.example`.

## Run (dev)

```bash
npm install
npm run dev      # http://localhost:3000
```

## Deploy (autostart at boot via a systemd user service on port 4000)

```bash
npm run build
mkdir -p ~/.config/systemd/user
cp deploy/smartroom-control.service ~/.config/systemd/user/
systemctl --user daemon-reload
loginctl enable-linger "$USER"        # so it runs at boot without a login
systemctl --user enable --now smartroom-control.service
```

Then open `http://localhost:4000`. Refresh after code changes with:

```bash
npm run build && systemctl --user restart smartroom-control.service
```

The unit pins the nvm node path (`~/.config/nvm/versions/node/v24.16.0/bin`).
If you `nvm install` a newer node, update `PATH=` and `ExecStart=` in
`deploy/smartroom-control.service` (and reinstall it), or switch to
`ExecStart=/bin/bash -lc 'cd <dir> && npm run start'`.
