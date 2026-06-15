# smartroom-control

A laptop control panel for the two Smartroom camera Pis. Shows both live MJPEG
feeds and starts/stops a recording on **both nodes with one button**.

It talks to each Pi's `smartroom_video_page.py` server (`:8000`): control calls
(`/record`, `/record/cancel`, `/record/status`) are proxied through this app's
own `app/api/*` routes server-side (the Pi sets no CORS headers), while the live
video loads directly via `<img>` (CORS-exempt). While a node is recording its
`/stream.mjpg` returns 503, so each tile swaps to the recorder's `/preview.jpg`
still until the recording ends.

> "Record All" fires both POSTs in one tick to minimize start skew, but the
> clap-at-t0 marker is still the fine-sync mechanism — don't treat the two
> streams as frame-locked.

## Configure

Nodes come from `SMARTROOM_NODES` in `.env.local` — comma-separated `id|name|host`
triples (host = bare host or IP). Defaults to the two nodes by **IP** because
`.local` mDNS from this laptop is intermittently flaky. See `.env.example`.

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
