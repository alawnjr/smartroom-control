# Handoff — camera time sync, live presentation sync, and room audio

Written 2026-08-03. Covers the cross-camera timing work, the delayed-but-synced live
view, the live audio relay, and what is still open. Code lives in
`detect/timing_sync.py`, `detect/live_infer.py`,
`systemd/smartroom-live-infer.service`, plus `components/live-spatial.tsx` and
`app/api/v1/live/` in **smartroom-mirror** and `reolink_audio_forward.py` in
**CityOSNode**.

---

## The problem this solves

Cross-camera fusion asks whether two cameras put a person at the same room point *at
the same time*. It used to take "the same time" from `time.time()` when the inference
loop picked a frame up — which is neither when the frame was captured nor even when it
arrived, and is later by a different amount for every camera:

```
RealSense (Pi)   sensor -> depth page -> live_forward -> LAN  -> ingest
Reolink   (NVR)  sensor -> NVR encode+buffer -> RTSP -> ffmpeg -> another host -> ingest
```

The two RealSense shared a path so their errors cancelled. The Reolink cameras are
~3.5 s behind and cancel against nothing. At 1.4 m/s that is **~5 m of pure fiction**,
against a `GEO_MERGE_MM` of 600 — so Reolink↔RealSense fusion was not merely
inaccurate, it was structurally impossible (`fuse()` excluded anything >0.5 s behind).

---

## How it works now

**One timeline.** Arrival is stamped in the ingest handler (`Shared.in_recv_ms`), and
each camera's frame time is `capture_time_s()`:

| camera type | clock used | why |
|---|---|---|
| RealSense | its own librealsense per-frame stamp (`"hw"`) | a real capture instant; tracks a *varying* delay, which no constant can |
| Reolink | server arrival − measured constant (`"arrival"`) | RTSP carries no clock, so a constant is all there is |

Which is which is normally **measured** (the solve runs a second time on the
forwarders' own timestamps; a camera landing within `HW_CLOCK_TOL_MS` of the reference
shares its domain) and can be **declared** via `SMARTROOM_HW_CLOCK_CAMS` for cameras
the measurement cannot settle.

**Identity trails.** An offset relabels *when* a frame was captured; it does not make
the other cameras' past available to compare against. The gallery keeps a bounded
`(t, pos)` trail per identity (`GEO_HIST_S`, 20 s) and `_pos_at(entry, t)` looks up
the position **at the queried instant**. Without this, correcting timestamps alone
still compared two different moments.

**Presentation sync.** Correcting timestamps fixes the analysis, not the picture.
Annotated frames are held in a bounded buffer until `present_delay` after capture
(largest **observed** lag + margin), released by one thread on a shared cutoff.
Positions travel with their own frame, so video and the 3D map stay in step. A burst
that comes due together publishes the *newest* frame rather than replaying the backlog
in slow motion.

**Audio.** One source — Reolink ch1, the only enabled microphone. Transcoded to mp3 on
the forwarder, relayed as bytes (never decoded here), held by
`present_delay − cam_offset(ch1) + AUDIO_TRIM_MS` so it lines up with the delayed
video.

**Recorded audio lands in the clip itself**, not beside it: while a segment is open the
recorder takes a tap on the *undelayed* relay and writes `.{cam}_audio.mp3`, then muxes
it into the mp4 by stream copy at close and deletes the sidecar. Only ch1's clip carries
it (one microphone; five copies would echo in the synced player), and `metadata.json`
gets an `audio` block only where a track really exists. Two things that clip needs and
the others do not: its container is **retimed** with `-itsscale 30/real_fps` — the blind
CFR-30 encode made a 25 s take an 8 s container and `-shortest` then cut the sound to
match — and it is **exempt from the people test**, because ch1 localizes nobody and was
being discarded on every take, mute recordings being the result.

---

## Current state (2026-08-03 15:16 calibration)

```
camera   clock      offset
cam1     arrival   +3400.2 ms
cam2     arrival   +3613.8 ms
cam3     arrival   +3620.1 ms
cam4     arrival   +3637.4 ms
d455     hw            +0.0 ms   (reference)

presentation delay 4038 ms      audio hold 638 ms
```

The **D435 was dropped from the pipeline** (see the comment in the service unit). It
cost more than it returned: a delay that follows the load rather than a constant
(228 ms to 2.7 s within a day), a brightness swing too weak for the calibration to
read, depth over-reading ~7%, and tags seen too head-on to calibrate its pose.
Reversible by adding `camera_d435_color` back to `--cam` and `SMARTROOM_HW_CLOCK_CAMS`.

---

## Re-calibrating (self-serve)

**Live → Sync camera timing**, then turn the room lights fully OFF and back ON 3–4
times **at UNEVEN intervals** (e.g. 1 s, 4 s, 2 s, 6 s). Applies without a restart.

Uneven matters: evenly spaced flips let an alignment off by exactly one flip score as
well as the truth, and the solve refuses to guess between them. That is what rejected
cam1 on one run.

Diagnostics without re-flipping anything:

```bash
python detect/timing_sync.py --show      # stored offsets
python detect/timing_sync.py --replay    # re-solve the last run's RAW series, read-only
python detect/timing_sync.py --selftest  # the estimator's own checks
```

---

## Config surface

| env var | default | meaning |
|---|---|---|
| `SMARTROOM_LIVE_TIMING` | `calibration/live_timing.json` | measured offsets file |
| `SMARTROOM_TIME_OFFSET_<CAM>` | — | pin/neutralise one camera's offset |
| `SMARTROOM_HW_CLOCK_CAMS` | *(unit sets `camera_d455_color`)* | cameras whose own stamp is a true capture clock |
| `SMARTROOM_PRESENT_SYNC` | `1` | hold cameras to one instant |
| `SMARTROOM_PRESENT_DELAY_MS` | `0` (derive) | pin the delay by hand |
| `SMARTROOM_PRESENT_MARGIN_MS` | `400` | jitter headroom over the worst lag |
| `SMARTROOM_LIVE_AUDIO` | `1` | enable the audio relay |
| `SMARTROOM_AUDIO_CAM` | `camera_cam1_color` | which camera's mic |
| `SMARTROOM_AUDIO_TRIM_MS` | `0` | **set by ear** — lip-sync trim |
| `SMARTROOM_GEO_HIST_S` | `20` | identity trail length; must exceed the worst delay |

---

## Deploying

Server code is a git checkout that pulls; nothing is edited in place.

```bash
# dev machine
git push origin HEAD                       # smartroom-control (branch: main)
# server
cd /home/intern26/smartroom-control && git pull --ff-only origin main
systemctl --user restart smartroom-live-infer      # ~40s to reload 5 cameras' models
# mirror needs a build (node via nvm, NOT the system node 18)
cd /home/intern26/smartroom-mirror && git pull --ff-only origin master
bash -lc 'source $HOME/.nvm/nvm.sh; nvm use --silent default; npm run build'
systemctl --user restart smartroom-mirror
```

Forwarders reconnect on their own within ~6 s of a live-infer restart.

---

## Open items

1. **The 3.4 s Reolink delay has never been decomposed.** It is a measured end-to-end
   number, and it is large for RTSP (usually a few hundred ms to ~1 s). It may be the
   NVR's encoder/pre-buffer, its sub-stream GOP settings, or our ffmpeg invocation, in
   any mix. **Fixing it at source would shrink everything** — the presentation delay,
   the audio hold, the correction recordings need. Worth checking the NVR's sub-stream
   settings first (a setting, not a code change), then timing ffmpeg against the RTSP
   URL directly to split NVR-side from forwarder-side. ffmpeg's RTSP reorder buffer was
   checked and is an unlikely cause: the forwarder uses `-rtsp_transport tcp`, and TCP
   cannot reorder.
2. ~~**`reolink_audio_forward.py` has never been run.**~~ **Running since 2026-08-03.**
   Started on the NVR-facing laptop; `/api/v1/live/audio` serves a 32 kHz mono mp3 and
   the Listen button is live. Measured −41.0 dB mean / −14.0 dB peak, matching ch1's
   recorded track, so the path carries real room sound.

   First run exposed a deadlock in **both** forwarders (fixed, CityOSNode `535d975`):
   they passed `stderr=PIPE` and only read it after the loop, which is fine for a
   subprocess you wait on and wrong for one you keep alive. ffmpeg warns continuously
   on this source (~3 KB/45 s of "non monotonically increasing dts"), the few-KB OS
   buffer filled in about a minute, and ffmpeg then **blocked writing stderr and stopped
   producing audio** — the forwarder sat alive on a stream that would never move again,
   logging nothing. `drain_stderr()` now pumps it on a daemon thread. Worth knowing
   before writing a third forwarder.

   Still manual: it is a `nohup` process on that laptop, so it dies on reboot or sleep.
   Wants a scheduled task if the live feed should be permanent.

   `SMARTROOM_AUDIO_TRIM_MS` is still `0` — the lip-sync trim has to be set by ear by
   someone in the room, since no calibration can measure an audio path.
3. **`smartroom-live-forward@d435` is still enabled on the Pi** and will reconnect and
   404 in a retry loop. Harmless but noisy: `systemctl --user disable --now
   smartroom-live-forward@d435`.
4. ~~**The mirror's recorded playback does not consume `sync_ms`.**~~ **Done
   2026-08-04** (smartroom-mirror `ad57b14`). It aligned cameras on
   `hw_timestamp_ms`, which is not one clock — the D455 stamps librealsense capture
   time, the Reolinks stamp the forwarding host's wall clock at decode, ~1.6 s later
   for the same frame — and then zeroed each clip against whichever CSV loaded first.
   Five tiles, five timelines, and a transport reporting 1.0 s for a 20 s take.
   `parseCsv` now reads `sync_ms`, the player parks the absolute clocks and rebases the
   session on the earliest camera (`rebaseAbs`) once every CSV is in. On
   rec_20260804_120357 the five cameras start within 75 ms of each other and the
   session is 19.97 s. `parseFrameRel` stays clip-relative — it feeds `buildRemap`,
   which is a within-clip container→real conversion — so `remap`'s semantics did not
   change. 8 checks in `scripts/sync-clock.test.ts`.
5. **cam1 records but counts no people.** Its segments used to be discarded for it; they
   are now kept because they carry the audio, so the take survives — but its people
   count is still 0 and nothing downstream sees cam1 detections. `found` only
   contains people who could be *localized*, and cam1 sits 2.6 m up in a corner where
   the floor-ray exceeds `MAX_RAY_REACH_MM`. Deprioritised by the user; the fix is to
   count detections that pass the pose-quality gate rather than localization successes.
6. **The recordings page still ships ~1.77 MB of HTML** (61% embedded URLs for 405
   sessions). Thumbnails were the dominant cost and are fixed; this is the remainder.

---

## Traps worth knowing before touching this

- **`frameCaptureMs` agreement does not prove the cameras are in sync.** The presenter
  releases each camera when its *claimed* capture time comes due, so the claims agree
  by construction. It validates the presenter, not whether each capture time is right.
  Only the lights test — or waving a hand — checks the latter. I reported a "median
  spread 56 ms" that was weaker evidence than it looked.
- **A correlation never tells you *what* it correlated on.** Two cameras watching one
  person move correlate perfectly well. That is why every sample carries mean
  brightness and the solve refuses unless the room's brightness actually swung. A run
  once published a person's motion as a timing offset.
- **Edge matching beats energy correlation, and correlation has a hard ceiling.** It
  can only find a lag inside `MAX_LAG_MS` (±2 s) and needs the spike *shape* to
  survive; the Reolink sub stream at 10 fps through the NVR's encoder blunts it. The
  light *step*'s 50% crossing is exposure-, gain- and rate-independent, and has no lag
  limit. Selftest: a 2600 ms lag recovered exactly where correlation says −1771 ms at
  r=0.06.
- **`MIN_CORRELATION` only discriminates with enough window**, because the peak is a
  max over ~800 lags. Worst-of-12 noise peaks vs a real ~0.85 signal:
  5 s→0.398, 10 s→0.275, 25 s→0.197. At the old 3 s minimum the noise floor sat *above*
  the 0.35 bar. Hence `MIN_OVERLAP_MS = 10000`.
- **Two matched edges is enough when nothing else explains them.** Raising the minimum
  to three rejected a correct answer. The real guard is the ambiguity check.
- **A negative arrival-clock offset is impossible**, once the reference is on a true
  capture clock — it would mean frames arriving before capture. One was applied
  (−2853 ms) before that guard existed.
- **Don't give a live panel its own poll.** The timing panel had one; it silently
  stopped updating while the 250 ms positions poll beside it kept working, so the panel
  froze on the arm request's reply through a run that succeeded server-side. Everything
  rides `/positions` now, and the arm reply is discarded so the poll is the only writer.
- **Never hardcode a rejection reason in the UI.** The panel asserted "they saw no
  light change" for every rejection; cam1 had seen the lights perfectly and was
  rejected for ambiguity, sending the reader to check the room lighting.
- **A mode's mechanism can outlive the mode.** Segments were cut on a 3-minute
  wall-clock grid, which is right for the always-on recorder — but always-on has been
  off by default for a while and the grid was never conditioned on it, so it went on
  cutting hand-started takes. Pressing Record at 11:44:58 gave a 2-second clip plus a
  separate recording. `RECORD.segment()` now decides: the take's own start when someone
  started it, the wall clock only for always-on, `TAKE_MAX_S` a ceiling rather than a
  rhythm. `SMARTROOM_SEGMENT_S` in the unit is now dead config unless always-on is
  turned back on.
- **The keep/discard rule silently eats whatever else a clip carries.** Same shape as
  the muting bug below, one layer down: once ch1's clip became the *only* place the room
  audio is stored, "nobody in it, throw it away" started throwing the sound away too —
  and ch1 is precisely the camera that can never localize anyone. Anything new attached
  to a segment has to be checked against that test.
- **A segment's mp4 duration is not the take's duration.** Frames are piped to ffmpeg at
  a blind `-r 30` while the Reolink sub streams deliver ~10 fps, so a 30 s take yields a
  10 s container. The timestamps CSV is the authority and the mirror's player uses it;
  only the audio-bearing clip is retimed, because there the lie was destructive.
- **Muting a camera must not stop it recording.** Segment keep/discard counted
  *published* positions, so with `SMARTROOM_SPATIAL_ONLY` set, five of six cameras
  discarded every segment while their startup line promised recording continued.
