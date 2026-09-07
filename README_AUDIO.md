# Audio Forensics Module — DeepFakeSentry

Owner: Person 1 (Audio Analysis). This module only does audio. It has no
UI code and no video code, so it can be developed, tested, and demoed on
its own, then wired into the team's Streamlit app in one line.

## What it does

`analyze_audio(file_path)` loads an audio file and runs four lightweight,
explainable checks — no neural network, no downloaded model, no API calls:

1. **Basic spectral/energy features** — RMS energy, zero-crossing rate,
   spectral centroid/bandwidth/rolloff/flatness. General-purpose signal
   descriptors, useful context even when they're not directly part of the
   score.
2. **Frequency drop-off detection** — averages the mel spectrogram over
   time to get one energy curve per frequency bin, then looks for the
   single steepest bin-to-bin drop above 3 kHz. A gradual natural
   roll-off produces small, spread-out drops; a hard bandlimit (common in
   some TTS/vocoder output or re-encoded audio) shows up as one sharp
   cliff.
3. **Silence / robotic pattern analysis** — uses `librosa.effects.split`
   to find non-silent intervals, derives the silence gaps between them,
   and flags (a) unusually long gaps and (b) gaps that are suspiciously
   *uniform* in length (natural pauses vary; some synthetic pipelines
   insert very regular pauses).
4. **Abrupt transition analysis** — measures how sharply amplitude
   rises/falls in a 20ms window around each silence↔audio boundary.
   Natural onsets/offsets tend to ramp; a "clipped" edge can indicate
   artificial silence insertion or trimming.

These four signals are combined into a 0–100 **anomaly score** using
configurable weights (40 / 30 / 30, editable in `CONFIG` at the top of
`audio_analysis.py`), and `authenticity_score = 100 - anomaly_score`.

**Important:** every result includes a `disclaimer` field. None of this
proves a file is real or fake — it's a heuristic screening signal meant
to be combined with the video module's findings and human judgment.

## Install

```bash
pip install -r requirements.txt
```

(If mp3 loading fails on your machine, install `ffmpeg` — Librosa uses it
via `audioread` as a fallback for formats `soundfile` can't read
natively. WAV/FLAC work out of the box with no extra install.)

## Run

```bash
# Runs two synthesized test clips (no sample audio needed) and prints
# their full JSON reports
python test_audio.py

# Runs the pipeline on a real file
python test_audio.py path/to/clip.wav

# Or call the module directly from the command line
python audio_analysis.py path/to/clip.wav
```

## Output format

```python
{
  "file_path": "sample.wav",
  "duration": 6.0,
  "sample_rate": 22050,

  "features": {
    "rms_energy_mean": 0.081,
    "rms_energy_std": 0.034,
    "zero_crossing_rate_mean": 0.047,
    "spectral_centroid_mean": 1423.6,
    "spectral_bandwidth_mean": 1890.2,
    "spectral_rolloff_mean": 3120.4,
    "spectral_flatness_mean": 0.012
  },

  "frequency_dropoff": {
    "detected": true,
    "severity": 0.72,          # 0-1, how far past the dB threshold
    "approx_cutoff_hz": 3210.0 # where the cliff was found, or null
  },

  "silence_analysis": {
    "abnormal": true,
    "silence_ratio": 0.31,             # fraction of clip that's silence
    "num_silence_segments": 6,
    "num_long_silences": 1,            # segments >= 1.5s
    "uniform_silence_pattern": true    # gaps suspiciously similar in length
  },

  "transition_analysis": {
    "abrupt": false,
    "severity": 0.21
  },

  "anomaly_score": 68.4,          # 0-100
  "anomaly_level": "High anomaly",
  "authenticity_score": 31.6,     # 100 - anomaly_score
  "indicators": [
    "Unusual high-frequency energy drop-off near 3210 Hz",
    "Repeated, suspiciously uniform silence segments"
  ],
  "disclaimer": "Heuristic forensic screening score based on classic ..."
}
```

Every value is a plain Python `int`/`float`/`str`/`bool`/`list`/`dict` —
the whole thing is already `json.dumps`-safe, no custom classes to
convert.

### Anomaly score bands

| Score  | Level             |
|--------|-------------------|
| 0–20   | Low anomaly       |
| 21–50  | Moderate anomaly  |
| 51–75  | High anomaly      |
| 76–100 | Very high anomaly |

### How the score is built

```
anomaly_score = frequency_dropoff.severity   * 40   (weight: frequency_dropoff)
              + silence_severity              * 30   (weight: silence_abnormality)
              + transition_analysis.severity  * 30   (weight: abrupt_transitions)
```

Where `silence_severity` is the larger of the silence-ratio signal and
the uniform-pattern flag. All weights and thresholds live in the
`CONFIG` dict at the top of `audio_analysis.py` — nothing is hardcoded
deep in the logic, so you can retune live during the hackathon if a
threshold is too sensitive/insensitive on your actual test clips.

## Integrating with the team

Nothing to install beyond `requirements.txt`, no shared state, no
classes to instantiate. Any teammate can do:

```python
from audio_analysis import analyze_audio

audio_result = analyze_audio("uploaded_clip.wav")
# audio_result is a plain dict — pass it straight into:
#   - Streamlit: st.json(audio_result) or pick fields for a UI
#   - The integration layer: combined_report = {
#         "audio": audio_result,
#         "video": video_result,   # Person 2's output
#     }
#   - json.dump(audio_result, f)  for saving/demo purposes
```

For Person 3 (Streamlit UI), the fields most worth surfacing prominently
are `authenticity_score`, `anomaly_level`, and `indicators` — everything
under `features` is supporting detail, not headline numbers.

For Person 4 (integration): `analyze_audio()` takes a file path and has
no other required setup — call it after saving an uploaded file to disk
(e.g. via `st.file_uploader` -> write bytes to a temp path -> pass that
path in). It typically runs in well under a second on a short clip.
