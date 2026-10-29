# Noise mixing and SNR calculation

To simulate noisy clinical recording environments, each AMI meeting recording is
mixed with a background-noise recording at a series of target signal-to-noise
ratios (SNRs). This page describes exactly how that mixing is done and what the
resulting SNR values do (and do not) mean.

Code: `mix_audio()` in [`asr_hallucination/audio.py`](../asr_hallucination/audio.py),
called from `process_session_audio()` in
[`asr_hallucination/processing.py`](../asr_hallucination/processing.py).

## Inputs

| Input | Source | Preparation |
|-------|--------|-------------|
| Signal | AMI `Mix-Lapel.wav` (or `Mix-Headset.wav` as a fallback) for one session | Loaded with pydub and converted to a float array in [-1, 1) by `audio_to_numpy()` (8-, 16-, 24-, and 32-bit PCM supported). The code **assumes the file is already mono, 16 kHz** (true for the AMI mixes). |
| Noise | A noise recording (`--noisefile`, default `./data/noise1.flac`; may also be an `s3://` URI) | Loaded with pydub, downmixed to mono, resampled to 16 kHz, then converted to a float array. |

The paper uses two noise recordings, both made in the same small office with a
Jabra USB microphone:

| File stem | Label in the analysis | Content |
|-----------|-----------------------|---------|
| `noise1`  | Music  | Room noise plus instrumental music played on a small speaker |
| `noise2`  | Office | Room noise only (HVAC hum, hallway footsteps, street noise) |

`make_noises(outpath, source_file)` in `processing.py` shows how the 75-minute
noise files were cut from a longer source recording (skipping the first 7
minutes). The two noise recordings used in the paper are included as
`data/noise1.flac` (Music) and `data/noise2.flac` (Office): 75 minutes each,
16 kHz, mono, 16-bit. FLAC is lossless, so these are sample-for-sample identical
to the original WAVs at less than half the size. Any format FFmpeg can read
works as a noise file. 75 minutes is
longer than any AMI session, so the noise normally does not need to repeat.

## Algorithm

Given signal samples $s[i]$, noise samples $n[i]$, and a target SNR in dB:

1. **Match lengths.** If the noise is shorter than the signal it is tiled
   (looped): `noise[np.arange(len(signal)) % len(noise)]`. If it is longer, it is
   truncated to the signal length.

2. **Measure energy.** Both energies are mean squared amplitudes over the
   *entire* file:

   $$E_s = \frac{1}{N}\sum_i s[i]^2, \qquad E_n = \frac{1}{N}\sum_i n[i]^2$$

3. **Compute the noise gain** $g$, the factor that puts the noise at the
   target SNR relative to the signal:

   $$g = \sqrt{10^{-\mathrm{SNR}/10}\cdot\frac{E_s}{E_n}}
   \quad\Longleftrightarrow\quad
   10\log_{10}\frac{E_s}{g^2E_n} = \mathrm{SNR}$$

   The factor is 10 rather than 20 because $E$ is an energy (power), not an
   amplitude.

4. **Compute mixing weights.** Rather than adding $g\cdot n$ directly to $s$,
   both terms are scaled down so that the mix stays in range:

   $$a = \sqrt{\frac{1}{1+g^2}}, \qquad b = \sqrt{\frac{g^2}{1+g^2}}, \qquad a^2 + b^2 = 1$$

5. **Mix and quantize.**

   $$\mathrm{mix}[i] = a\,s[i] + b\,n[i]$$

   The result is scaled by 32767, clipped to the `int16` range, cast to `int16`,
   and wrapped in a 16 kHz mono
   pydub `AudioSegment`. It is written to
   `out/{session}/audio_{noise}_snr{SNR}.wav`.

### Why the SNR comes out right

The ratio of signal energy to noise energy in the mix is

$$\frac{a^2E_s}{b^2E_n} = \frac{E_s}{g^2E_n} = 10^{\mathrm{SNR}/10},$$

which is exactly the target. Only the *ratio* $b/a = g$ matters for the SNR; the
normalization by $\sqrt{1+g^2}$ just rescales both terms together.

The total energy of the mix is $a^2E_s + b^2E_n$ plus a cross term
$2ab\cdot\overline{s\,n}$. The cross term is ≈ 0 because the noise recording
is independent of the meeting audio, which is the "uncorrelated" assumption in
the code comment.

## Interpreting the SNR values

Keep these points in mind when reading results labeled with an SNR:

- **It is a whole-file SNR, not a speech-only SNR.** $E_s$ is averaged over
  the whole meeting, silences included. During speech, signal energy is above
  that average, so the local SNR while someone is talking is *higher* than the
  nominal value. Between utterances the local SNR is far lower. That is when
  ASR systems are most prone to inserting words (see
  [hallucination detection](hallucination_detection.md)).
- **Overall loudness is preserved only approximately.** The mix energy is
  $E_s\,(1 + 10^{-\mathrm{SNR}/10}) / (1 + g^2)$. This equals $E_s$ only when
  $E_n = E_s$. In general, the output level differs from the original
  depending on how loud the raw noise file is relative to the speech. The SNR
  is still exact; only the absolute level changes. Most ASR front ends
  normalize gain, so this should have little effect.
- **Peaks are clipped.** $a + b$ can exceed 1 (e.g. when $g = 1$,
  $a = b \approx 0.707$), so peaks where speech and noise coincide can exceed
  full scale. These samples are clipped to the `int16` range. With typical
  speech/noise crest factors this is rare and very brief.
- **SNR = +∞ means the original audio.** No mixing is done, the original AMI
  file is transcribed directly, and per-transcriber files use `none` as the
  noise name (e.g. `xscript_wsp_none_snr+inf.json`).

## SNR levels and file naming

The default SNR list (`scripts/run_transcriptions.py`) is

```
inf, 20, 15, 10, 5, 0, -5, -10, -15, -20   (dB)
```

SNRs are written into filenames with the format spec `{snr:+03.0f}`, which gives
`+20`, `+05`, `-05`, `-20`, `+inf`. Non-integer SNRs are rounded in the filename,
so two SNRs that round to the same integer will overwrite each other.

## Example

```python
from asr_hallucination.audio import load_audio, audio_to_numpy, mix_audio

signal = audio_to_numpy(load_audio(["/data/ami/ES2002a/audio/ES2002a.Mix-Lapel.wav"]))
noise = audio_to_numpy(load_audio(["data/noise2.flac"]).set_channels(1).set_frame_rate(16000))

mixed = mix_audio(signal, noise, snr=0)   # pydub AudioSegment, 16 kHz mono int16
mixed.export("ES2002a_office_snr+00.wav", format="wav")
```
