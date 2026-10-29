"""Audio processing utilities."""

import io
import numpy as np
import torch
import filetype
from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio

from .config import setup_logging


def load_audio(files, gap_ms=0, fade_ms=0, s3_session=None):
    """
    Load audio data to a pydub AudioSegment from a local file path, S3 URI, or a (seekable) file object.
    If loading from a file object or S3, the file type is inferred from its magic bytes, which should work
    for all major file formats: mp3, m4a, wav, ogg, flac, mid, amr.

    Args:
        files: list of file objects, paths to local files, or s3 URIs (i.e, begins with s3://)
        gap_ms: gap in milliseconds between files when concatenating
        fade_ms: fade in/out duration in milliseconds
        s3_session: boto3 session for S3 access

    Returns:
        A pydub AudioSegement instance.
    """
    from .s3_utils import open_read

    aseg = AudioSegment.empty()
    silence = AudioSegment.silent(duration=gap_ms)
    for idx, file in enumerate(files):
        if isinstance(file, str):
            if file.startswith("s3://"):
                fp = open_read(s3_session, file)
            else:
                aseg += AudioSegment.from_file_using_temporary_files(file)
                continue
        else:
            # assume it's a file-like object
            fp = file

        # ensure we're at the beginning of the buffer
        fp.seek(0)
        fmt = filetype.guess(fp)
        # If guessing the type fails, just let the exception happen
        ext = fmt.extension
        fp.seek(0)
        # needed to load large (>4GB uncompressed) audio files.
        # https://github.com/jiaaro/pydub/issues/757
        if idx > 0 and idx < len(files)-1:
            aseg += silence
        if fade_ms > 0:
            aseg += AudioSegment.from_file_using_temporary_files(fp, format=ext).fade_in(fade_ms).fade_out(fade_ms)
        else:
            aseg += AudioSegment.from_file_using_temporary_files(fp, format=ext)
    return aseg


def audio_to_numpy(aseg):
    """Convert a PyDub AudioSegment to a float numpy array with values in [-1, 1).

    Multi-channel audio is returned interleaved (call aseg.set_channels(1) first for mono).
    """
    if aseg.sample_width == 3:
        # numpy has no 24-bit integer type; let pydub convert to 32-bit
        aseg = aseg.set_sample_width(4)
    dtypes = {1: np.int8, 2: np.int16, 4: np.int32}
    if aseg.sample_width not in dtypes:
        raise ValueError(f"Unsupported sample width: {aseg.sample_width} bytes")
    nparray = np.frombuffer(aseg.raw_data, dtype=dtypes[aseg.sample_width])
    # Normalize to float values between -1 and 1. Note that nparray is read-only, so we can't divide inplace.
    return nparray / float(2 ** (8 * aseg.sample_width - 1))


def mix_audio(signal, noise, snr):
    """Mix signal with noise at a specific SNR level.

    https://stackoverflow.com/questions/71915018/mix-second-audio-clip-at-specific-snr-to-original-audio-file-in-python

    Example usage:
        signal = np.random.randint(0, 2, 10**7) - 0.5
        # use some non-standard noise distribution
        noise = np.sin(np.random.randn(6*10**7))
        noisy = mix_audio(signal, noise, 10)
        plt.hist(noisy, bins=300);

    Args:
        signal: numpy array of signal
        noise: numpy array of noise
        snr: signal-to-noise ratio in dB

    Returns:
        AudioSegment with mixed audio
    """
    # if signal is longer than noise then repeat the noise for the duration of the audio
    noise = noise[np.arange(len(signal)) % len(noise)]

    # convert to float in case audio was uint8 or uint16, which may overflow when mixed
    noise = noise.astype(np.float32)
    signal = signal.astype(np.float32)

    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculate the gain to be applied to the noise to achieve the given SNR
    # note the 10: https://dspillustrations.com/pages/posts/misc/decibel-conversion-factor-10-or-factor-20.html
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)

    # Assuming that signal and noise are uncorrelated, calculate (a, b) such that energy of
    # a*signal + b*noise matches the energy of the input signal.
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    # mix the signals. Peaks where speech and noise coincide can exceed full scale, so clip
    # rather than letting the int16 cast wrap around.
    mix = np.clip((a * signal + b * noise) * 32767, -32768, 32767)
    return AudioSegment(mix.astype(np.int16).tobytes(), frame_rate=16000, sample_width=2, channels=1)


def get_vad_probabilities(file_path, kind: str = "silero"):
    """Compute voice activity detection probabilities for an audio file.

    Args:
        file_path: path to audio file
        kind: VAD model to use (currently only "silero" is supported)

    Returns:
        tuple of (speech_probs, window_size_seconds)
    """
    if kind == "silero":
        sampling_rate, window_size_samples = 16000, 512
        audio = read_audio(file_path, sampling_rate=sampling_rate)
        model = load_silero_vad()

        speech_probs = []
        for i in range(0, len(audio), window_size_samples):
            chunk = audio[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob = model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)

        return speech_probs, window_size_samples / sampling_rate
    else:
        raise ValueError(f"Unknown VAD kind: {kind}")
