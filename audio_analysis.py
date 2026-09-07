"""
Deceptor — Audio Forensics Module
Person 1: Audio Analysis

Lightweight, explainable DSP heuristics for synthetic-media screening.

IMPORTANT:
This is a heuristic forensic screening tool.
It is NOT a definitive deepfake classifier and does not produce
a probability that audio is real or AI-generated.
"""

from pathlib import Path
import json

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Silence detection
    "silence_top_db": 30,

    # Frequency analysis
    "high_freq_start_hz": 1500,
    "high_freq_ratio_threshold": 0.08,
    "low_rolloff_hz": 1800,

    # Minimum frequency-profile severity required to flag it
    "frequency_detection_threshold": 0.55,

    # Silence pattern
    "silence_uniformity_std_ratio_threshold": 0.15,
    "long_silence_threshold_sec": 1.5,

    # Transition analysis
    "transition_db_threshold": 18.0,

    # Overall score weights
    "weights": {
        "frequency_profile": 50,
        "silence_abnormality": 30,
        "abrupt_transitions": 20,
    },
}


# ============================================================================
# AUDIO LOADING
# ============================================================================

def _safe_load(file_path, sr=None):
    """
    Load audio as mono.

    sr=None preserves the original sample rate.
    The signal is normalized to make transition analysis more stable.
    """

    y, sr = librosa.load(
        file_path,
        sr=sr,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    # Normalize amplitude.
    y = librosa.util.normalize(y)

    return y, sr


# ============================================================================
# BASIC AUDIO FEATURES
# ============================================================================

def _basic_features(y, sr):
    """
    Calculate basic explainable audio features.
    """

    rms = librosa.feature.rms(y=y)[0]

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=y
    )[0]

    return {
        "rms_energy_mean": float(np.mean(rms)),
        "rms_energy_std": float(np.std(rms)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
    }


# ============================================================================
# SPECTROGRAM ANALYSIS
# ============================================================================

def _spectrogram_analysis(y, sr):
    """
    Create a Mel spectrogram.

    Returns:
        S_db
        mean_db_per_bin
        mel_freqs
    """

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=sr / 2
    )

    S_db = librosa.power_to_db(
        S,
        ref=np.max
    )

    mean_db_per_bin = np.mean(
        S_db,
        axis=1
    )

    mel_freqs = librosa.mel_frequencies(
        n_mels=128,
        fmax=sr / 2
    )

    return (
        S_db,
        mean_db_per_bin,
        mel_freqs
    )


# ============================================================================
# FREQUENCY PROFILE ANALYSIS
# ============================================================================

def _frequency_profile_analysis(y, sr, config):
    """
    Detect unusually restricted high-frequency content.

    Instead of searching for one random sharp drop between adjacent
    Mel-frequency bins, this combines:

        1. High-frequency energy ratio
        2. Spectral rolloff
        3. Estimated 95% spectral cutoff

    This makes the detector less sensitive to random spectral cliffs.
    """

    n_fft = 2048

    S = np.abs(
        librosa.stft(
            y,
            n_fft=n_fft
        )
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft
    )

    power = S ** 2

    total_energy = (
        float(np.sum(power)) +
        1e-12
    )

    # ------------------------------------------------------------------
    # High-frequency energy ratio
    # ------------------------------------------------------------------

    hf_start = min(
        config["high_freq_start_hz"],
        sr / 2 * 0.8
    )

    hf_mask = freqs >= hf_start

    hf_energy = float(
        np.sum(power[hf_mask])
    )

    hf_energy_ratio = (
        hf_energy /
        total_energy
    )

    # ------------------------------------------------------------------
    # Spectral rolloff
    # ------------------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        S=power,
        sr=sr,
        roll_percent=0.85
    )[0]

    rolloff_mean = float(
        np.mean(rolloff)
    )

    # ------------------------------------------------------------------
    # Estimated spectral cutoff
    #
    # Frequency containing approximately 95% of average spectral energy.
    # ------------------------------------------------------------------

    average_power = np.mean(
        power,
        axis=1
    )

    cumulative_energy = np.cumsum(
        average_power
    )

    if cumulative_energy[-1] > 0:

        cumulative_energy /= cumulative_energy[-1]

        cutoff_index = int(
            np.searchsorted(
                cumulative_energy,
                0.95
            )
        )

        cutoff_index = min(
            cutoff_index,
            len(freqs) - 1
        )

        cutoff_hz = float(
            freqs[cutoff_index]
        )

    else:

        cutoff_hz = 0.0

    # ------------------------------------------------------------------
    # Severity calculation
    # ------------------------------------------------------------------

    # If HF energy is below the expected threshold,
    # severity increases.
    ratio_component = np.clip(
        (
            config["high_freq_ratio_threshold"]
            - hf_energy_ratio
        )
        /
        max(
            config["high_freq_ratio_threshold"],
            1e-9
        ),
        0.0,
        1.0
    )

    # If rolloff is unusually low,
    # severity increases.
    rolloff_component = np.clip(
        (
            config["low_rolloff_hz"]
            - rolloff_mean
        )
        /
        max(
            config["low_rolloff_hz"],
            1.0
        ),
        0.0,
        1.0
    )

    # High-frequency energy is more important.
    severity = float(
        0.65 * ratio_component +
        0.35 * rolloff_component
    )

    detected = (
        severity >=
        config["frequency_detection_threshold"]
    )

    return {
        "detected": bool(detected),

        "severity": round(
            severity,
            3
        ),

        "approx_cutoff_hz": round(
            cutoff_hz,
            1
        ),

        "high_frequency_energy_ratio": round(
            hf_energy_ratio,
            4
        ),

        "spectral_rolloff_mean_hz": round(
            rolloff_mean,
            1
        ),

        "analysis_band_start_hz": round(
            float(hf_start),
            1
        ),
    }


# ============================================================================
# SILENCE ANALYSIS
# ============================================================================

def _silence_analysis(y, sr, config):
    """
    Detect silence patterns.

    Indicators:

        - Overall silence ratio
        - Number of silence segments
        - Long silence segments
        - Suspiciously uniform silence durations
    """

    intervals = librosa.effects.split(
        y,
        top_db=config["silence_top_db"]
    )

    total_samples = len(y)

    duration = (
        total_samples / sr
        if sr
        else 0.0
    )

    silence_segments = []

    previous_end = 0

    for start, end in intervals:

        if start > previous_end:

            silence_segments.append(
                (
                    previous_end,
                    start
                )
            )

        previous_end = end

    # Silence at the end.
    if previous_end < total_samples:

        silence_segments.append(
            (
                previous_end,
                total_samples
            )
        )

    silence_durations = [
        (end - start) / sr
        for start, end
        in silence_segments
    ]

    total_silence = float(
        sum(silence_durations)
    )

    silence_ratio = (
        total_silence / duration
        if duration > 0
        else 0.0
    )

    # Long silence detection.
    long_silences = [
        duration
        for duration
        in silence_durations
        if duration >=
        config["long_silence_threshold_sec"]
    ]

    # Uniformity detection.
    uniform_pattern = False

    if len(silence_durations) >= 3:

        mean_duration = float(
            np.mean(silence_durations)
        )

        std_duration = float(
            np.std(silence_durations)
        )

        if mean_duration > 0:

            coefficient_variation = (
                std_duration /
                mean_duration
            )

            uniform_pattern = (
                coefficient_variation <
                config[
                    "silence_uniformity_std_ratio_threshold"
                ]
            )

    abnormal = bool(
        len(long_silences) > 0
        or uniform_pattern
    )

    return {
        "abnormal": abnormal,

        "silence_ratio": round(
            silence_ratio,
            3
        ),

        "num_silence_segments": len(
            silence_segments
        ),

        "num_long_silences": len(
            long_silences
        ),

        "uniform_silence_pattern": (
            uniform_pattern
        ),
    }, intervals


# ============================================================================
# TRANSITION ANALYSIS
# ============================================================================

def _abrupt_transition_analysis(
    y,
    sr,
    intervals,
    config
):
    """
    Analyze energy changes around audio/silence boundaries.

    Uses RMS energy rather than raw waveform sample differences.

    This is more stable because raw waveform differences can become
    artificially large simply because of waveform shape.
    """

    if len(intervals) == 0:

        return {
            "abrupt": False,
            "severity": 0.0,
            "mean_boundary_energy_change_db": 0.0,
        }

    rms = librosa.feature.rms(
        y=y,
        frame_length=1024,
        hop_length=256
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=256
    )

    changes_db = []

    for start, end in intervals:

        for boundary_sample in (
            start,
            end
        ):

            boundary_time = (
                boundary_sample /
                sr
            )

            index = int(
                np.argmin(
                    np.abs(
                        times -
                        boundary_time
                    )
                )
            )

            before_start = max(
                0,
                index - 2
            )

            after_end = min(
                len(rms),
                index + 3
            )

            if (
                index <= before_start
                or after_end <= index
            ):
                continue

            left_energy = (
                float(
                    np.mean(
                        rms[
                            before_start:index
                        ]
                    )
                )
                + 1e-9
            )

            right_energy = (
                float(
                    np.mean(
                        rms[
                            index:after_end
                        ]
                    )
                )
                + 1e-9
            )

            change_db = abs(
                20.0 *
                np.log10(
                    right_energy /
                    left_energy
                )
            )

            changes_db.append(
                change_db
            )

    if not changes_db:

        return {
            "abrupt": False,
            "severity": 0.0,
            "mean_boundary_energy_change_db": 0.0,
        }

    average_change_db = float(
        np.mean(changes_db)
    )

    # Small natural changes are not penalized.
    severity = float(
        np.clip(
            (
                average_change_db -
                6.0
            )
            /
            max(
                config[
                    "transition_db_threshold"
                ] - 6.0,
                1.0
            ),
            0.0,
            1.0
        )
    )

    abrupt = (
        severity > 0.65
    )

    return {
        "abrupt": bool(abrupt),

        "severity": round(
            severity,
            3
        ),

        "mean_boundary_energy_change_db": round(
            average_change_db,
            2
        ),
    }


# ============================================================================
# SCORE CALCULATION
# ============================================================================

def _compute_scores(
    frequency_info,
    silence_info,
    transition_info,
    config
):
    """
    Combine independent indicators into a 0–100 anomaly score.
    """

    weights = config["weights"]

    # Frequency component.
    frequency_component = (
        frequency_info["severity"] *
        weights["frequency_profile"]
    )

    # Silence component.
    silence_severity = 0.0

    if silence_info["abnormal"]:

        ratio_component = np.clip(
            silence_info["silence_ratio"] * 1.5,
            0.0,
            1.0
        )

        uniform_component = (
            1.0
            if silence_info[
                "uniform_silence_pattern"
            ]
            else 0.0
        )

        long_component = min(
            silence_info[
                "num_long_silences"
            ] / 2.0,
            1.0
        )

        silence_severity = max(
            ratio_component,
            uniform_component * 0.85,
            long_component * 0.75
        )

    silence_component = (
        silence_severity *
        weights["silence_abnormality"]
    )

    # Transition component.
    transition_component = (
        transition_info["severity"] *
        weights["abrupt_transitions"]
    )

    anomaly_score = float(
        np.clip(
            frequency_component +
            silence_component +
            transition_component,
            0.0,
            100.0
        )
    )

    authenticity_score = (
        100.0 -
        anomaly_score
    )

    return (
        round(anomaly_score, 1),
        round(authenticity_score, 1)
    )


# ============================================================================
# ANOMALY LEVEL
# ============================================================================

def _anomaly_level(score):

    if score <= 20:
        return "Low anomaly"

    if score <= 50:
        return "Moderate anomaly"

    if score <= 75:
        return "High anomaly"

    return "Very high anomaly"


# ============================================================================
# HUMAN-READABLE INDICATORS
# ============================================================================

def _build_indicators(
    frequency_info,
    silence_info,
    transition_info
):
    indicators = []

    if frequency_info["detected"]:

        indicators.append(
            "Unusually restricted high-frequency "
            "spectral content"
        )

    if (
        frequency_info[
            "high_frequency_energy_ratio"
        ] < 0.03
    ):

        indicators.append(
            "Very low energy in the "
            "high-frequency band"
        )

    if silence_info[
        "uniform_silence_pattern"
    ]:

        indicators.append(
            "Repeated, suspiciously uniform "
            "silence segments"
        )

    if (
        silence_info[
            "num_long_silences"
        ] > 0
    ):

        indicators.append(
            "Unusually long silence "
            "segment(s) detected"
        )

    if transition_info["abrupt"]:

        indicators.append(
            "Sharp energy transitions at "
            "silence/audio boundaries"
        )

    if not indicators:

        indicators.append(
            "No strong forensic indicators detected"
        )

    return indicators


# ============================================================================
# SPECTROGRAM IMAGE
# ============================================================================

def generate_spectrogram(
    file_path,
    output_path
):
    """
    Generate and save a Mel spectrogram.

    X-axis: Time
    Y-axis: Frequency
    Color/intensity: Energy in dB

    Returns the saved image path.
    """

    y, sr = _safe_load(
        file_path
    )

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=sr / 2
    )

    S_db = librosa.power_to_db(
        S,
        ref=np.max
    )

    plt.figure(
        figsize=(10, 5)
    )

    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmax=sr / 2
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        "Deceptor — Audio Spectrogram"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    return str(output_path)


# ============================================================================
# FREQUENCY SPECTRUM IMAGE
# ============================================================================

def generate_frequency_spectrum(
    file_path,
    output_path
):
    """
    Generate and save an average frequency spectrum.

    X-axis: Frequency
    Y-axis: Magnitude
    """

    y, sr = _safe_load(
        file_path
    )

    spectrum = np.abs(
        librosa.stft(
            y,
            n_fft=4096
        )
    )

    mean_spectrum = np.mean(
        spectrum,
        axis=1
    )

    frequencies = librosa.fft_frequencies(
        sr=sr,
        n_fft=4096
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        frequencies,
        mean_spectrum
    )

    plt.xlim(
        0,
        sr / 2
    )

    plt.title(
        "Deceptor — Average Frequency Spectrum"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.tight_layout()

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    return str(output_path)


# ============================================================================
# MAIN PUBLIC FUNCTION
# ============================================================================

def analyze_audio(
    file_path,
    config=None
):
    """
    Run the complete audio forensic screening pipeline.

    Parameters
    ----------
    file_path : str
        Path to WAV/FLAC/MP3/etc. supported by Librosa.

    config : dict, optional
        Custom configuration overrides.

    Returns
    -------
    dict
        JSON-serializable forensic report.
    """

    cfg = config or CONFIG

    # Load audio.
    y, sr = _safe_load(
        file_path
    )

    # Duration.
    duration = float(
        librosa.get_duration(
            y=y,
            sr=sr
        )
    )

    # Basic features.
    features = _basic_features(
        y,
        sr
    )

    # Spectrogram analysis.
    (
        S_db,
        mean_db_per_bin,
        mel_freqs
    ) = _spectrogram_analysis(
        y,
        sr
    )

    # Frequency analysis.
    frequency_info = (
        _frequency_profile_analysis(
            y,
            sr,
            cfg
        )
    )

    # Silence analysis.
    (
        silence_info,
        intervals
    ) = _silence_analysis(
        y,
        sr,
        cfg
    )

    # Transition analysis.
    transition_info = (
        _abrupt_transition_analysis(
            y,
            sr,
            intervals,
            cfg
        )
    )

    # Overall scores.
    (
        anomaly_score,
        authenticity_score
    ) = _compute_scores(
        frequency_info,
        silence_info,
        transition_info,
        cfg
    )

    # Final result.
    result = {

        "file_path": str(
            file_path
        ),

        "duration": round(
            duration,
            3
        ),

        "sample_rate": int(
            sr
        ),

        "features": features,

        "frequency_dropoff": frequency_info,

        "silence_analysis": silence_info,

        "transition_analysis": transition_info,

        "anomaly_score": anomaly_score,

        "anomaly_level": _anomaly_level(
            anomaly_score
        ),

        "authenticity_score": (
            authenticity_score
        ),

        "indicators": _build_indicators(
            frequency_info,
            silence_info,
            transition_info
        ),

        "disclaimer": (
            "Heuristic forensic screening score based "
            "on classic signal-processing indicators. "
            "It is NOT proof that audio is real or "
            "AI-generated. Combine this result with "
            "the video module's findings before drawing "
            "a conclusion."
        ),
    }

    return result


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python audio_analysis.py "
            "<path_to_audio_file>"
        )

        sys.exit(1)

    result = analyze_audio(
        sys.argv[1]
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )