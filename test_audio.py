from pathlib import Path

from audio_analysis import (
    analyze_audio,
    generate_spectrogram,
    generate_frequency_spectrum,
)


# ============================================================
# DECEPTOR AUDIO TEST
# ============================================================

BASE_DIR = Path(__file__).parent

natural_audio = BASE_DIR / "sample_natural.wav"
suspicious_audio = BASE_DIR / "sample_suspicious.wav"


def print_section(title):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


def run_test(name, audio_file):

    print("\n" + "=" * 60)
    print(f"DECEPTOR AUDIO ANALYSIS: {name}")
    print("=" * 60)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not audio_file.exists():
        print(f"\n[ERROR] {audio_file.name} not found!")
        print(f"Expected: {audio_file}")
        return

    # --------------------------------------------------------
    # AUDIO ANALYSIS
    # --------------------------------------------------------

    try:
        result = analyze_audio(str(audio_file))
    except Exception as e:
        print("\n[ERROR] Audio analysis failed:")
        print(e)
        return

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print_section("BASIC INFORMATION")

    print(f"File:        {audio_file.name}")
    print(f"Duration:    {result.get('duration', 'N/A')} sec")
    print(f"Sample Rate: {result.get('sample_rate', 'N/A')} Hz")

    # --------------------------------------------------------
    # AUDIO FEATURES
    # --------------------------------------------------------

    print_section("AUDIO FEATURES")

    features = result.get("features", {})

    if features:

        for key, value in features.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value}")

    else:
        print("No audio features returned.")

    # --------------------------------------------------------
    # FREQUENCY ANALYSIS
    # --------------------------------------------------------

    print_section("FREQUENCY PROFILE ANALYSIS")

    frequency = result.get("frequency_profile", {})

    if frequency:

        for key, value in frequency.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value}")

    else:
        print("No frequency analysis returned.")

    # --------------------------------------------------------
    # SILENCE ANALYSIS
    # --------------------------------------------------------

    print_section("SILENCE ANALYSIS")

    silence = result.get("silence_analysis", {})

    if silence:

        for key, value in silence.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value}")

    else:
        print("No silence analysis returned.")

    # --------------------------------------------------------
    # TRANSITION ANALYSIS
    # --------------------------------------------------------

    print_section("TRANSITION ANALYSIS")

    transition = result.get("transition_analysis", {})

    if transition:

        for key, value in transition.items():

            if isinstance(value, float):
                print(f"{key}: {value:.4f}")

            else:
                print(f"{key}: {value}")

    else:
        print("No transition analysis returned.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print_section("DECEPTOR RESULT")

    anomaly_score = result.get(
        "anomaly_score",
        "N/A"
    )

    authenticity_score = result.get(
        "authenticity_score",
        "N/A"
    )

    anomaly_level = result.get(
        "anomaly_level",
        "N/A"
    )

    print(f"Anomaly Score:      {anomaly_score}")
    print(f"Authenticity Score: {authenticity_score}")
    print(f"Anomaly Level:      {anomaly_level}")

    # --------------------------------------------------------
    # FORENSIC INDICATORS
    # --------------------------------------------------------

    print_section("FORENSIC INDICATORS")

    indicators = result.get("indicators", [])

    if indicators:

        for indicator in indicators:
            print(f"[!] {indicator}")

    else:
        print("None detected.")

    # --------------------------------------------------------
    # GENERATE SPECTROGRAM
    # --------------------------------------------------------

    print_section("GENERATING SPECTROGRAM")

    safe_name = name.lower().replace(" ", "_")

    spectrogram_path = (
        BASE_DIR /
        f"{safe_name}_spectrogram.png"
    )

    try:

        generate_spectrogram(
            str(audio_file),
            str(spectrogram_path)
        )

        print(
            f"[OK] Created: "
            f"{spectrogram_path.name}"
        )

    except Exception as e:

        print(
            "[ERROR] Spectrogram generation failed:"
        )

        print(e)

    # --------------------------------------------------------
    # GENERATE FREQUENCY SPECTRUM
    # --------------------------------------------------------

    print_section("GENERATING FREQUENCY SPECTRUM")

    frequency_path = (
        BASE_DIR /
        f"{safe_name}_frequency_spectrum.png"
    )

    try:

        generate_frequency_spectrum(
            str(audio_file),
            str(frequency_path)
        )

        print(
            f"[OK] Created: "
            f"{frequency_path.name}"
        )

    except Exception as e:

        print(
            "[ERROR] Frequency spectrum generation failed:"
        )

        print(e)


# ============================================================
# NATURAL AUDIO
# ============================================================

run_test(
    "Natural",
    natural_audio
)


# ============================================================
# SUSPICIOUS AUDIO
# ============================================================

run_test(
    "Suspicious",
    suspicious_audio
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DECEPTOR AUDIO TEST COMPLETE")
print("=" * 60)

print("\nExpected visualization files:")

print("  natural_spectrogram.png")
print("  natural_frequency_spectrum.png")
print("  suspicious_spectrogram.png")
print("  suspicious_frequency_spectrum.png")

print("\n")