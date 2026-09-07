
import tempfile
from pathlib import Path

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from audio_analysis import analyze_audio


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Deceptor — DeepFake Audio Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #070b14;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 20%, rgba(74,144,226,.18), transparent 28%),
            linear-gradient(135deg, #101827, #080d18);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-subtitle {
        color: #9ca9bd;
        font-size: 17px;
        margin-top: 8px;
    }

    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(74,144,226,.12);
        border: 1px solid rgba(74,144,226,.25);
        color: #8fc5ff;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .5px;
        text-transform: uppercase;
    }

    .score-card {
        padding: 22px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.09);
        background: #0d1422;
        min-height: 145px;
    }

    .score-label {
        color: #8997ac;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    .score-value {
        font-size: 42px;
        font-weight: 800;
        margin-top: 6px;
    }

    .indicator {
        padding: 12px 15px;
        margin: 8px 0;
        border-radius: 12px;
        background: rgba(255, 180, 70, .07);
        border: 1px solid rgba(255, 180, 70, .14);
        color: #dce5f2;
    }

    .footer {
        color: #68768a;
        font-size: 12px;
        text-align: center;
        padding-top: 25px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <span class="badge">PS07 · Multimodal Synthetic Media Detection</span>
        <div class="hero-title">🛡️ DECEPTOR</div>
        <div class="hero-subtitle">
            Lightweight audio forensic screening for synthetic / manipulated speech.
            Upload an audio clip and inspect spectral, silence and transition anomalies.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Deceptor")
    st.caption("Audio Forensics Module")

    st.markdown(
        """
        **Signals analyzed**

        • Spectral characteristics  
        • High-frequency drop-off  
        • Silence patterns  
        • Audio/silence transitions  
        • RMS energy  
        • Zero-crossing rate  
        • Spectral flatness
        """
    )

    st.divider()

    st.caption(
        "This is a heuristic forensic screening tool, "
        "not a definitive deepfake classifier."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an audio clip",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Upload speech/audio for forensic analysis.",
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload an audio file above to start the Deceptor forensic scan."
    )

    st.markdown("### How it works")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 01 · Extract")
        st.write(
            "Decode the uploaded audio and extract classic DSP features."
        )

    with c2:
        st.markdown("#### 02 · Inspect")
        st.write(
            "Inspect spectral shape, silence regularity and transition sharpness."
        )

    with c3:
        st.markdown("#### 03 · Score")
        st.write(
            "Combine the forensic signals into an anomaly and authenticity score."
        )

else:

    # Save uploaded file temporarily
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    with st.spinner("Running Deceptor forensic analysis..."):

        try:
            result = analyze_audio(temp_path)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    # --------------------------------------------------------
    # RESULT HEADER
    # --------------------------------------------------------

    st.success("Forensic scan completed.")

    st.subheader(uploaded_file.name)

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    authenticity = float(result.get("authenticity_score", 0))
    anomaly = float(result.get("anomaly_score", 0))
    level = result.get("anomaly_level", "Unknown")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Authenticity Score</div>
                <div class="score-value">{authenticity:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Anomaly Score</div>
                <div class="score-value">{anomaly:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">Detection Level</div>
                <div class="score-value" style="font-size:28px">
                    {level}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Progress bars
    st.caption("Authenticity")
    st.progress(max(0.0, min(1.0, authenticity / 100.0)))

    st.caption("Anomaly")
    st.progress(max(0.0, min(1.0, anomaly / 100.0)))

    # --------------------------------------------------------
    # AUDIO INFO
    # --------------------------------------------------------

    st.subheader("Audio Profile")

    duration = result.get("duration", 0)
    sample_rate = result.get("sample_rate", 0)
    features = result.get("features", {})

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric("Duration", f"{duration:.2f} sec")

    with i2:
        st.metric("Sample Rate", f"{sample_rate} Hz")

    with i3:
        st.metric(
            "Spectral Centroid",
            f"{features.get('spectral_centroid_mean', 0):.0f} Hz",
        )

    with i4:
        st.metric(
            "Spectral Bandwidth",
            f"{features.get('spectral_bandwidth_mean', 0):.0f} Hz",
        )

    # --------------------------------------------------------
    # VISUAL FORENSICS
    # --------------------------------------------------------

    st.subheader("Visual Forensics")

    try:
        y, sr = librosa.load(temp_path, sr=None, mono=True)

        left, right = st.columns(2)

        # Mel spectrogram
        with left:
            st.markdown("#### Mel Spectrogram")

            S = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
            )

            S_db = librosa.power_to_db(
                S,
                ref=max,
            )

            fig, ax = plt.subplots(figsize=(8, 4.5))

            img = librosa.display.specshow(
                S_db,
                sr=sr,
                x_axis="time",
                y_axis="mel",
                ax=ax,
            )

            ax.set_title("Deceptor — Audio Spectrogram")
            ax.set_ylabel("Frequency (Mel)")
            ax.set_xlabel("Time (s)")

            fig.colorbar(
                img,
                ax=ax,
                format="%+2.0f dB",
            )

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

        # Average frequency spectrum
        with right:
            st.markdown("#### Average Frequency Spectrum")

            n = len(y)

            spectrum = abs(
                __import__("numpy").fft.rfft(y)
            )

            freqs = __import__("numpy").fft.rfftfreq(
                n,
                d=1 / sr,
            )

            fig, ax = plt.subplots(figsize=(8, 4.5))

            ax.plot(freqs, spectrum)

            ax.set_title(
                "Deceptor — Average Frequency Spectrum"
            )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude")
            ax.set_xlim(0, sr / 2)

            ax.grid(alpha=0.18)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    except Exception as exc:
        st.warning(f"Visualization could not be generated: {exc}")

    # --------------------------------------------------------
    # FORENSIC SIGNALS
    # --------------------------------------------------------

    st.subheader("Forensic Signals")

    freq = result.get("frequency_dropoff", {})
    silence = result.get("silence_analysis", {})
    transition = result.get("transition_analysis", {})

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("**Frequency Drop-off**")
        st.write(
            "Detected:"
            f" **{freq.get('detected', False)}**"
        )
        cutoff = freq.get("approx_cutoff_hz")
        if cutoff is not None:
            st.write(f"Approx. cutoff: **{cutoff:.0f} Hz**")
        st.progress(
            max(
                0.0,
                min(1.0, float(freq.get("severity", 0))),
            )
        )

    with f2:
        st.markdown("**Silence Pattern**")
        st.write(
            "Abnormal:"
            f" **{silence.get('abnormal', False)}**"
        )
        st.write(
            f"Silence ratio: "
            f"**{float(silence.get('silence_ratio', 0))*100:.1f}%**"
        )
        st.write(
            "Uniform pattern:"
            f" **{silence.get('uniform_silence_pattern', False)}**"
        )

    with f3:
        st.markdown("**Audio Transitions**")
        st.write(
            "Abrupt:"
            f" **{transition.get('abrupt', False)}**"
        )
        st.write(
            "Severity:"
            f" **{float(transition.get('severity', 0)):.2f}**"
        )
        st.progress(
            max(
                0.0,
                min(1.0, float(transition.get("severity", 0))),
            )
        )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    st.subheader("Forensic Indicators")

    indicators = result.get("indicators", [])

    for indicator in indicators:
        st.markdown(
            f"""
            <div class="indicator">
                ⚠️ {indicator}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    with st.expander("Methodology & Disclaimer"):
        st.write(
            result.get(
                "disclaimer",
                "This tool provides heuristic forensic screening only.",
            )
        )

    # Clean up temp file
    try:
        Path(temp_path).unlink(missing_ok=True)
    except Exception:
        pass


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DECEPTOR · Lightweight Multimodal Synthetic Media Detector
        · Audio Forensics Module
    </div>
    """,
    unsafe_allow_html=True,
)