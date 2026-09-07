🛡️ DeepFakeSentry

Visual Media Authenticity & Deepfake Anomaly Detection System

DeepFakeSentry is a lightweight computer-vision based prototype designed to analyze images and videos for visual anomalies associated with AI-generated or manipulated media.

The system focuses exclusively on visual analysis. Audio analysis is intentionally handled separately and is not included in this module.

🚀 Key Features

🖼️ Image Analysis

🎥 Video Analysis

👤 Face Detection using MediaPipe

🔬 Visual Forensic Feature Extraction

🧩 Boundary / Blending Analysis

📊 Texture & Edge Analysis

📡 Noise & Frequency Analysis

⏱️ Temporal Inconsistency Detection

📈 Video Anomaly Timeline

⚠️ Suspicious Evidence Frames

🎯 Explainable Authenticity Score

🌐 Temporary Web Portal using Gradio

☁️ Google Colab compatible

🧠 System Architecture

                  ┌─────────────────────┐
                  │     IMAGE / VIDEO   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Media Loader     │
                  └──────────┬──────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ Image        │          │ Video        │
        │ Analysis     │          │ Frame        │
        └──────┬───────┘          │ Sampling     │
               │                  └──────┬───────┘
               │                         │
               ▼                         ▼
        ┌─────────────────────────────────────┐
        │          FACE DETECTION             │
        │             MediaPipe               │
        └──────────────────┬──────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   VISUAL FORENSICS     │
              ├────────────────────────┤
              │ Texture                │
              │ Noise                  │
              │ Frequency              │
              │ Edge Density           │
              │ Boundary / Blending    │
              └────────────┬───────────┘
                           │
                           │ Video only
                           ▼
              ┌────────────────────────┐
              │ TEMPORAL ANALYSIS      │
              ├────────────────────────┤
              │ Frame Difference       │
              │ Flickering Signals     │
              │ Temporal Anomalies     │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   SCORE FUSION         │
              │ Spatial + Temporal     │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   FINAL REPORT         │
              ├────────────────────────┤
              │ Authenticity Score     │
              │ Anomaly Score          │
              │ Suspicious Timestamps  │
              │ Evidence Frames        │
              │ Timeline               │
              └────────────────────────┘

🔬 Visual Forensic Analysis

DeepFakeSentry extracts multiple visual signals from detected faces.

1. Texture Analysis

Uses image texture characteristics and Laplacian variance to identify unusual sharpness or smoothness.

2. Noise Analysis

Analyzes local image noise using a Gaussian residual. Manipulated regions can sometimes have different noise characteristics from surrounding pixels.

3. Frequency Analysis

Examines high-frequency image information to detect unusual frequency patterns.

4. Edge Density

Measures the amount and distribution of edges within the detected face.

5. Boundary / Blending Analysis

Compares the region around a detected face with its surrounding area to provide evidence of blending artifacts, inconsistent color transitions, boundary discontinuities, and texture mismatch.

🎥 Temporal Analysis

Video analysis compares normalized face crops across sampled frames.

Frame t
   ↓
Face Crop
   ↓
Normalize
   ↓
Compare
   ↓
Frame t+1

The system calculates mean visual difference, high-difference pixel ratio, and temporal anomaly score.

This can highlight potential flickering, sudden appearance changes, temporal inconsistencies, and unstable facial regions.

📊 Authenticity Score

The system generates an anomaly-based authenticity score from 0–100.

70–100 → LOW SUSPICION
40–69  → SUSPICIOUS
0–39   → HIGHLY SUSPICIOUS

Higher authenticity score means fewer detected visual anomalies.

The score is an analysis indicator, not a calibrated probability of authenticity.

⚠️ Important Limitation

DeepFakeSentry is currently a prototype anomaly detection system.

It should NOT be interpreted as definitive forensic proof.

The system detects:

Visual anomalies associated with manipulated media

rather than guaranteeing:

This media is definitely AI-generated.

Natural images, compression, lighting, camera quality, motion blur and other factors can also produce anomalies.

🤖 Pretrained Deepfake Model

The project also experimented with a lightweight pretrained image deepfake classifier.

Because pretrained models can behave differently across datasets and generation techniques, its output is not blindly treated as ground truth.

No accuracy claim is made for DeepFakeSentry itself.

🌐 Web Portal

The system includes a temporary Gradio-based web portal suitable for hackathon demonstrations.

Upload Media
     ↓
Analyze
     ↓
┌──────────────────────────┐
│ Authenticity Score       │
│ Anomaly Score            │
│ Assessment               │
│ Video Information        │
│ Suspicious Evidence      │
│ Temporal Timeline        │
└──────────────────────────┘

The portal can be launched directly from Google Colab with share=True, which provides a temporary public URL.

🛠️ Technology Stack

Technology

Purpose

Python

Core implementation

OpenCV

Image/video processing

NumPy

Numerical computation

MediaPipe

Face detection

PIL

Image handling

Matplotlib

Visualization & timeline

Scikit-learn

Supporting ML utilities

PyTorch

Model inference

Hugging Face Transformers

Pretrained model experiments

Gradio

Web interface

Google Colab

Development & deployment

📁 Project Structure

DeepFakeSentry/
│
├── README.md
├── DeepFakeSentry.pkl
│
├── notebooks/
│   └── DeepFakeSentry_Colab.ipynb
│
├── models/
│   └── face_detector.tflite
│
├── src/
│   ├── face_detection.py
│   ├── forensic_features.py
│   ├── temporal_analysis.py
│   ├── scoring.py
│   └── video_analysis.py
│
└── demo/
    └── gradio_portal.py

▶️ Running on Google Colab

Install dependencies

pip install opencv-python-headless mediapipe gradio transformers torch pillow matplotlib numpy scikit-learn

Upload media

Supported image formats:

JPG

JPEG

PNG

WEBP

Supported video formats:

MP4

AVI

MOV

MKV

Launch the portal

Run the Gradio portal cell in the Colab notebook.

Colab will provide a temporary public URL such as:

https://xxxxxxxx.gradio.live

Open the URL and upload your media.

⚡ Demo Workflow

1. Upload video
       ↓
2. Face detection
       ↓
3. Frame sampling
       ↓
4. Visual forensic analysis
       ↓
5. Temporal consistency analysis
       ↓
6. Generate anomaly score
       ↓
7. Show suspicious timestamp
       ↓
8. Show evidence frame
       ↓
9. Show anomaly timeline

Suggested Pitch

DeepFakeSentry doesn't just return a fake or real label. It analyzes visual forensic signals and temporal inconsistencies to show where a video exhibits manipulation-related anomalies.

🔐 Privacy

The current demonstration runs inside the active Google Colab runtime.

Uploaded media is processed for analysis and is not intentionally stored in an external database by DeepFakeSentry.

The Gradio share=True URL is temporary and should be treated as a demonstration endpoint rather than a production deployment.

🔮 Future Improvements

Fine-tuned deepfake classification model

Better cross-dataset validation

Face tracking across frames

Multi-face tracking

Learned manipulation localization

Better score calibration

Grad-CAM / attention-based explanations

Real-time webcam analysis

Production API

Persistent investigation reports

Integration with audio-analysis module

Combined multimodal authenticity assessment

🏆 Hackathon Value

DeepFakeSentry focuses on explainability instead of a black-box fake/real prediction.

The system provides:

Detection → Evidence → Location → Timestamp → Explanation

making the result easier for investigators, moderators and end users to understand.

👥 Scope

Current module:
🖼️ Image + 🎥 Video visual analysis

Not included:
🔊 Audio analysis

The audio component can be integrated separately as a multimodal layer in the complete system.

📜 Disclaimer

DeepFakeSentry is a research/hackathon prototype intended for demonstration and experimentation.

Its anomaly scores are heuristic indicators and should not be used as definitive evidence of media authenticity or manipulation.
