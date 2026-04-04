# Deepfake Audio Detection using Sub-Band Frequency Attention Conformer (SB-FAC)

A novel architecture for detecting AI-generated fake audio by analyzing different frequency bands independently, achieving **1.61% Equal Error Rate** on the ASVspoof 2019 benchmark.

## Overview

Modern speech synthesis can generate audio that sounds real to human ears. This project addresses the challenge of automatically detecting such deepfake audio by introducing a **Sub-Band Frequency Attention Conformer (SB-FAC)** that learns which frequency regions contain spoofing artifacts for each input sample.

## Key Idea

Different spoofing methods leave artifacts in different frequency bands — voice conversion distorts 2-4 kHz, neural vocoders affect 6-8 kHz. Instead of processing all frequencies equally, SB-FAC splits the spectrum into 4 sub-bands and learns to focus on the most informative bands per sample.

```
Audio Input → RawBoost Augmentation → Mel Spectrogram (80 bins)
  → Split into 4 Sub-Bands (20 bins each)
    ├── Band 0 (0-2 kHz)  → Conformer Branch → 
    ├── Band 1 (2-4 kHz)  → Conformer Branch → Sub-Band Attention → Classification
    ├── Band 2 (4-6 kHz)  → Conformer Branch →   (Real / Fake)
    └── Band 3 (6-8 kHz)  → Conformer Branch → 
```

## Results

### Development Set (Same attack types as training)

| Metric | Score |
|--------|-------|
| Equal Error Rate (EER) | 1.61% |
| Accuracy | 97.2% |
| Bonafide Precision | 1.00 |
| Spoof Recall | 1.00 |

### Evaluation Set (13 completely unseen attack types)

| Metric | Score |
|--------|-------|
| EER | 11.23% |
| Accuracy | 88.8% |
| Attacks detected with EER < 6% | 10 / 13 |

## Architecture Details

- **4 parallel Conformer branches** (3 layers each, d_model=128)
- **Sigmoid-based sub-band attention** with KL-divergence diversity loss
- **Attentive temporal pooling** with padding masks
- **RawBoost augmentation** (LnL + ISD + SSI) at 70% probability during training
- **~3.5M parameters**, ~14 MB model size

## Tech Stack

- **Framework:** PyTorch, torchaudio
- **Dataset:** ASVspoof 2019 Logical Access (LA)
- **Deployment:** Gradio web application
- **Hardware:** NVIDIA RTX 4060 (8 GB VRAM)
- **Training Time:** ~2.5 hours

## Web Application

The trained model is deployed as a **Gradio** web app that accepts audio via file upload or microphone recording and provides:
- Binary classification (Real / Fake) with confidence score
- Mel spectrogram visualization
- Sub-band attention weight chart showing which frequency bands influenced the prediction

## Project Structure

```
├── deepfake_sbfac_conformer.ipynb     # Full pipeline: model, training, evaluation, Gradio app
├── SB_FAC_Project_Pipeline.pdf        # Technical documentation (13 pages)
└── README.md
```

The notebook contains the complete end-to-end pipeline: data loading, RawBoost augmentation, SB-FAC model architecture, training loop, evaluation metrics, per-attack analysis, and Gradio deployment code.

> **Note:** Model weights (.pth) are not included due to size. Dataset available at: https://www.asvspoof.org/

## Citation

```
Vejeya Prasad C. "Deepfake Audio Detection Using Sub-Band Frequency 
Attention Conformer." SASTRA Deemed University, 2025.
```

## License

This project is part of academic coursework at SASTRA Deemed University, Thanjavur, India.
