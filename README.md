# 🌀 FLOW-ITS: Global-Local Flow Transport for Irregular Time Series Generation

This project provides a training pipeline for **complex-valued diffusion models** operating in the wavelet domain, designed to reconstruct and generate complex-valued (real, imaginary) data.  
The `main_mult.py` script integrates multiple loss functions including affine-invariant, phase, frequency-aware, and perceptual components. 

**README Update : 2026.02.26**
(There may be differences between the source code and the descriptions, as the code is continuously being updated.)

---

## Overview
Many recent diffusion approaches generate time series via **time–frequency spectrograms**, but often treat spectrograms as static images, which can miss continuous temporal dynamics and subtle variations.
FLOW-ITS addresses this by:
- Partitioning the input time series and transforming it into **multiple spectrogram frames**.
- Modeling temporal transitions between consecutive frames through **continuous latent transport** guided by a learned velocity field.

**Key ideas**
- **Ship (global regularity):** a shared latent trajectory across frames
- **Containers (local irregularity):** frame-wise latent components capturing local deviations/irregular dynamics
- **Bidirectional flow learning:** forward + reverse integration with **cycle-consistency** for temporal coherence and self-correction

## 🧩 Requirements

### 1. Environment
- Python ≥ 3.9  
- CUDA ≥ 11.3  
- PyTorch ≥ 1.12  
- TensorBoard ≥ 2.0  

### 2. Installation

```bash
# Clone repository
git clone https://github.com/<your-repo>/wavelet-diffusion.git
cd wavelet-diffusion

# Create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Both `waveform` and `spectrogram` should contain `.npy` files representing the **1d waveform** and **2d spectrogram** parts of the data, respectively.

Example structure:
```
data/
 ├── waveform/
 │    └── (data_name)  
 │          ├── 00001.npy
 │          ├── 00002.npy
 │          └── ...
 └── spectrogram/
 │    └── (data_name)  
 │          └── mag
 │          └── re
 │          └── im
 │              ├── 00001.npy
 │              ├── 00002.npy
 │              └── ...
```

- `waveform/`: raw waveform time-series saved as `.npy`.
- `spectrogram/`: spectrogram representations saved as `.npy`, split into:
  - `mag/`: **magnitude** of the spectrogram
  - `re/`: **real part** of the complex spectrogram
  - `im/`: **imaginary part** of the complex spectrogram

---

## ⚙️ Preprocessing (Optional)

After running the scripts in the `preprocess/` folder, generate spectrogram data from waveforms using:

```bash
python wavelet_data_gen.py --wave_dir <waveform_path> --output_dir <spectrogram_path>
```

---

## 🚀 Training

Basic training example:

```bash
python train_decoder.py --wave_dir <waveform_path> --feat_root <spectrogram_path> --ckpt_dir <checkpoint_path> --stage1_epochs 30  --stage2_epochs 200
```

---

## 🎨 Sampling (Inference)

```bash
python decoder_from_samples.py --out_dir <output_path> --ckpt <checkpoint_path>
```
