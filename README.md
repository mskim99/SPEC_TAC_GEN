
# 🌀 Wavelet-Diffusion Training

This project provides a training pipeline for **complex-valued diffusion models** operating in the wavelet domain, designed to reconstruct and generate complex-valued (real, imaginary) data.  
The `main_mult.py` script integrates multiple loss functions including affine-invariant, phase, frequency-aware, and perceptual components. 

**README Update : 2025.11.05**
(There may be differences between the source code and the descriptions, as the code is continuously being updated.)

---

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

**Example requirements.txt**

```txt
torch>=1.12
torchvision
numpy
tensorboard
scipy
```

---

## 📂 Dataset Preparation

Both `real_dir` and `imag_dir` should contain `.npy` files representing the **real** and **imaginary** parts of the data, respectively.

Example structure:
```
data/
 ├── real/
 │    ├── 00001.npy
 │    ├── 00002.npy
 │    └── ...
 └── imag/
      ├── 00001.npy
      ├── 00002.npy
      └── ...
```

Each `.npy` file should contain a 2D array of shape `(H, W)`.

---

## ⚙️ Preprocessing (Optional)

If `--norm_type global_z` is used, you must compute **global mean and standard deviation** statistics first.  
You can generate a `stats.json` file using a helper script such as:

```bash
python compute_stats.py --real_dir data/real --imag_dir data/imag --output stats.json
```

---

## 🚀 Training

Basic training example:

```bash
python main_mult.py   --real_dir data/real   --imag_dir data/imag   --ckpt_dir ckpts_v6   --epochs 200   --batch_size 8   --lr 1e-4   --norm_type per_scale_z   --amp
```

---

## 🧠 Argument Description

| Argument | Default | Description |
|-----------|----------|-------------|
| `--real_dir` | **(required)** | Directory of `.npy` files for the real component |
| `--imag_dir` | **(required)** | Directory of `.npy` files for the imaginary component |
| `--ckpt_dir` | `ckpts_v6` | Directory to save checkpoints |
| `--epochs` | `200` | Number of training epochs |
| `--batch_size` | `8` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--norm_type` | `per_scale_z` | Normalization type (`per_scale_z`, `zscore`, `global_z`, `none`) |
| `--stats_file` | `None` | Path to stats.json (required if using `global_z`) |
| `--amp` | `False` | Use automatic mixed precision (AMP) |
| `--timesteps` | `1000` | Number of diffusion timesteps |
| `--base_ch` | `64` | Base channel size for UNet |
| `--ch_mult` | `[1,2,4,8]` | Channel multipliers for UNet |
| `--save_interval` | `20` | Save checkpoint interval (in epochs) |

---

## 🧮 Loss Function Weights

| Argument | Default | Description |
|-----------|----------|-------------|
| `--l_noise` | `1.0` | Noise prediction loss (L1) |
| `--l_ai` | `1.0` | Affine-invariant reconstruction loss |
| `--l_phase` | `0.5` | Phase difference loss |
| `--l_freq` | `0.5` | Frequency-aware loss (low-frequency weighted) |
| `--l_perc` | `0.3` | Wavelet perceptual loss |
| `--l_spec` | `0.3` | Spectral-domain magnitude loss |
| `--l_ms` | `0.2` | Multi-scale MSE loss |

---

## 🧾 Logging & Checkpoints

- **TensorBoard logs**: `ckpt_dir/tb/`
- **Model checkpoints**: `ckpt_dir/ckpt_epXXXX.pt`
- Real-time loss logging and TensorBoard support available:
  ```bash
  tensorboard --logdir ckpts_v6/tb
  ```

---

## 🧱 Example (with global normalization)

```bash
python main_mult.py   --real_dir data/real   --imag_dir data/imag   --norm_type global_z   --stats_file stats.json   --l_ai 1.0 --l_phase 0.5 --l_spec 0.3   --ckpt_dir ckpts_global   --amp
```

---

## 🎨 Sampling (Inference)

After training with `main_mult.py`, you can generate new complex-valued samples using **`sample_mult.py`**.  
This script supports both **DDPM** and **DDIM** sampling modes and saves real/imag outputs as `.npy` and `.png` files.

---

### 🔧 Basic Usage

```bash
python sample_mult.py   --ckpt ckpts_v6/ckpt_ep0200.pt   --out_dir samples_complex   --shape 1024 1024   --num_samples 4   --batch 1   --tensorboard
```

This command:
- Loads a trained checkpoint (`ckpt_ep0200.pt`)
- Generates **4 samples** of size **1024×1024**
- Saves outputs to `samples_complex/`
- Logs intermediate results to TensorBoard

---

### ⚙️ Key Arguments

| Argument | Default | Description |
|-----------|----------|-------------|
| `--ckpt` | **(required)** | Path to trained checkpoint (`.pt`) |
| `--out_dir` | `samples_complex` | Output directory for generated samples |
| `--shape` | `1024 1024` | Height and width of generated images |
| `--num_samples` | `4` | Total number of samples to generate |
| `--batch` | `1` | Batch size during sampling |
| `--save_every` | `0` | Save intermediate `x_t` every N steps (0 = off) |
| `--tb_every` | `0` | Log intermediate images to TensorBoard every N steps (0 = off) |
| `--tensorboard` | `False` | Enable TensorBoard logging |
| `--seed` | `None` | Random seed for reproducibility |
| `--noise_scale` | `1.0` | Extra noise temperature for reverse step |
| `--vmin`, `--vmax` | `None` | Min/max values for PNG normalization |
| `--ddim_steps` | `0` | Use DDIM sampler with N steps (0 = use DDPM) |
| `--eta` | `0.0` | DDIM stochasticity (0.0 = deterministic, 1.0 = stochastic) |

---

### 🧩 Sampling Modes

#### 🐌 DDPM (Default)
Classic denoising diffusion process with `T=1000` steps.

```bash
python sample_mult.py --ckpt ckpt.pt --out_dir samples_ddpm
```

#### 🚀 DDIM (Fast Deterministic Sampling)
Use fewer steps with optional noise control via `eta`.

```bash
python sample_mult.py   --ckpt ckpt.pt   --out_dir samples_ddim   --ddim_steps 50   --eta 0.0
```

---

### 🧠 TensorBoard Visualization

Enable TensorBoard logging with `--tensorboard` and visualize intermediate results:

```bash
tensorboard --logdir samples_complex/tb
```

- `real_tXXXX` / `imag_tXXXX`: intermediate diffusion steps  
- `final_real` / `final_imag`: final output at step 0  

---

### 🗂️ Output Structure

Example output directory:
```
samples_complex/
 ├── sample_000_real.npy
 ├── sample_000_imag.npy
 ├── sample_000_real.png
 ├── sample_000_imag.png
 ├── sample_001_real.png
 ├── ...
 └── tb/
      ├── events.out.tfevents...
      └── ...
```

Each sample contains:
- `*_real.npy` / `*_imag.npy`: raw numeric arrays  
- `*_real.png` / `*_imag.png`: normalized grayscale visualizations  

---

### ✨ Notes
- The script automatically restores normalization parameters (`norm_mu`, `norm_std`) stored in the checkpoint.
- Supports dynamic DDIM/DDPM switching for speed-quality tradeoff.
- Compatible with checkpoints produced by `main_mult.py`.

---

## 🧪 Evaluation

After generating samples with `sample_mult.py`, you can quantitatively evaluate the results using **`eval.py`**.  
It compares generated `.npy` files against ground-truth data and computes multiple metrics such as MSE, PSNR, SSIM, and spectral similarity.

---

### 🔧 Basic Usage

```bash
python eval.py   --gt_root data/gt_real   --gen_root samples_complex   --suffix _real.npy   --affine_fix
```

This will:
- Compare `.npy` files in `samples_complex/` (generated results) to ground truth in `data/gt_real/`
- Compute various similarity metrics
- Apply affine correction (`--affine_fix`) to match scale and bias if enabled

---

### ⚙️ Key Arguments

| Argument | Default | Description |
|-----------|----------|-------------|
| `--gt_root` | **(required)** | Directory containing ground-truth `.npy` files |
| `--gen_root` | **(required)** | Directory containing generated `.npy` files |
| `--ckpt` | `None` | Optional checkpoint path (used to restore normalization stats) |
| `--suffix` | `_real.npy` | File suffix for generated data (e.g., `_real.npy`, `_imag.npy`) |
| `--is_phase` | `False` | Whether to evaluate phase data (uses circular MSE) |
| `--affine_fix` | `False` | Apply real-valued affine correction between prediction and GT |
| `--complex_affine_fix` | `False` | Apply complex-valued affine correction (scale + rotation + bias) |

---

### 📊 Supported Metrics

| Metric | Description |
|---------|--------------|
| **MSE** | Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **PSNR** | Peak Signal-to-Noise Ratio |
| **SSIM** | Structural Similarity Index |
| **Cosine** | Cosine similarity between flattened arrays |
| **Correlation** | Pearson correlation coefficient |
| **Spectral Convergence** | Frobenius norm ratio between magnitude spectra |
| **Log-Spectral Distance** | Mean squared log difference between spectra |
| **Circular MSE** | (Phase mode) circular distance error |

---

### 🧩 Optional: Complex Affine Correction

To compensate for global amplitude and phase shifts between generated and ground-truth data,  
you can enable **complex affine correction** using:

```bash
python eval.py   --gt_root data/gt_real   --gen_root samples_complex   --complex_affine_fix
```

---

### 🗂️ Example Output

Console output example:

```
[INFO] Evaluating GT (100 files) vs GEN (100 files matching '_real.npy')
=== Evaluation Results ===
MSE: 0.0015
PSNR: 34.67
SSIM: 0.9123
Cosine: 0.9814
Correlation: 0.9849
Spectral Convergence: 0.0481
Log-Spectral Distance: 0.0072
```

---

### ✨ Notes
- Automatically loads normalization stats (`norm_mu`, `norm_std`) from checkpoint if available.
- Affine or complex-affine corrections can significantly improve evaluation robustness.
- Compatible with all `.npy` outputs generated from `sample_mult.py`.

---