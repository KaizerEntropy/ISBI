# WRATTS-Net: A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Sugeno Fusion Network for Nuclei and Instance Segmentation

**Status:** submitted to the IEEE International Symposium on Biomedical Imaging (ISBI), under review.

**Authors:** Diptarko Bhattacharjee¹, Mukut Chatterjee¹, Sergei Romanov³, Dimitri Kaplun²,³ (corresponding author), Ram Sarkar¹

¹ Department of Computer Science & Engineering, Jadavpur University, Kolkata, India
² School of CS and Technology / School of AI, China University of Mining and Technology, Xuzhou, China
³ Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", St. Petersburg, Russia

**Contact:** Dimitri Kaplun, dikaplun@etu.ru

This repository is the supplementary material for the paper. Because of the 4-page limit, derivations, pseudocode, dataset splits and hyperparameters are documented here.

---

## Contents

1. [Overview](#1-overview)
2. [Method](#2-method)
3. [Results](#3-results)
4. [Datasets and evaluation protocol](#4-datasets-and-evaluation-protocol)
5. [Training and inference details](#5-training-and-inference-details)
6. [Repository contents](#6-repository-contents)
7. [Limitations and notes on interpreting the results](#7-limitations-and-notes-on-interpreting-the-results)
8. [Citation, license and acknowledgements](#8-citation-license-and-acknowledgements)

---

## 1. Overview

Nuclei segmentation in H&E histopathology images is hard because of indistinct margins, stain variability, crowded clusters and the edge blurring caused by pooling. Semantic segmentation needs pixel-accurate boundaries; instance segmentation additionally needs touching nuclei to be separated.

WRATTS-Net addresses both with a dual-branch network:

- **Wavelet branch (LLWT).** A Learnable Lifting Wavelet Transform downsamples with a polyphase split, learnable directional predict/update steps, and passes the detail sub-bands to a symmetric inverse-lifting decoder.
- **ResNet branch.** A ResUNet with learnable Generalized Mean (GeM) pooling, whose skip connections are regularised by a feature alignment loss (L_feat).
- **Bi-directional cross-attention (BiCA).** Channel-wise cross-attention exchanges information between the two branches.
- **Learnable Sugeno fuzzy integral.** Combines the two branches' foreground probabilities using two learned fuzzy densities (g₁, g₂).
- **Multi-task objective.** Segmentation, feature alignment, horizontal/vertical distance maps (HoVer) and a Sobel boundary-consistency term.
- **Instance decoding.** Marker-controlled watershed on the gradient of the predicted distance maps.

Summary of the results (Section 3): WRATTS-Net obtains the highest Dice and AJI on MoNuSeg, TNBC and CPM-15, and the highest PQ on MoNuSeg and CPM-15, among the compared methods. On TNBC, HoVer-Net has the higher PQ in our comparison table (76.50 vs. 69.69).

---

## 2. Method

![WRATTS-Net architecture](assets/model_architecture.png)

*Figure 1. Schematic of the dual-branch WRATTS-Net: LLWT encoder/decoder, ResUNet branch, bi-directional cross-attention, Sugeno fuzzy integral fusion and multi-task heads. A higher-resolution version is in [`Detailed_Model_Diagram.pdf`](Detailed_Model_Diagram.pdf).*

### 2.1 Learnable Lifting Wavelet Transform (LLWT) branch

The wavelet branch has four encoder stages with 3, 3, 4 and 3 residual blocks (13 in total) and 64, 128, 256 and 512 channels. Each stage ends with an LLWT downsampling step.

**Polyphase split.** A feature map X is split into four sub-grids, which is a lossless rearrangement of its samples:

x₀₀ = X(2i, 2j) (reference), x₀₁ = X(2i, 2j+1) (horizontal), x₁₀ = X(2i+1, 2j) (vertical), x₁₁ = X(2i+1, 2j+1) (diagonal).

**Predict step.** Three direction-specific depthwise-convolution predictors P_h, P_v, P_d predict the three odd sub-grids from x₀₀. A squeeze-and-excitation-style calibration S(x, f) = f ⊙ σ(MLP(GAP(x))), with x = x₀₀, rescales each prediction channel-wise:

$$d_h = x_{01} - S_h\big(x_{00}, P_h(x_{00})\big)$$
$$d_v = x_{10} - S_v\big(x_{00}, P_v(x_{00})\big)$$
$$d_d = x_{11} - S_d\big(x_{00}, P_d(x_{00})\big)$$

**Update step.** Depthwise-convolution operators U_h, U_v, U_d and learnable scalars k_h, k_v, k_d fold the details back into the coarse approximation:

$$c_l = x_{00} + k_h U_h(d_h) + k_v U_v(d_v) + k_d U_d(d_d)$$

The coarse approximation and the three detail bands are concatenated and passed on, and the concatenated detail sub-bands bypass the deep encoder and go directly to the decoder.

**Inverse LLWT (decoder).** The decoder inverts the update and predict steps using the preserved detail sub-bands, reconstructing the polyphase grids at the finer scale:

$$x_{00} = c_l - \big(l_h U'_h(d_h) + l_v U'_v(d_v) + l_d U'_d(d_d)\big),\qquad x_{01} = d_h + P'_h(x_{00}),\; x_{10} = d_v + P'_v(x_{00}),\; x_{11} = d_d + P'_d(x_{00})$$

and interleaves x₀₀, x₀₁, x₁₀, x₁₁ onto the full-resolution grid. Given the detail sub-bands, each lifting step is exactly invertible for any choice of predictor and update operators. The forward path additionally applies a 1×1 convolution with batch normalisation to the concatenated sub-bands, so the encoder as a whole is a learned (not exactly invertible) downsampling operator.

### 2.2 ResUNet branch

The ResUNet branch has four encoder stages with 3, 3, 4 and 3 residual blocks. Spatial downsampling uses Generalized Mean (GeM) pooling over 2×2 windows with a learnable exponent p, which interpolates between average pooling (p = 1) and max pooling (p → ∞):

$$f_{\mathrm{GeM}}(\mathbf{x}) = \Big(\tfrac{1}{|\Omega|}\sum_{i\in\Omega}\max(x_i,\epsilon)^{p}\Big)^{1/p},\qquad \epsilon = 10^{-6}$$

The decoder upsamples with 2×2 transposed convolutions and concatenates the result with the encoder skip features. To reduce semantic drift across skip connections, each upsampled decoder feature map is aligned to its encoder counterpart with the feature alignment loss L_feat (Section 2.5).

### 2.3 ASPP bottleneck

Each branch has an ASPP bottleneck operating on 1024-channel features, with a 1×1 convolution, global average pooling and three dilated branches with rates {1, 4, 7}. The dilated branches use depthwise-separable convolutions, and the output is regularised with 2D dropout (rate 0.1).

### 2.4 Bi-directional cross-attention (BiCA)

BiCA connects the two branches at nine junctions: the four encoder stages, the bottleneck and the four decoder stages. To avoid the O((HW)²) cost of spatial attention, attention is computed over channels. For features x_w, x_r ∈ ℝ^{C×H×W} flattened to N = HW, queries, keys and values come from 1×1 convolutions with batch normalisation:

$$\hat{x}_w = x_w + W_{out}\Big(\mathrm{Softmax}\Big(\tfrac{Q_w K_r}{\sqrt{N}}\Big) V_r\Big),\qquad \hat{x}_r = x_r + W_{out}\Big(\mathrm{Softmax}\Big(\tfrac{Q_r K_w}{\sqrt{N}}\Big) V_w\Big)$$

with Q, V ∈ ℝ^{C×N}, K ∈ ℝ^{N×C}, so the affinity matrix is C × C. The output projection W_out uses a zero-initialised batch normalisation, so each BiCA module starts as an identity mapping, which stabilises early training.

### 2.5 Multi-task heads, Sugeno fusion and loss

**Heads.** The main decoder and two auxiliary decoder stages (D3, D2) produce segmentation logits through 1×1 convolutions. An auxiliary HoVer head predicts horizontal and vertical distance-to-centroid maps in [−1, 1] by fusing decoder features with upsampled wavelet detail sub-bands.

**Learnable Sugeno fuzzy integral.** The branch probabilities p_w and p_r are combined pixel-wise with two singleton fuzzy densities g₁, g₂ ∈ [0, 1], each parameterised by a sigmoid of a learned logit (initialised at 0.5):

$$p_{\mathrm{fused}} = \max\Big(\min\big(\max(p_w,p_r),\, g_{\max}\big),\; \min(p_w,p_r)\Big),\qquad g_{\max} = \begin{cases} g_1 & p_w \ge p_r \\ g_2 & \text{otherwise}\end{cases}$$

This is the closed form of the Sugeno integral for two sources. Equivalently, p_fused is g_max clipped to the interval [min(p_w, p_r), max(p_w, p_r)]: where both branches are above g_max the fused value is the smaller of the two, where both are below it is the larger, and where g_max lies between them the fused value equals g_max. The learned densities therefore act as confidence thresholds on the two branches. The fused probability is converted back to logits for the loss.

**Composite loss.**

$$\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_f \mathcal{L}_{feat} + \lambda_{hv}\mathcal{L}_{hv} + \lambda_b \mathcal{L}_{boundary},\qquad \lambda_f = 0.1,\; \lambda_{hv} = 1.0,\; \lambda_b = 0.15$$

- **L_seg** combines Focal Tversky (α = 0.7, β = 0.3, γ = 0.75) and binary cross-entropy with logits, applied to the fused output and to the two auxiliary outputs (weights 1.0, 0.4 and 0.2). The BCE positive-class weight is derived from the training foreground fraction and capped at 15.
- **L_hv** blends a global mean squared error on the distance maps with a foreground-only mean squared gradient error (Sobel gradients).
- **L_boundary** penalises the L1 difference between Sobel gradient magnitudes of the prediction and the target mask:

$$\mathcal{L}_{boundary} = \big\|\, |\nabla_S(\mathrm{pred})| - |\nabla_S(\mathrm{target})| \,\big\|_1$$

- **L_feat** is the channel-normalised MSE between each transposed-convolution-upsampled decoder feature u_l and its encoder skip feature s_l, summed over the four ResNet stages. Its minimiser also maximises their cosine similarity:

$$\mathcal{L}_{feat} = \sum_{l=1}^{4} \Big\| \tfrac{u_l}{\|u_l\|_2} - \tfrac{s_l}{\|s_l\|_2} \Big\|_2^2$$

### 2.6 Inference and post-processing

Full-image probability and HoVer maps are reconstructed from 256 × 256 patches with stride 192, blended with a separable raised-cosine (Hann) window whose weights are floored at 0.1. Instances are then obtained with marker-controlled watershed:

1. Compute the Sobel gradient magnitude of the horizontal and vertical distance maps, take the pixel-wise maximum of the two, and min-max normalise it.
2. Take as seeds the foreground pixels whose normalised gradient is below τ_seed, erode them with a disk of radius r_erode, and label the connected components.
3. Flood the gradient surface from the seeds inside the predicted foreground mask.
4. Discard instances smaller than 10 pixels.

If no seed survives, the connected components of the foreground mask are used.

---

## 3. Results

All numbers are percentages and are taken from the paper (Tables 1–3). Dice measures pixel-level agreement; AJI and PQ measure instance-level quality (Section 4.3).

### 3.1 Comparison with other methods (paper Table 3)

| Method | MoNuSeg Dice | AJI | PQ | TNBC Dice | AJI | PQ | CPM-15 Dice | AJI | PQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HoVer-Net | 79.60 | 59.10 | 60.80 | 83.50 | 67.80 | **76.50** | 85.74 | 64.65 | 60.85 |
| BoNuS | – | – | – | 76.73 | 60.73 | – | 79.99 | 54.54 | – |
| Swin-UNet | 80.12 | 53.12 | 54.87 | 84.71 | 61.46 | 63.12 | 86.92 | 65.79 | 61.28 |
| DELTASEG | 81.42 | – | – | 82.95 | – | – | – | – | – |
| CellViT-256 | 79.77 | 55.14 | 53.50 | 81.49 | 59.34 | 57.19 | 86.02 | 65.29 | 66.30 |
| U²-Net | 80.92 | 56.72 | 56.02 | 85.92 | 62.01 | 62.58 | 87.78 | 67.34 | 62.39 |
| **WRATTS-Net (ours)** | **82.46** | **62.44** | **61.49** | **88.00** | **70.23** | 69.69 | **90.02** | **70.82** | **67.40** |

WRATTS-Net is best on eight of the nine metric columns; the exception is TNBC PQ, where the HoVer-Net entry is higher. A dash means the metric was not available for that method.

### 3.2 Qualitative pipeline

![Qualitative pipeline](assets/qualitative_pipeline.png)

*Figure 2. Prediction pipeline: (a) H&E input, (b) fused foreground probability p_fused, (c) HoVer distance maps, (d) marker-controlled watershed instances.*

| (a) H&E input | (b) Fused mask | (c) HoVer maps | (d) Instances |
|:---:|:---:|:---:|:---:|
| ![](assets/input_image.png) | ![](assets/segmentation_mask.png) | ![](assets/hover_map.png) | ![](assets/instance_segmentation.png) |

### 3.3 Component ablation (paper Table 1)

| Group | Configuration | MoNuSeg Dice | AJI | PQ | TNBC Dice | AJI | PQ | CPM-15 Dice | AJI | PQ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResUNet component | EfficientNet block | 80.66 | 57.91 | 58.87 | 86.10 | 67.18 | 62.95 | 87.89 | 67.61 | 63.45 |
| | VGG block | 80.59 | 57.27 | 56.11 | 86.78 | 62.95 | 61.73 | 87.38 | 67.71 | 63.14 |
| | DenseNet block | 80.90 | 58.82 | 58.27 | 86.77 | 67.98 | 65.30 | 88.98 | 66.88 | 61.71 |
| | MobileNet block | 80.71 | 58.87 | 59.00 | 85.98 | 67.91 | 64.95 | 87.93 | 67.81 | 63.19 |
| | ResUNet block | 80.99 | 59.04 | 58.19 | 87.08 | 68.62 | 65.69 | 89.46 | 70.38 | 64.94 |
| Wavelet component | Haar | 80.52 | 60.95 | 58.96 | 86.09 | 61.36 | 61.97 | 88.30 | 67.58 | 65.81 |
| | Weighted | 80.42 | 59.25 | 59.13 | 86.55 | 60.05 | 61.48 | 88.35 | 64.94 | 63.65 |
| | Single MLP | 80.87 | 57.02 | 57.65 | 86.97 | 63.52 | 62.72 | 88.24 | 65.77 | 63.21 |
| | 3 MLPs | 81.11 | 58.81 | 58.50 | 86.82 | 65.01 | 65.51 | 89.14 | 65.29 | 65.02 |
| | 3 Weighted MLPs | 80.95 | 60.36 | 59.92 | 87.23 | 67.18 | 66.87 | 89.11 | 68.58 | 65.71 |
| Fusion strategy | Concatenation | 80.78 | 59.21 | 58.93 | 84.62 | 62.01 | 62.91 | 85.70 | 67.45 | 64.34 |
| | Majority voting | 80.41 | 58.75 | 57.21 | 85.45 | 61.97 | 63.02 | 87.34 | 67.62 | 62.03 |
| | Gated | 80.29 | 59.34 | 58.83 | 86.78 | 61.82 | 63.59 | 88.32 | 67.91 | 63.00 |
| | Sugeno fusion | 81.56 | 60.12 | 59.78 | 87.01 | 63.00 | 64.72 | 88.98 | 69.31 | 63.49 |
| | **BiCA + Sugeno fusion (ours, complete model)** | **82.46** | **62.44** | **61.49** | **88.00** | **70.23** | **69.69** | **90.02** | **70.82** | **67.40** |

Reading the table:

- The ResUNet block is the best backbone variant on eight of the nine metrics (MoNuSeg PQ is the exception).
- Within the wavelet group, the 3 Weighted MLPs configuration is better than fixed Haar on seven of nine metrics (MoNuSeg AJI and CPM-15 PQ are the exceptions). Differences among wavelet variants are small on Dice and larger and less consistent on AJI.
- The complete model (BiCA + learnable Sugeno fusion) is best on all nine metrics among the fusion variants. Relative to the "Sugeno fusion" row it adds +0.90 Dice, +2.32 AJI and +1.71 PQ on MoNuSeg.

### 3.4 Loss ablation (paper Table 2)

| L_seg | L_feat | L_hv | L_bnd | MoNuSeg Dice | AJI | PQ | TNBC Dice | AJI | PQ | CPM-15 Dice | AJI | PQ |
|:-:|:-:|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ✓ | – | – | – | 77.65 | 53.45 | 53.15 | 84.15 | 61.40 | 61.15 | 85.30 | 61.80 | 59.35 |
| ✓ | ✓ | – | – | 79.20 | 55.70 | 55.15 | 85.42 | 63.60 | 63.10 | 86.95 | 64.25 | 61.05 |
| ✓ | ✓ | ✓ | – | 81.15 | 60.70 | 59.75 | 87.10 | 68.60 | 67.85 | 89.15 | 69.10 | 65.25 |
| ✓ | ✓ | ✓ | ✓ | **82.46** | **62.44** | **61.49** | **88.00** | **70.23** | **69.69** | **90.02** | **70.82** | **67.40** |

AJI gains from each added term:

| Added term | MoNuSeg | TNBC | CPM-15 |
|---|---:|---:|---:|
| + L_feat | +2.25 | +2.20 | +2.45 |
| + L_hv | +5.00 | +5.00 | +4.85 |
| + L_bnd | +1.74 | +1.63 | +1.72 |

The distance-regression term L_hv gives the largest instance-level gain (about 5 AJI points); L_feat and L_bnd give smaller, consistent gains.

---

## 4. Datasets and evaluation protocol

### 4.1 Datasets

| Dataset | Train | Validation | Test | Total | Image size |
|---|---:|---:|---:|---:|---|
| MoNuSeg | 29 | 8 | 14 | 51 | 1000 × 1000 |
| TNBC | 35 | 10 | 5 | 50 | 512 × 512 |
| CPM-15 | 10 | 3 | 2 | 15 | 400 × 400 to 600 × 600 |

The official splits of the datasets are used for all three datasets. The same table is in [`Dataset_info.md`](Dataset_info.md). Validation images are used only for checkpoint selection, foreground-threshold selection and watershed parameter selection; test images are used once, for the final evaluation. A separate model is trained and evaluated for each dataset.

### 4.2 Model selection

- **Checkpoint:** the EMA weights with the best validation Dice.
- **Foreground threshold:** selected on the validation images (maximising Dice).
- **Watershed parameters:** the full grid τ_seed ∈ {0.3, 0.4, 0.5, 0.6} × r_erode ∈ {0, 1, 2, 3} (16 combinations) is evaluated on full-resolution validation images, and the combination with the highest mean validation AJI is kept.

### 4.3 Metrics

- **Dice** is computed on foreground pixels pooled over all test images.
- **AJI** (Kumar et al., 2017) and **PQ** are computed per test image on the full-resolution stitched prediction and averaged over test images. For PQ, a prediction and a ground-truth nucleus match when IoU > 0.5; PQ = SQ × RQ, with SQ the mean IoU of matched pairs and RQ = TP / (TP + ½FP + ½FN).

---

## 5. Training and inference details

| Setting | Value |
|---|---|
| Framework / hardware | PyTorch with automatic mixed precision, NVIDIA T4 GPUs |
| Training crops | 256 × 256 native-resolution crops, foreground-biased (best of several random candidates); one crop per training image per epoch |
| Epochs / batch size | 500 / 4 |
| Optimiser | AdamW, learning rate 3 × 10⁻⁴, weight decay 1 × 10⁻³ |
| Schedule | One-cycle schedule with cosine annealing (30% warm-up) |
| Gradient clipping | Maximum norm 1.0 |
| EMA | Decay 0.999 with warm-up; the EMA weights are used for validation and testing |
| Initialisation | All weights randomly initialised (Kaiming initialisation); no pretrained weights are used, i.e. both branches are trained from scratch |
| Loss weights | λ_f = 0.1, λ_hv = 1.0, λ_b = 0.15; auxiliary segmentation weights 0.4 (D3) and 0.2 (D2) |
| Random seed | 42 |
| Inference | 256 × 256 patches, stride 192, raised-cosine blending |
| Watershed | Seeds from low-gradient foreground, disk erosion, minimum instance area 10 pixels |

**Augmentation.** Random horizontal and vertical flips, random 90° rotations, small affine transforms (rotation up to ±15°, shear up to ±5°, nearest-neighbour interpolation on instance labels), colour jitter (hue ±0.05, saturation 0.8–1.2, brightness and contrast 0.75–1.25), Gaussian blur, additive Gaussian noise and random erasing.

**Note on initialisation.** The conclusion of the paper refers to the ResNet branch as "pre-trained". The reported models do not use pretrained weights: every module, including the ResNet branch, is randomly initialised and trained from scratch.

The full hyperparameter listing is in [`Hyperparameters_and_Optimization.md`](Hyperparameters_and_Optimization.md).

---

## 6. Repository contents

| File / folder | Description |
|---|---|
| `README.md` | This file |
| `Algorithm.pdf` | Pseudocode for the model, losses, training, inference, post-processing and metrics |
| `Comprehensive_Details.pdf` | Detailed derivations and module descriptions |
| `Detailed_Model_Diagram.pdf` | High-resolution architecture diagram |
| `Dataset_info.md` | Dataset split sizes |
| `Hyperparameters_and_Optimization.md` | Training hyperparameters and watershed search |
| `assets/` | Figures used in this README |

---

## 7. Limitations and notes on interpreting the results

- **Computational cost.** The model has approximately 133 million parameters (counted from the model definition). The dual-branch design with cross-attention at nine junctions adds computational overhead relative to a single-branch network.
- **Post-processing.** Instance separation relies on watershed post-processing, which has two tuned parameters (τ_seed, r_erode). Future work will explore lightweight attention and differentiable clustering to remove this heuristic step.
- **Small test sets.** The TNBC and CPM-15 test sets contain 5 and 2 images respectively (Section 4.1), so results on them have high variance and small differences should be read with caution.
- **Single run.** Reported numbers come from one training run per dataset (seed 42); no confidence intervals are reported.
- **Per-dataset training.** Each model is trained and tested on a single dataset. Cross-dataset generalisation was not evaluated.

---

## 8. Citation, license and acknowledgements

If you use this work, please cite the paper (under review):

```bibtex
@misc{bhattacharjee_wrattsnet,
  title  = {{WRATTS-Net}: A Dual-Branch Wavelet-ResNet Model with Cross-Attention and Sugeno Fusion Network for Nuclei and Instance Segmentation},
  author = {Bhattacharjee, Diptarko and Chatterjee, Mukut and Romanov, Sergei and Kaplun, Dimitri and Sarkar, Ram},
  note   = {Submitted to the IEEE International Symposium on Biomedical Imaging (ISBI); under review}
}
```

**License.** Distributed under the Apache License 2.0; see the `LICENSE` file.

**Acknowledgements.** Computational resources were provided by the Centre for Microprocessor Applications for Training Education and Research (CMATER), Department of Computer Science & Engineering, Jadavpur University, Kolkata, India. Part of this research was supported by the Intelligent Devices Institute, St. Petersburg Electrotechnical University "LETI", Russia.
