# Pixel-Precise Segmentation of Solar Filaments in GONG H-alpha Images

A high-resolution CNN ensemble with learned instance post-processing for the **Solar Filament Segmentation 2026** competition (IEEE BigData 2026 Cup, hosted on Kaggle).

**Authors:** Siddharth Krishna, Raghav Shreyas TC, Prithvi Vijay Karthik, Danush Velachi
**Affiliation:** Department of Computer Science and Engineering, Shiv Nadar University Chennai, India

| Metric | Result |
|---|---|
| Out-of-fold PQ (707 training images) | **0.436** |
| Public-leaderboard PQ | **0.38** |
| U-Net baseline (public LB) | 0.34 |
| DINOv2 + Mask2Former (public LB) | 0.30 |

> All leaderboard scores are **public-leaderboard** scores. Private-leaderboard results were not available at the time of writing.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Description](#2-dataset-description)
3. [Models and Pipeline Architecture](#3-models-and-pipeline-architecture)
4. [Implementation Details](#4-implementation-details)
5. [Results and Metrics](#5-results-and-metrics)
6. [Limitations](#6-limitations)
7. [Future Work](#7-future-work)
8. [Repository Contents](#8-repository-contents)
9. [References](#9-references)

---

## 1. Problem Statement

Solar filaments are long, thin clouds of cool plasma held above the solar surface by magnetic fields along a polarity inversion line. In hydrogen-alpha (H-alpha, 656.3 nm) images they appear as dark, thread-like streaks on the solar disk. They matter for space weather because unstable filaments can erupt, and their side threads (barbs) indicate the chirality of the magnetic field. Long-term studies need a catalogue in which **each filament is a separate, precisely outlined object**, which is too laborious to produce by hand for decades of full-disk observations.

**Task.** Pixel-precise *instance segmentation* of filaments in full-disk GONG H-alpha images.

- **Input:** one 2048×2048 grey JPEG image from the Global Oscillation Network Group (GONG).
- **Output:** a set of binary masks `{M_k}` with `M_j ∩ M_k = ∅` for `j ≠ k`, one mask per filament, single object class, encoded as COCO run-length encoding (RLE) at 2048×2048 px.
- **Metric:** Panoptic Quality (PQ), computed against every annotator record of each image.

**Why it is hard**

1. **Thin structures.** Filaments are only a few pixels wide. For true width `w` and boundary error `e` per side, overlap is roughly `(w − 2e)/w`, so a 4 px filament with a 1 px error per side drops to the 0.5 IoU matching threshold.
2. **Extreme class imbalance.** About 99% of disk pixels are background.
3. **Annotator disagreement.** 41% of training images have two or three annotator records, and two annotators score only about 0.34 PQ against each other.
4. **Fragmentation.** Faint gaps split one filament into several dark "islands".

---

## 2. Dataset Description

The data are a subset of **MAGFiLO**, a manually annotated dataset of 10,244 filaments from 1,593 GONG observations (2011–2022) with polygons, spines, bounding boxes and chirality labels, reviewed by experts.

| Item | Value |
|---|---|
| Training images | 707 |
| Annotator records (training) | 1,154 (up to 3 independent annotators per image) |
| Test images | 180 (labels hidden) |
| Image size | 2048×2048 px, grey JPEG |
| Label format | Polygon sets in 2048-px coordinates |
| External data | None used |

### Targets

Each annotator record is rasterised into a label image. The semantic target `y` is the **mean of the records' binary masks**, so a pixel drawn by two of three annotators has `y = 0.67`. With binary cross-entropy, the optimum is `p* = y`, so the network learns the probability that an annotator marks the pixel. This matches a metric that compares predictions with every annotator.

### Splits

- **M1, M2, M4:** random 15% hold-out (seed 42): 601 training / 106 validation images.
- **M3 (final model):** the 707 images are sorted by GONG station and timestamp and dealt round-robin into **4 folds**, so each fold covers every station and the whole time range. Each training image receives an out-of-fold (OOF) prediction from the one fold model that never saw it.
- **Post-processing models:** cross-fitted with 5-fold GroupKFold grouped by image.

### Leakage precautions

- All records of one image stay on the same side of every split.
- Test labels are never seen.
- Input channels are computed without labels.
- All thresholds and post-processing models are fitted only on OOF predictions.
- Residual risk: frames of the same filament taken minutes apart can fall into different folds, which may make OOF scores optimistic.

---

## 3. Models and Pipeline Architecture

Five designs were compared under the same data split, scorer and GPU budget (2× NVIDIA T4 on Kaggle).

### Model progression

```
M1 Baseline (ResNet-34 U-Net, panoptic heads)      Val 0.399 | LB 0.34
        │
        ▼
Diagnosis: only 0.43% of filaments touch another;
PQ is lost in pixel precision and keep/merge decisions
        │
        ├──► M2 Gate test (frozen DINOv2 + CNN stem)         (no saved val. result)
        │         └──► M4 Approach B (DINOv2 + Mask2Former)  LB 0.30, 9.25 masks/img
        │                   └──► Probes (B masks + CNN)      LB 0.30 / 0.34 / 0.30
        │
        └──► M3 Approach A (EfficientNetV2-S ×4, learned merge + keep)
                  OOF 0.436 | LB 0.37
                  └──► M5 A2 refiner (dropped)               OOF 0.4356 vs 0.4364
                  │
                  ▼
        Final submission: Approach A, 4 flips, ≥1 mask per image   LB 0.38
```

| ID | Design | Notes |
|---|---|---|
| **M1** | ResNet-34 U-Net + Panoptic-DeepLab heads (semantic, centre heatmap, offsets) | Baseline. Loss = BCE + Dice + 200× centre MSE + L1 offset. 1024 px input, 512 px crops. |
| **M2** | Frozen DINOv2 ViT-B/14 with registers + trainable conv stem | Gate test of DINOv2 features on H-alpha. 88.9 M params, 2.3 M trainable. |
| **M4 (B)** | DINOv2 ViT-B/14 (last 4 blocks fine-tuned) + FPN + 6-layer Mask2Former decoder, 80 queries | 1120 px, Hungarian matching. 40.4 M trainable params. Over-split and did not converge. |
| **M3 (A)** | 4× EfficientNetV2-S U-Net + learned merge + learned keep rule | **Final model.** |
| **M5 (A2)** | Per-piece re-cut at six probability levels, GBM picks the cut | Dropped; slightly worse OOF PQ. |

### Final pipeline (M3)

```
Input: GONG H-alpha image (2048×2048, JPEG)
        │
        ▼
Disk fit + 3 label-free channels at 1536 px
  (flat-field, local contrast, dark ridge)
        │
        ▼
4 EfficientNetV2-S U-Nets (one per fold) × 4 flips
  → mean probability map (16 maps averaged)
        │
        ▼
Hysteresis 0.25 / 0.45 + connected components
  → candidate pieces
        │
        ▼
Learned merge (GBM, 21 pair features): join if score ≥ 0.40
        │
        ▼
Learned keep rule (GBM): keep piece if predicted matched IoU ≥ 0.24 (≈ PQ/2)
        │
        ▼
Up-sample to 2048 px, trim along probability map, ≥1 mask per image
        │
        ▼
Disjoint masks → COCO RLE → submission CSV
```

#### Step 1: Preprocessing

The image is resized to R = 1536 px. The disk is located by Otsu thresholding of a blurred image, and only pixels within 0.985 of the fitted radius are used. Three label-free channels are computed:

1. **Flat-field:** `F = I / med_r(I)` (intensity divided by the median at the same radius, 100 radial bins), used as `clip((F − 1)/0.25, ±3)`. Removes the radial brightness trend.
2. **Local contrast:** `z = (F − μ_16s(F)) / σ_32s(F)`, clipped at ±4, with Gaussian-weighted local mean and spread (`s = R/1024`).
3. **Dark-ridge response:** `max_σ σ² λ_max(H_σ F)` over σ ∈ {1.5, 3, 6}·s, the largest Hessian eigenvalue, high on thin dark lines; log-compressed and clipped to [0, 4].

#### Step 2: Network

U-Net with an EfficientNetV2-S encoder (ImageNet-21k pretraining, ImageNet-1k fine-tuning), decoder widths 256/128/64/32 plus a full-resolution block (24 channels) concatenated with the three input channels, ending in one logit per pixel. Centre and offset heads are dropped because touching filaments are rare. Loss: `BCE(p, y) + Dice(p, y)` on the soft target `y`.

#### Step 3: Training

Four fold models, each trained for 1.75 h on 768 px crops (70% centred on filament pixels to counter the 99% background). Augmentation: the eight flips/rotations of the square, per-channel gain in [0.85, 1.15], Gaussian noise. Optimiser: AdamW, exponential moving average of weights (decay 0.999), mixed precision, gradient clipping.

#### Step 4: Inference

Each of the four fold models is applied to four flipped copies (identity, horizontal, vertical, both). The 16 probability maps are averaged into `p(x)`.

#### Step 5: Instances

Hysteresis keeps 8-connected regions with `p ≥ 0.25` that contain a pixel with `p ≥ 0.45`. Each region is a candidate piece; pieces smaller than `40(R/1024)²` pixels are removed.

#### Step 6: Learned merge

For each pair of pieces within `48R/1024` px, 21 features describe the gap, widths, areas, probabilities, alignment with the joining line, and the probability and channel values along that line. A histogram gradient-boosting regressor predicts the fraction of annotator records in which both pieces lie mostly in the same filament. Pairs scoring ≥ 0.40 are joined with a bridge drawn only over background pixels (union-find), so masks stay disjoint.

#### Step 7: Learned keep rule

For each piece, 23 features (size, shape, probability statistics, channel values, contrast against a surrounding ring, distance to nearest piece, station) feed a second regressor that predicts the expected matched IoU `u`.

Writing `PQ = N/D` (N = sum of matched IoUs, D = TP + ½FP + ½FN), adding a piece raises D by ½ and N by `u` in expectation, so it helps only if

```
(N + u) / (D + 1/2) > N / D   ⟺   u > N / (2D) = PQ / 2
```

Pieces are kept when `û ≥ τ`. The PQ-maximising `τ = 0.24` is close to `PQ/2 ≈ 0.22`.

#### Step 8: Output

Labels are up-sampled to 2048 px by nearest neighbour and trimmed where the bilinearly up-sampled `p` falls below 0.25. An image never receives zero masks: if nothing survives, the largest component above half the maximum probability is submitted.

---

## 4. Implementation Details

| Item | Value |
|---|---|
| Language | Python 3.13 |
| Hardware | 2× NVIDIA T4 (16 GB each), Kaggle GPU image |
| Libraries | PyTorch, timm, OpenCV, SciPy, scikit-learn (`HistGradientBoostingRegressor`), pycocotools |
| Seed | 42 (plus fold index) |
| Parallelism | M1: `DataParallel`; M4: `DistributedDataParallel`; M3: two folds in parallel, one per GPU |

**Main training configuration**

| Setting | M1 baseline | M4 (B) | M3 (A) |
|---|---|---|---|
| Encoder | ResNet-34 | ViT-B/14 | EffNetV2-S |
| Input | grey | grey + 3 ch. | 3 channels |
| Resolution / crop (px) | 1024 / 512 | 1120 / full | 1536 / 768 |
| Batch | 16 | 2 per GPU | 8 per GPU |
| Learning rate | n/r | 2×10⁻⁴ | 3×10⁻⁴ |
| Training budget | 1.75 h | 3.0 h | 4 × 1.75 h |
| Steps | 4,696 | 6,300 | n/r |
| Outputs | sem.+ctr.+off. | 80 queries+sem. | 1 map |

*n/r: not recorded in the saved output.*

**Reproducibility note:** data-loader workers also mix in wall-clock time, so reruns are not bit-identical. Exact package versions are those of the Kaggle image at run time.

---

## 5. Results and Metrics

### Evaluation metric: Panoptic Quality

A prediction and a ground-truth filament *match* if IoU > 0.5 (matches are unique because masks are disjoint).

```
PQ = Σ_matches IoU / (TP + ½FP + ½FN)  =  SQ × DQ
```

- **SQ** (segmentation quality) = mean IoU of matched pairs
- **DQ** (detection quality) = `TP / D`, equal to the F1 score
- Precision = `TP / (TP + FP)`, Recall = `TP / (TP + FN)`

The scorer replica scores each annotator record separately against the same predicted masks (IoU via pycocotools) and pools counts and IoU sums over all records before computing PQ.

Three kinds of numbers are reported and **are not interchangeable**: validation PQ (M1, M2, M4; 106 held-out images), OOF PQ (M3, M5; all 707 images), and public-leaderboard PQ (hidden test labels).

### Main results

| Method | Eval. | PQ ↑ | SQ ↑ | DQ ↑ | LB PQ ↑ | Train (h, 2×T4) |
|---|---|---|---|---|---|---|
| M1 baseline | val | 0.399 | 0.662 | 0.602 | 0.34 | 1.75 |
| M2 frozen DINOv2 | val | pending | – | – | – | 1.75 |
| M4 Approach B | val | pending | – | – | 0.30 | 3.0 |
| **M3 Approach A** | OOF | **0.436** | 0.676 | 0.646 | 0.37 / **0.38**\* | 4 × 1.75 |
| M5 A2 refiner | OOF | 0.4356† | – | – | – | – |

\*2 flips: 0.37; then 4 flips and ≥1 mask/image: 0.38. †M3 in the same run: 0.4364. "Pending" = not available in the saved notebook output.

### Ablation-style comparisons

| Comparison | Setting | Result |
|---|---|---|
| M1 grouping | voting / components | 0.4003 / 0.4003 |
| M1 threshold | 0.3 / 0.4 / 0.5 | 0.400 / 0.398 / 0.392 |
| M1 min. score | 0.5 / 0.6 | 0.396 / 0.400 |
| M3 hysteresis (5 pairs) | best | 0.25 / 0.45 (lowest pair tried) |
| M3 thr. / +merge / +keep | OOF | pending |
| M3 keep threshold τ | tuned / Eq. (1) | 0.24 / 0.22 |
| M3 vs. M5 refiner | OOF | 0.4364 / 0.4356 |
| M5 IoU per piece | original / learned / oracle | 0.385 / 0.386 / 0.408 |
| M3 2→4 flips, ≥1 mask\* | public LB | 0.37 → 0.38 |
| M4 vs. M3 | masks per test image | 9.25 / 5.75 |
| B + CNN probes† | public LB | 0.30 / 0.34 / 0.30 |

\*Two changes made together. †Input file roles were swapped, so these probes do not test the intended combination.

### Error breakdown

| Model | Precision | Recall | TP | FP | FN | Masks / test image |
|---|---|---|---|---|---|---|
| M1 | 0.56 | 0.65 | 822 | 645 | 442 | 8.44 |
| M3 | 0.69 | 0.61 | 4,984 | 2,246 | 3,215 | 5.75 |

M3's OOF PQ by GONG station ranges from 0.427 to 0.452.

### Key findings

- **Touching filaments are not the bottleneck.** Only 0.43% of annotated filaments touch another; splitting annotators' own masks into connected components scores PQ 0.996.
- **Most of the gain is detection.** From M1 to M3, PQ rose by about 0.038. Of that, ΔSQ × DQ_M3 ≈ 0.009 and SQ_M1 × ΔDQ ≈ 0.029, so roughly three quarters of the gain came from better detection decisions (higher precision from the learned keep rule).
- **Why Approach B lost.** It over-split (9.25 vs 5.75 masks per image; a split filament costs one FN and two FPs) and had not converged (query mask Dice loss only fell from 0.92 to about 0.55). Likely causes, not isolated: quadratic attention cost limiting resolution to 1120 px, 14-pixel patches too coarse for filaments a few pixels wide, too little data for set-based training, and annotator disagreement on splits.
- **Why the refiner failed.** With perfect choices it would raise mean matched IoU per piece only from 0.385 to 0.408; the learned choice reached 0.386 (about 4% of that ceiling). Since FN (3,215) outnumber FP (2,246), the remaining error is missed filaments, which re-cutting cannot fix.
- **Learning keep/merge decisions against the metric paid off** more than end-to-end instance networks on this data.

---

## 6. Limitations

- The contribution of individual changes (input channels, encoder, ensembling, merge, keep rule) is **not isolated**; the three-step post-processing ablation was not saved and is pending.
- OOF scores may be **optimistic** because folds are not separated in time (near-duplicate frames).
- Both M1 and M3 lose about 0.06 from validation/OOF to the leaderboard (0.399→0.34; 0.436→0.37).
- Two grid searches ended at their edge (M1 minimum score 0.6; M3 lowest hysteresis pair), so better settings may exist.
- Each model was trained once; leaderboard differences of 0.01 are within plausible noise.
- All data come from one network of telescopes; transfer to other instruments is untested.
- Inter-annotator agreement of about 0.34 PQ shows the labels are ambiguous; this is not evidence that the model outperforms annotators, as the numbers are computed differently.
- No prediction overlays were saved, so qualitative figures are not included.

---

## 7. Future Work

1. Split folds by **time block**, rerun the three-step post-processing ablation, and use multiple seeds.
2. Use a **DINOv2 encoder inside a dense decoder**, blending its probability map with the CNN map to keep global context without query-based grouping.
3. Train at the **full 2048 px** resolution.

---

## 8. Repository Contents

The repository contains the notebooks and run instructions. **No competition data is included**; download it from the Kaggle competition page.

```
.
├── notebooks/        # M1, M2, M4, M3 (final), M5 notebooks
├── README.md
└── ...
```

[GITHUB-REPO](https://github.com/Dhanush7-tech/Deep-Learning-Lab-Project)

---

## 9. References

1. A. Ahmadzadeh et al., "A dataset of manually annotated filaments from H-alpha observations," *Scientific Data*, vol. 11, 2024, doi: 10.1038/s41597-024-03876-y.
2. A. Ahmadzadeh, D. J. Kempton, Q. Li, and A. A. Pevtsov, "Pixel-precise segmentation of solar filaments," IEEE BigData 2026 Cup, 2026. https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/
3. Kaggle, "Solar Filament Segmentation 2026," 2026. https://www.kaggle.com/competitions/filament-segmentation-2026
4. G. Zhu, G. Lin, D. Wang, S. Liu, and X. Yang, "Solar filament recognition based on deep learning," *Solar Physics*, vol. 294, no. 9, 2019.
5. D. Liu, W. Song, G. Lin, and H. Wang, "Solar filament segmentation based on improved U-Nets," *Solar Physics*, vol. 296, art. 176, 2021, doi: 10.1007/s11207-021-01920-3.
6. A. Diercke et al., "A universal method for solar filament detection from Hα observations using semi-supervised deep learning," *Astronomy & Astrophysics*, vol. 686, art. A213, 2024, doi: 10.1051/0004-6361/202348314.
7. J. R. Hu, Q. Hao, Z. Zheng, P. F. Chen, C. Li, and Y. Meng, "A modern ConvNet for solar filament detection," arXiv:2607.24525, 2026.
8. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.
9. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
10. A. Kirillov, K. He, R. Girshick, C. Rother, and P. Dollár, "Panoptic segmentation," in *Proc. IEEE/CVF CVPR*, 2019, pp. 9404–9413.
11. B. Cheng et al., "Panoptic-DeepLab: A simple, strong, and fast baseline for bottom-up panoptic segmentation," in *Proc. IEEE/CVF CVPR*, 2020, pp. 12475–12485.
12. B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, "Masked-attention mask transformer for universal image segmentation," in *Proc. IEEE/CVF CVPR*, 2022, pp. 1290–1299.
13. M. Oquab et al., "DINOv2: Learning robust visual features without supervision," *Trans. Mach. Learn. Res.*, 2024.
14. T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski, "Vision transformers need registers," in *Proc. ICLR*, 2024.
15. M. Tan and Q. V. Le, "EfficientNetV2: Smaller models and faster training," in *Proc. ICML*, 2021, pp. 10096–10106.
16. F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in *Proc. 3DV*, 2016, pp. 565–571.
17. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.
18. J. Canny, "A computational approach to edge detection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. PAMI-8, no. 6, pp. 679–698, 1986.
