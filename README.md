# Collective Intelligence for Edge ECG Diagnosis

> A complete MLOps + Edge-IoT pipeline that trains a 1-D CNN arrhythmia classifier,
> compresses it through **8 optimization techniques**, deploys the 3 tier champions on
> **Docker-simulated edge nodes (500 MB / 1 GB / 2 GB)**, fuses their predictions via a
> **weighted-voting collective intelligence layer**, and supervises everything through
> **ThingsBoard** over MQTT.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white">
  <img alt="MQTT" src="https://img.shields.io/badge/MQTT-Paho-660066?logo=eclipsemosquitto&logoColor=white">
  <img alt="ThingsBoard" src="https://img.shields.io/badge/IoT-ThingsBoard-FF5722">
  <img alt="License" src="https://img.shields.io/badge/License-Academic-blue">
</p>

**Module:** Systèmes Embarqués et Objets Connectés — Master Data Science, ENS Martil
**Instructor:** Prof. Said Ohamouddou
**Author:** Abdelhamid Merghad && Ayoub Aarab

📊 **[Full per-run results →](RESULTS.md)** · 🎯 **[Grading rubric mapping →](GRADING.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Headline Results](#headline-results)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Model — ECGNet1D](#model--ecgnet1d)
6. [Phase 1 — Baseline](#phase-1--baseline)
7. [Phase 2 — Eight Optimization Techniques](#phase-2--eight-optimization-techniques)
8. [Phase 3 — Deployment on Three VM Tiers](#phase-3--deployment-on-three-vm-tiers)
9. [Phase 4 — Champion Selection per Tier](#phase-4--champion-selection-per-tier)
10. [Phase 5 — Collective Intelligence](#phase-5--collective-intelligence)
11. [Phase 6 — ThingsBoard Supervision](#phase-6--thingsboard-supervision)
12. [Engineering Lessons](#engineering-lessons)
13. [Repository Structure](#repository-structure)
14. [Getting Started](#getting-started)
15. [Deliverables Checklist](#deliverables-checklist)
16. [References](#references)

---

## Overview

The project follows the 6-phase specification for the *Systèmes Embarqués et Objets
Connectés* module. A single high-accuracy ECG beat classifier (`ECGNet1D`, ~77 K
parameters) is progressively compressed, deployed, and orchestrated on three
resource-constrained edge tiers, then supervised live over MQTT in ThingsBoard.

| Phase | Objective | Status |
|-------|-----------|--------|
| **1 — Baseline**          | Train `ECGNet1D` on MIT-BIH, compare with literature | ✅ |
| **2 — Optimization**      | Apply 8 techniques (Q1–Q5, P1–P3) and benchmark | ✅ |
| **3 — Deployment**        | Run all 8 techniques inside 3 Docker VM tiers (3 × 8 matrix) | ✅ |
| **4 — Selection**         | Pick one champion per tier via weighted score | ✅ |
| **5 — Collective**        | Weighted voting + confidence gating + load balancing | ✅ |
| **6 — Supervision**       | MQTT telemetry and ThingsBoard dashboards | ✅ |

---

## Headline Results

| Metric | Value |
|---|---|
| Best baseline (ECGNet1D FP32)   | **97.43 % acc · 0.882 macro-F1 · 77,445 params · 0.317 MB** |
| Best accuracy across all 8 techniques | **98.21 % (P3 — Global Magnitude Pruning)** |
| Smallest deployable model       | **0.161 MB (P2 — Structured Pruning 30 %)** |
| Fastest technique on VM1 (500 MB) | **17.6 ms · 0.184 MB (Q5 — Mixed Precision)** |
| **Champion VM1** (500 MB, 1 vCPU) | **Q2 — Static PTQ** · 97.48 % · 367 ms |
| **Champion VM2** (1 GB, 2 vCPU)   | **P2 — Structured Pruning** · 97.73 % · 4.60 ms |
| **Champion VM3** (2 GB, 2 vCPU)   | **P3 — Global Magnitude Pruning** · 98.21 % · 0.27 ms |
| Collective layer — 10-specimen smoke test | **100 % consensus · 0 escalations** (see caveat in Phase 5) |

---

## System Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │       MIT-BIH Arrhythmia (CSV)              │
                     │       87,554 train / 21,892 test            │
                     └──────────────────────┬──────────────────────┘
                                            │
                               ┌────────────▼────────────┐
                               │  ECGNet1D  Baseline     │   Phase 1
                               │  (77,445 params · 97.4 %)│
                               └────────────┬────────────┘
                                            │
         ┌──────────────────────────────────▼──────────────────────────────────┐
         │   Eight optimization techniques (Q1–Q5 + P1–P3)                    │   Phase 2
         └──────────────────────────────────┬──────────────────────────────────┘
                                            │
      ┌──────────────┬───────────────────────┼───────────────────────┬──────────────┐
      ▼              ▼                       ▼                       ▼              ▼
┌──────────┐   ┌──────────┐           ┌──────────┐           3×8 Matrix      Phase 3
│   VM1    │   │   VM2    │           │   VM3    │           + OOM handling
│ 500 MB   │   │   1 GB   │           │   2 GB   │
│ 1 vCPU   │   │  2 vCPU  │           │  2 vCPU  │
└────┬─────┘   └────┬─────┘           └────┬─────┘
     │ Q2 pred      │ P2 pred              │ P3 pred
     └──────────────┼──────────────────────┘
                    ▼
        ┌──────────────────────────────┐
        │  Orchestrator + Aggregator    │  Phase 5 — Collective Intelligence
        │  Weighted voting (acc × conf) │
        │  Confidence gate (< 70 %)     │
        │  Load balancing (CPU 85/RAM 90)│
        └───────────────┬──────────────┘
                        │ MQTT telemetry (JSON per inference)
                        ▼
        ┌──────────────────────────────┐
        │  ThingsBoard CE (Docker)      │  Phase 6 — Supervision & Alerts
        │  Live predictions · CPU/RAM   │
        │  Alerts: latency > 300 ms,    │
        │          RAM > 90 %, acc < 80%│
        └──────────────────────────────┘
```

---

## Dataset

**MIT-BIH Arrhythmia** in pre-processed Kaggle CSV form — the single-lead heartbeats
are resampled to **187 points** and labelled into the 5 AAMI classes.

| Label | Class | Description | Test n |
|-------|-------|-------------|--------|
| 0 | N | Normal beat | 18,118 |
| 1 | S | Supraventricular ectopic beat | 556 |
| 2 | V | Ventricular ectopic beat | 1,448 |
| 3 | F | Fusion beat | 162 |
| 4 | Q | Unknown / unclassifiable beat | 1,608 |

**Split.** 87,554 train · 21,892 test (seed 42). Imbalance ratio ≈ **113 : 1**
(N vs. F). Dataset-selection scoring across five candidate datasets lives in
`results/eda/`.

| Candidate dataset | Score | Samples | Imbalance |
|---|---|---|---|
| **mitbih (selected)** | **35** | 87,554 | 113.06 |
| ptbxl                 | 23   | 21,837 | 1.01 |
| mitbih_database       | 20   | 109,494 | 112.87 |
| mit_bih_arrhythmia    | 19   | 48 | 1.00 |

---

## Model — ECGNet1D

`ECGNet1D` (`models/ecg_net.py`) is a **compact 1-D CNN (~77 K parameters)**
purpose-built for single-lead ECG beat classification on edge devices. It is the
final result of a three-iteration hardening against the 113 : 1 class imbalance.

The repository also defines the wrapper classes required for eager-mode optimization:

- `ECGNet1D_Narrow` — channel-reduced variant produced by **structured pruning**.
- `QuantizableClassifier`, `FP16Wrapper`, `ManualMixedPrecisionWrapper` — single-module
  wrappers so that `torch.load` is reliable across heterogeneous deployment hosts.

---

## Phase 1 — Baseline

The classifier was iteratively hardened against the MIT-BIH class imbalance
(N ≈ 83 %, F ≈ 0.7 %). Each row corresponds to a JSON artifact in
`results/mitbih_csv/`.

### Baseline evolution

| Stage | Model | Params | Test acc | Macro-F1 | Key change |
|---|---|---|---|---|---|
| Naïve baseline   | ECGNet1D            | ~62 K   | **88.45 %** | 0.497 | F-class collapse (F1 = 0.00) |
| Weighted CNN     | ECGNet1D + class weights | ~62 K | 96.93 % | 0.856 | α-weighted loss + light aug |
| **Final baseline** | **ECGNet1D**      | **77,445** | **97.43 %** | **0.882** | FocalLoss γ = 2.0, calibrated heads |

> **Ablation warning.** An over-aggressive SMOTE + FocalLoss(γ = 2.5) +
> WeightedRandomSampler run collapsed to **16.4 % accuracy** by driving N-recall to
> zero (see `results/imbalance_optim/ANALYSIS.md`). The winning recipe keeps the
> *real* class distribution and re-weights conservatively.

### Per-class performance — final baseline

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| N (normal)     | 0.991 | 0.982 | **0.986** |
| S (supravent.) | 0.644 | 0.770 | **0.701** |
| V (ventric.)   | 0.937 | 0.959 | **0.948** |
| F (fusion)     | 0.754 | 0.815 | **0.783** |
| Q (unknown)    | 0.986 | 0.993 | **0.989** |

### Literature comparison

| Paper | Accuracy | Note |
|-------|----------|------|
| Kachuee et al. (2018)  | 93.4 % | 1-D CNN, same dataset (MIT-BIH) |
| Acharya et al. (2017)  | 94.2 % | 9-layer CNN, same dataset |
| Hannun et al. (2019)   | 97.0 % | 34-layer ResNet — **different dataset** (single-lead ambulatory ECG), not directly comparable |
| **This work (baseline)** | **97.43 %** | **77 K params, MIT-BIH** |

> On the two directly comparable papers, this baseline improves on the
> state-of-the-art 1-D CNN by **+4.0 pp (Kachuee)** and **+3.2 pp (Acharya)** with
> a model that fits in 0.32 MB.

---

## Phase 2 — Eight Optimization Techniques

Each technique is a self-contained `optimize.py` under `optimization/<ID>/`.
All raw inference numbers below are **single-sample CPU measurements with no
tier throttling** — the container-throttled figures are in Phase 3.

| ID | Technique | Family | Acc | Macro-F1 | Size (MB) | Inference (ms) |
|----|-----------|--------|-----|----------|-----------|----------------|
| B0 | Baseline FP32              | —            | 97.43 % | 0.882 | 0.317 | 0.0174 |
| P1 | Unstructured Pruning 50 %  | Pruning      | 98.01 % | 0.896 | 1.185 | 0.0180 |
| P2 | Structured Pruning 30 %    | Pruning      | 97.73 % | 0.879 | **0.161** | **0.0151** |
| P3 | Global Magnitude 40 %      | Pruning      | **98.21 %** | **0.902** | 1.196 | 0.0166 |
| Q1 | Dynamic Quantization       | Quantization | 97.45 % | 0.881 | 0.293 | 0.594 |
| Q2 | Static PTQ                 | Quantization | 97.48 % | 0.882 | 0.295 | 0.594 |
| Q3 | Quantization-Aware Training| Quantization | 97.69 % | 0.886 | 0.295 | 0.595 |
| Q4 | Weight-only FP16           | Quantization | 97.44 % | 0.882 | 0.168 | 0.267 |
| Q5 | Mixed Precision FP16+FP32  | Quantization | 97.44 % | 0.882 | 0.184 | 0.270 |

**Key insights.**
- **Global magnitude pruning *improves* accuracy** (+0.78 pp) — the 40 % mask acts
  as a regularizer on an already-converged model.
- **Structured pruning (P2)** delivers a **~2× size reduction** with only a 0.3 pp
  accuracy loss — the best accuracy-per-MB trade-off in the entire table.
- **QAT (Q3)** is the strongest quantized variant, recovering ~0.2 pp over static
  PTQ while staying at 0.295 MB.
- **Quantization slows *eager* CPU inference ~35×** vs FP32 (0.59 ms vs 0.017 ms) —
  PyTorch's x86 INT8 kernels are not competitive with its MKL FP32 path; the real
  benefit of quantization here is **size**, not latency.

---

## Phase 3 — Deployment on Three VM Tiers

Three Docker services with hard `cpus` / `memory` limits simulate the three hardware
profiles required by the specification.

| Service | CPUs | Memory | Role |
|---------|------|--------|------|
| `vm1` | 1.0 | 500 MB | Ultra-constrained edge node (sensor, low-end) |
| `vm2` | 2.0 | 1 GB   | Balanced edge node (IoT gateway) |
| `vm3` | 2.0 | 2 GB   | High-performance edge node (light edge server) |

> **Reading these numbers.** Latencies below are measured **inside the throttled
> Docker container**, so they are 2–3 orders of magnitude larger than the raw
> `torch` figures in Phase 2. Both are correct — they measure different things.
> Phase 2 answers *"what does the model cost?"*; Phase 3 answers *"what does the
> deployment cost?"* The gap is container scheduling, I/O, and — on VM1 — swap
> pressure.

### Tier champions at a glance

| Tier | Accuracy | Latency | Champion |
|------|----------|---------|----------|
| VM1  | 97.48 % | 367.1 ms | Q2 — Static PTQ |
| VM2  | 97.73 % | 4.60 ms  | P2 — Structured Pruning |
| VM3  | 98.21 % | 0.27 ms  | P3 — Global Magnitude Pruning |

📊 **[Full 3 × 8 matrix with every technique on every tier →](RESULTS.md#phase-3-full-matrix)**

### Key observations

- **Throughput scales ~60× between VM1 and VM3** on the same model
  (Q5: 17.6 ms → 0.30 ms) — CPU starvation, not model complexity, dominates
  the ultra-constrained tier.
- **FP16 (Q4 / Q5) is the fastest family on VM1** (17–21 ms), because halved
  memory traffic relieves the RAM-starved tier. It is *not* the most CPU-efficient.
- **Static PTQ (Q2) uses the least CPU of any variant on VM1** (64 %), which is
  why it wins the weighted selection for that tier despite mid-pack accuracy.
- **P3 is the highest-accuracy technique on every tier** — a hardware-independent
  ~98.21 % ceiling.
- **All 27 runs completed with no OOM** — even the largest model (P3, 1.196 MB)
  fits comfortably in the 500 MB tier.

> **Measurement caveat.** Docker's memory metric includes shared PyTorch runtime
> pages, so the resident set is reported at ~670 MB even inside the 500 MB
> container. CPU > 100 % indicates multi-threaded bursts beyond nominal core count.

---

## Phase 4 — Champion Selection per Tier

The specification defines an explicit weighted-score rule per tier. The table below
shows which criterion each tier prioritises and how the selected champion was chosen.

| Tier | Weighting (from spec) | Champion | Why |
|------|----------------------|----------|-----|
| **VM1** — 500 MB, 1 vCPU | **RAM 0.40 · CPU 0.40 · Acc 0.20** | **Q2 — Static PTQ** | Lowest CPU of any technique (64.1 % → 0.40 weight); smallest RAM footprint at 0.295 MB; accuracy (97.48 %) is competitive within the 0.20 weight. P1/P3 would blow the CPU budget; Q4/Q5 would exceed the CPU envelope despite being fastest. |
| **VM2** — 1 GB, 2 vCPU | **RAM 0.30 · Speed 0.30 · Acc 0.40** | **P2 — Structured Pruning 30 %** | Smallest model in the entire sweep (0.161 MB → 0.30 weight); second-fastest non-FP16 latency (4.60 ms → 0.30 weight); 97.73 % accuracy is only 0.48 pp behind P3 (0.40 weight). |
| **VM3** — 2 GB, 2 vCPU | **Acc 0.60 · Speed 0.25 · RAM 0.15** | **P3 — Global Magnitude** | Highest accuracy across all 27 runs (98.21 % → 0.60 weight); 0.271 ms is effectively tied with P2/B0 (0.25 weight); the 1.196 MB footprint is irrelevant in a 2 GB tier (0.15 weight). |

Full scored rationale in `optimization/select_best.py` and `results/phase4/selection.json`.

---

## Phase 5 — Collective Intelligence

The aggregator (`collective/aggregator.py`) implements **all three mechanisms**
required by the specification:

| Mechanism | Spec requirement | Implementation |
|---|---|---|
| **Weighted voting**     | Vote scaled by historical accuracy × softmax confidence | `w_i = acc_i × conf_i`, normalized, then `argmax Σ w_i · 𝟙[class]` |
| **Confidence gating**   | If collective confidence < 70 %, re-query all VMs on an augmented input | `if conf < 0.70 → re-run` |
| **Load balancing**      | Redirect request if a VM exceeds CPU 85 % or RAM 90 % | Checked before dispatch in `orchestrator.py` |

### Evaluation — 10-specimen smoke test

> ⚠️ **Scope caveat — read this first.** All 10 specimens in the evaluated batch
> were **class N (normal)**. This test validates the *plumbing* (orchestration,
> MQTT telemetry, weighted voting, overload alerting, consensus logic) but **does
> not** stress the collective on minority classes (S / V / F / Q). The 100 %
> consensus figure below should be read as a **correctness check of the collective
> pipeline**, not as a classification benchmark. A stratified suite spanning all
> 5 AAMI classes is listed under *Future Work*.

| Metric | Value |
|---|---|
| Specimens evaluated | 10 |
| Collective consensus rate | **100 %** (30/30 node predictions agreed) |
| Collective accuracy      | **100 %** (10/10) |
| Mean collective confidence | **0.990** |
| Min / max collective confidence | 0.931 / 0.9998 |
| Specimens flagged `validation_needed` | **0** |

### Per-node behaviour across the batch

| Node | Champion | Mean latency (ms) | Latency range (ms) | Overloads | Trigger |
|------|----------|-------------------|--------------------|-----------|---------|
| VM1 | Q2 | 218.9 | 2.9 – 651.1   | **10 / 10** | `RAM = 100 %` |
| VM2 | P2 | 54.7  | 3.8 – 194.6   | 1 / 10 (spec 0) | `CPU = 100 %` |
| VM3 | P3 | 22.6  | 1.1 – 50.0    | 1 / 10 (spec 0) | `CPU = 100 %` |

### What the alerts tell us

- **VM1 (Q2) is memory-starved.** Every prediction triggers `RAM = 100 %` on the
  500 MB container. The ~672 MB resident set exceeds the limit and forces the OS
  to swap — which explains its 2–3 order-of-magnitude latency variance
  (2.9 ms → 651 ms).
- **VM2 / VM3 only alert on specimen 0.** Both hit 100 % CPU on the first
  specimen — a **cold-start effect** (model warm-up + first PyTorch allocator
  pass). Steady-state utilisation stays below 10 %.
- **The collective layer masks individual node weakness.** Despite VM1 being on
  the edge of failure on every specimen, weighted voting produced unanimous,
  high-confidence results because VM2 and VM3 were healthy.

### Future work

- Replace the all-N batch with a **stratified suite** covering all 5 AAMI classes.
- Measure collective **recall on minority classes** to quantify the true fusion gain.
- Compare weighted voting against **majority voting** and **max-confidence** baselines.

---

## Phase 6 — ThingsBoard Supervision

A local ThingsBoard Community Edition instance runs in Docker alongside the three
tier containers. Every VM publishes a JSON telemetry message over MQTT after each
inference; the orchestrator publishes the collective decision and alert flags.

### Telemetry schema (per node, per inference)

```json
{
  "vm_id": "VM1",
  "timestamp": "2026-05-17T22:35:29Z",
  "technique": "Quantification statique PTQ",
  "prediction": "N",
  "prediction_class": 0,
  "confidence": 0.9998,
  "inference_time_ms": 496.19,
  "cpu_usage_pct": 48.2,
  "ram_usage_mb": 671.79,
  "ram_free_mb": 0.0,
  "model_size_mb": 0.295,
  "patient_id": "P-2026-000",
  "specimen_id": 0,
  "tech_id": "Q2",
  "overloaded": true,
  "alert_msg": "RAM=100.0%"
}
```

### Dashboards delivered

| # | Dashboard | Contents |
|---|-----------|----------|
| 1 | **Real-time monitoring** | Live predictions, collective diagnosis, confidence gauge |
| 2 | **System resources**     | Per-VM CPU and RAM time-series |
| 3 | **Technique comparison** | 3 × 8 heatmap of composite scores |
| 4 | **Collective intelligence** | Collective vs. individual accuracy, consensus rate |
| 5 | **Alerts**               | Latency > 300 ms · RAM > 90 % · collective accuracy < 80 % |

Dashboard JSON import files live in `thingsboard/dashboards/`. The full
demonstration video is hosted at `docs/demo.mp4` *(pending final recording)*.

---

## Engineering Lessons

1. **Over-correction is real.** Aggressive SMOTE + FocalLoss + WeightedSampler
   pushed N-recall to zero (16.4 % accuracy). The winning recipe keeps the *real*
   distribution and re-weights conservatively.
2. **Pruning can regularize.** Global magnitude pruning (P3) *improved* accuracy by
   0.78 pp over the FP32 baseline.
3. **Quantization cost is real on x86 CPU.** All Q1–Q5 eager-mode variants land at
   ~0.27–0.60 ms/sample vs 0.017 ms for FP32 — the win is **model size**, not
   latency, unless the target ships a dedicated INT8 runtime.
4. **Classical baselines matter.** A well-tuned RandomForest (Phase 2) matches the
   CNN within 0.2 pp at a fraction of the training complexity.
5. **Edge tiers fail in different ways.** VM1 fails on *memory* (swap-thrashing
   on every specimen); VM2/VM3 fail on *CPU* (cold-start only). A single tier-
   agnostic model would have hidden both failure modes.
6. **Collective accuracy ≠ individual accuracy.** The collective layer returned
   100 % on the evaluated batch even though one of its three members (VM1) was
   overloaded on *every* prediction — but see the Phase 5 scope caveat.

---

## Repository Structure

```
collective-intelligence/
├── README.md                        # This file
├── RESULTS.md                       # Full per-run results (all 27 VM rows)
├── GRADING.md                       # Rubric mapping — 100-point grid
├── MASTER_GUIDE.md                  # Full command-by-command run guide
├── walkthrough.md                   # Phase 3–6 narrative writeup
├── requirements.txt
│
├── environment/                     # Phase 0 — VM simulation
│   ├── Dockerfile
│   ├── docker-compose.yml           # vm1, vm2, vm3, thingsboard
│   └── check_resources.sh           # Spec-required resource verifier
│
├── dataset/                         # MIT-BIH download + preprocessing
├── baseline/                        # Phase 1 — baseline training
├── models/                          # ECGNet1D + optimization wrappers
│
├── optimization/                    # Phase 2 — 8 techniques
│   ├── compare_techniques.py
│   ├── select_best.py               # Phase 4 — champion selection
│   ├── P1_unstructured_pruning/
│   ├── P2_structured_pruning/
│   ├── P3_magnitude_pruning/
│   ├── Q1_dynamic_quant/
│   ├── Q2_static_ptq/
│   ├── Q3_qat/
│   ├── Q4_weight_only/
│   └── Q5_mixed_precision/
│
├── deployment/                      # Phase 3 — per-VM inference engine
├── collective/                      # Phase 5 — orchestrator + aggregator
├── thingsboard/                     # Phase 6 — dashboards + MQTT client
├── notebooks/                       # Interactive runs
├── results/                         # Every JSON / PNG artifact
│   ├── mitbih_csv/  ·  optimization/  ·  phase3/
│   ├── phase4/      ·  collective/    ·  graphs_and_images/
│   └── eda/
│
└── run_phase3.py                    # 3 × 8 matrix sweep driver
```

---

## Getting Started

### Prerequisites

- Python 3.13
- PyTorch 2.x (CPU build is enough)
- Docker Desktop with ≥ 4 GB RAM free
- MIT-BIH Arrhythmia CSV dataset mounted at `/mitbih`

### Installation

```bash
git clone <repository-url>
cd collective-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Phase 1 — Train the baseline model
python baseline/prepare_data.py
python baseline/train.py --epochs 20

# Phase 2 — Generate all 8 optimized variants
python optimization/compare_techniques.py

# Phase 3 — Benchmark all techniques across the 3 VM tiers
docker compose -f environment/docker-compose.yml build
python run_phase3.py

# Phase 4 — Select the champion per tier
python optimization/select_best.py

# Phase 5–6 — Collective hub + supervision
docker compose -f environment/docker-compose.yml up -d thingsboard
python collective/orchestrator.py
```

The ThingsBoard UI is available at <http://localhost:8080> (default credentials:
`tenant@thingsboard.org` / `tenant`).

See `MASTER_GUIDE.md` for the complete annotated command sequence.

---

## Deliverables Checklist

Per the specification (`projet_iot.pdf`, §6):

- [x] **Technical report (10–12 pages, PDF)** — `docs/report.pdf`
- [x] **Structured GitHub repository** — this repo
- [ ] **Demonstration video (Phase 6 only)** — `docs/demo.mp4` *(pending)*
- [x] **Google Sheets tracking row completed** — see course link
- [x] **VM creation scripts** — `environment/docker-compose.yml`
- [x] **Resource verification script** — `environment/check_resources.sh`
- [x] **Preprocessing script + class description** — `dataset/`
- [x] **One script per optimization technique (8 total)** — `optimization/`
- [x] **Comparative table + accuracy/size/speed plots** — `results/optimization/`
- [x] **Deployment scripts + 3 × 8 matrix CSV** — `deployment/`, `results/phase3/`
- [x] **Selection table + per-VM justification** — `results/phase4/selection.json`
- [x] **Orchestrator + comparison tables (collective vs. individual)** — `collective/`
- [x] **MQTT Python client + dashboard JSON + annotated screenshots** — `thingsboard/`

---

## References

[1] Liang, T., Glossner, J., Wang, L., Shi, S., & Zhang, X. (2021).
    *Pruning and quantization for deep neural network acceleration: A survey.*
    **Neurocomputing**, 461, 370–403.
    <https://doi.org/10.1016/j.neucom.2021.07.045>

[2] Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018).
    *ECG Heartbeat Classification: A Deep Transferable Representation.*
    IEEE ICHI 2018.

[3] Acharya, U. R., et al. (2017).
    *A deep convolutional neural network model to classify heartbeats.*
    **Computers in Biology and Medicine**, 89, 389–396.

[4] Hannun, A. Y., et al. (2019).
    *Cardiologist-level arrhythmia detection and classification in ambulatory
    electrocardiograms using a deep neural network.*
    **Nature Medicine**, 25, 65–69.

---

## License

Developed for academic purposes in the *Systèmes Embarqués et Objets Connectés*
module at **ENS Martil — Abdelmalek Essaâdi University**. Please contact the author
before any reuse.
