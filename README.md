# 🚶 Pedestrian Crossing Intent Prediction at Unsignalized Intersections

> **Machine Learning Course Project — Masters Programme | A.Y. 2025/2026**  
> **Supervisor:** Prof. Cigdem Beyan

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-JAAD%202.0-green)](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/)
[![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple)]()
[![Best F1](https://img.shields.io/badge/Best%20F1-0.9406-brightgreen)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [References](#references)

---

## Overview

Binary classification task for autonomous driving safety:  
**Will a pedestrian cross the road in the next 1–2 seconds?**

At unsignalized intersections, no traffic signal governs pedestrian movement. Autonomous vehicles must predict crossing intent from behavioral cues — head orientation, gaze direction, walking speed, proximity — to initiate emergency braking or speed adjustment before a collision occurs.

This project develops and evaluates a complete supervised machine learning pipeline on the **JAAD 2.0** benchmark, covering feature engineering, class imbalance handling, multi-model benchmarking, ablation analysis, hyperparameter optimisation, and ensemble combination.

---

## Dataset

**JAAD 2.0 — Joint Attention in Autonomous Driving**

| Property | Value |
|---|---|
| Total pedestrian tracks | 685 |
| Total annotated videos | 320 |
| Class balance | 72.3% crossing / 27.7% staying |
| Annotation type | Per-frame bounding box + behavioral attributes |
| Source | Ego-vehicle camera, real urban traffic |
| Train / Val / Test | 479 / 69 / 137 (stratified 70/10/20) |

**Download:** https://data.nvision2.eecs.yorku.ca/JAAD_dataset/

```bash
git clone https://github.com/ykotseruba/JAAD.git /content/JAAD
```

After cloning, download video clips from the official page and place them in `JAAD_clips/`.

---

## Results

### Baseline Models (27 features · StandardScaler · SMOTE · default params)

| Model | F1 (Test) | Accuracy | ROC-AUC | CV-F1 (5-fold) |
|---|---|---|---|---|
| Logistic Regression | 0.8923 | 0.8467 | 0.8676 | 0.8573 |
| **SVM (RBF kernel)** | **0.9353** | **0.9051** | **0.9244** | **0.9230** |
| **Random Forest** | **0.9353** | **0.9051** | 0.9213 | 0.9157 |
| Gradient Boosting | 0.9347 | 0.9051 | 0.9288 | 0.9259 |
| K-Nearest Neighbours | 0.8808 | 0.8321 | 0.8761 | 0.8950 |

### Enhanced Models (32/43 features · StandardScaler · SMOTE · tuned params)

| Model | F1 (Test) | Accuracy | ROC-AUC | Best Params |
|---|---|---|---|---|
| SVM (tuned) | 0.8976 | 0.8467 | 0.8813 | C=2.0, γ=0.2, RBF |
| Random Forest (tuned) | 0.9246 | 0.8905 | 0.9125 | n=300, depth=12 |
| Gradient Boosting (tuned) | 0.9400 | 0.9124 | 0.9354 | lr=0.12, n=150, depth=5 |
| **Ensemble (soft-vote, val-weighted)** | **0.9406** | **0.9124** | **0.9168** | SVM+RF+GB |

### Confusion Matrix — Best Model (Ensemble, test set = 137 samples)

```
                 Predicted Stay    Predicted Cross
Actual Stay           30                8           ← 8 false alarms
Actual Cross           4               95           ← 4 missed (safety-critical)

False Negative rate on crossers: 4.0% (4/99)
```

---

## Project Structure

```
pedestrian-intent-prediction/
│
├── Pedestrian_Intent_ML.ipynb    ← Main notebook (57 cells, 15 figures)
├── pedestrian_intent_prediction.py     ← Standalone Python script
├── README.md                           ← This file
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
```

### Google Colab (recommended)

All dependencies are pre-installed on Colab. Just mount Drive and clone JAAD:

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run('git clone https://github.com/ykotseruba/JAAD.git /content/JAAD', shell=True)
```

---

## Usage

### Run the full notebook

Open `Pedestrian_Intent_ML_FIXED.ipynb` in Google Colab and run all cells top to bottom.

Set the JAAD path in Cell 6:
```python
JAAD_ROOT = '/content/JAAD'   # path to cloned JAAD repo
```

### Run as Python script

```bash
python pedestrian_intent_prediction.py --jaad_path /path/to/JAAD
```

### Run inference on a single pedestrian

```python
import pickle

with open('outputs/best_model_tuned.pkl', 'rb') as f:
    artifact = pickle.load(f)

sample = {
    'bbox_speed': 0.045,
    'head_orientation': 0.72,
    'dist_to_curb': 1.2,
    'looking_flag': 1,
    'ego_speed': 28.5,
    # ... all 32 selected features
}

result = predict_single(artifact, sample)
print(result)
# {'prediction': 'WILL CROSS', 'confidence': 0.87, 'risk_level': 'HIGH'}
```

---

## Methodology

### 1. Feature Engineering

**27 base features** extracted from JAAD XML annotations across 6 groups:

| Group | Features | Ablation Impact |
|---|---|---|
| BBox / Motion | position, velocity, speed, acceleration | −0.060 F1 |
| Pose / Body Language | head orientation, torso angle | −0.049 F1 |
| Vehicle Context | ego speed, deceleration, TTC | −0.021 F1 |
| Spatial | dist to curb, ped-vehicle distance | −0.011 F1 |
| Composite (engineered) | risk score, approach urgency, intent composite | −0.011 F1 |
| Behavioral Cues | looking flag, hand gesture, phone use | +0.009 F1 |

**16 additional engineered features** (risk_score, approach_urgency, intent_composite, etc.) → 43 total → top 32 selected by Random Forest MDI importance.

### 2. Data Pipeline

```
Raw JAAD XML → Feature extraction (27 base)
                     ↓
              Feature engineering (+16)
                     ↓
         Stratified split 70 / 10 / 20
                     ↓
         StandardScaler (fitted on train only)
                     ↓
         SMOTE (training set only, k=5)
                     ↓
         SelectKBest top-32 features
```

### 3. Hyperparameter Optimisation

Constrained search ranges anchored around known good parameters — deliberately narrow to prevent overfitting to 5-fold CV on a 692-sample training set:

| Model | Method | Iterations | Best |
|---|---|---|---|
| SVM | GridSearchCV | 48 combinations × 5-fold | C=2.0, γ=0.2 |
| Random Forest | RandomizedSearchCV | 25 × 5-fold | n=300, depth=12 |
| Gradient Boosting | RandomizedSearchCV | 25 × 5-fold | lr=0.12, n=150, depth=5 |

### 4. Ensemble

Soft-voting ensemble with validation-F1-weighted votes:

```
Weights: SVM=0.328  RF=0.335  GB=0.337
Final probability = weighted average of class probabilities
Decision threshold: 0.56 (optimised on validation set)
```

---

## Key Findings

1. **Motion features are the most critical signal** — removing bounding box / motion features drops F1 by 0.060 (largest ablation drop). Moving toward a crossing is the strongest behavioral indicator.

2. **SVM with SMOTE produces miscalibrated probabilities** — the optimal threshold shifts to 0.69 (far from default 0.50), indicating the model learned an overly confident positive class bias.

3. **Wide hyperparameter search overfits on small datasets** — a 60-iteration unconstrained search on 692 samples degrades test performance despite improving CV F1. Constrained 25-iteration search is more reliable.

4. **Ensemble reduces false negatives** — from 5 (baseline SVM) to 4 (Ensemble), catching one more crossing pedestrian.

5. **Behavioral cues (looking flag, gesture, phone) add marginal noise** — on this 685-sample dataset, these features slightly hurt F1. They may be beneficial on larger datasets.

---

## Ablation Study Summary

```
Full feature set baseline F1 : 0.9360
Without BBox / Motion         : 0.8763  (−0.0597)  ← most critical
Without Pose / Body Language  : 0.8867  (−0.0493)
Without Vehicle Context       : 0.9154  (−0.0205)
Without Spatial features      : 0.9254  (−0.0106)
Without Composite features    : 0.9254  (−0.0106)
Without Behavioral Cues       : 0.9447  (+0.0088)  ← slight noise
```

---

## References

```bibtex
@inproceedings{rasouli2017they,
  title={Are They Going to Cross? A Benchmark Dataset and Baseline
         for Pedestrian Crosswalk Behavior},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={ICCVW},
  pages={206--213},
  year={2017}
}

@inproceedings{kotseruba2016joint,
  title={Joint attention in autonomous driving (JAAD)},
  author={Kotseruba, Iuliia and Rasouli, Amir and Tsotsos, John K},
  booktitle={arXiv:1609.04741},
  year={2016}
}

@article{chawla2002smote,
  title={SMOTE: Synthetic minority over-sampling technique},
  author={Chawla, N V and Bowyer, K W and Hall, L O and Kegelmeyer, W P},
  journal={Journal of Artificial Intelligence Research},
  volume={16},
  pages={321--357},
  year={2002}
}

@inproceedings{pedregosa2011scikit,
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, F and others},
  journal={JMLR},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

---

## Authors

| Name | Role |
|---|---|
| Fayyaz Hussain Shah | Feature engineering, model training, evaluation |
| Rafay Saif | Data pipeline, ablation study, report |

**Course:** Machine Learning — Masters Programme  
**Supervisor:** Prof. Cigdem Beyan  
**Academic Year:** 2025/2026

---

## License

This project is released for academic use only.  
JAAD dataset is subject to its own license — see https://github.com/ykotseruba/JAAD.
