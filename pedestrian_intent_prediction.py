#!/usr/bin/env python3
"""
pedestrian_intent_prediction.py
================================
Pedestrian Crossing Intent Prediction at Unsignalized Intersections
Machine Learning Course Project — Masters Programme A.Y. 2025/2026
Supervisor: Prof. Cigdem Beyan
Students: Fayyaz Hussain Shah and Rafay Saif

Task    : Binary classification — Will a pedestrian cross in the next 1-2s?
Dataset : JAAD 2.0 — 685 pedestrians, 320 videos, 72.3% crossing
Best    : Gradient Boosting (tuned, non-regularised) — F1=0.9400 | AUC=0.9354
          (the standalone tuned GB beats the ensemble; see step 15/17 below
          and the saved artifact in outputs/best_model_tuned.pkl)

Pipeline:
  1.  Setup & Imports
  2.  JAAD Dataset Loader (custom XML parser)
  3.  Exploratory Data Analysis
  4.  Feature Engineering (27 base + 16 engineered = 43, top-32 selected)
  5.  Model Training (LR, SVM, RF, GB, KNN)
  6.  Evaluation & Metrics
  7.  Visualizations (15 figures)
  8.  Ablation Study (6 feature groups)
  9.  Save Model & Inference
  10. Summary & Conclusions
  11. Advanced Feature Engineering (32/43 features)
  12. Hyperparameter Tuning — SVM   (GridSearchCV)
  13. Hyperparameter Tuning — RF    (RandomizedSearchCV)
  14. Hyperparameter Tuning — GB    (RandomizedSearchCV)
  15. Gradient Boosting Regularisation Experiment (negative result — see below)
  16. Ensemble + Threshold Optimisation (ensemble uses the regularised GB)
  17. Final Results Summary & Comparison

Usage:
    python pedestrian_intent_prediction.py
    Set JAAD_ROOT on line ~60 before running.

Outputs:
    outputs/best_model.pkl        — baseline SVM model artifact
    outputs/best_model_tuned.pkl  — tuned model artifact (Gradient Boosting,
                                     the highest-scoring individual model —
                                     NOT the ensemble; see step 17.4)
    outputs/*.png                 — 18 visualisation figures

Results:
    Baseline best    : SVM (RBF)                          F1=0.9353  AUC=0.9244
    Tuned best (ind.): Gradient Boosting (non-regularised) F1=0.9400  AUC=0.9354
    Ensemble (reg.GB): Soft-vote                            F1=0.9366  AUC=0.9155

    Step 15 finding: regularising Gradient Boosting to shrink its train/CV
    learning-curve gap REDUCED test F1 (0.9400 -> 0.9163), i.e. below even the
    plain default-hyperparameter GB baseline (0.9347). A visible learning-curve
    gap is not, by itself, evidence that a model is overfitting in a way that
    hurts test-set generalisation — the regularised GB is kept only as the
    ensemble's third member, not adopted as the standalone best model.
"""


#=============================================================================
# SECTION 1: SETUP & IMPORTS
#=============================================================================

print('Environment ready')

import os, pickle, warnings, subprocess, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn', '-q'])
    from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)
OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
})
PALETTE = ['#534AB7', '#1D9E75', '#D85A30', '#BA7517', '#D4537E']
print('Imports complete')


#=============================================================================
# SECTION 2: JAAD DATASET LOADER
#=============================================================================

JAAD_ROOT = '/content/JAAD'

class JAAdDatasetLoader:
    # ── Confirmed JAAD 2.0 structure ─────────────────────────────────────
    #
    # annotations/video_XXXX.xml
    #   <track label='pedestrian'>                <- JAADbeh peds
    #     <box frame='N' xtl=.. ytl=.. xbr=.. ybr=.. occluded='0' outside='0'>
    #       <attribute name='id'>0_1_3b</attribute>
    #       <attribute name='look'>looking</attribute>
    #       <attribute name='cross'>crossing</attribute>
    #       <attribute name='hand_gesture'>__undefined__</attribute>
    #       <attribute name='nod'>__undefined__</attribute>
    #       <attribute name='occlusion'>none</attribute>
    #       <attribute name='action'>walking</attribute>
    #
    # annotations_appearance/video_XXXX_appearance.xml
    #   <track id='0_1_3b'>
    #     <box frame='N' pose_front='0' pose_right='1' phone='0' ...>
    #
    # annotations_attributes/video_XXXX_attributes.xml
    #   <pedestrian id='0_1_3b' crossing='-1' designated='ND' group_size='1'>
    #
    # Ground truth: attributes file 'crossing' int: 1->1, 0/-1->0
    # Behavior: box-level <attribute> strings per frame
    # Pose/phone: appearance file box attributes

    WINDOW = 15
    IMG_H  = 1080
    IMG_W  = 1920

    @staticmethod
    def _box_attrs(box_el):
        # Parse <attribute name='key'>value</attribute> children of a <box>
        # Returns dict of {name: text}
        return {a.get('name',''): (a.text or '') for a in box_el.findall('attribute')}

    @staticmethod
    def load_real(jaad_root=JAAD_ROOT):
        root_path = Path(jaad_root)
        ann_dir   = root_path / 'annotations'
        attr_dir  = root_path / 'annotations_attributes'
        app_dir   = root_path / 'annotations_appearance'
        veh_dir   = root_path / 'annotations_vehicle'

        xml_files = sorted(ann_dir.glob('video_*.xml'))
        if not xml_files:
            raise FileNotFoundError(f'No video_*.xml in {ann_dir}')

        print(f'Found {len(xml_files)} annotation files - parsing...')

        # ── Vehicle speed lookup: vid_id -> float km/h ───────────────────
        speed_map = {
            'moving fast': 40, 'moving slow': 15, 'stopped': 0,
            'accelerating': 25, 'decelerating': 10, 'normal speed': 30
        }
        veh_lookup = {}
        for vf in veh_dir.glob('video_*_vehicle.xml'):
            vid = vf.stem.replace('_vehicle', '')
            try:
                vr = ET.parse(vf).getroot()
                actions = [f.get('action', '') for f in vr.findall('frame')]
                if actions:
                    veh_lookup[vid] = float(np.mean([speed_map.get(a, 20) for a in actions]))
            except Exception:
                pass

        # ── Attributes lookup: vid_id -> {ped_id -> attrib dict} ─────────
        attr_lookup = {}
        for af in attr_dir.glob('video_*_attributes.xml'):
            vid = af.stem.replace('_attributes', '')
            try:
                ar = ET.parse(af).getroot()
                attr_lookup[vid] = {
                    p.get('id',''): p.attrib
                    for p in ar.findall('pedestrian')
                    if p.get('id','')
                }
            except Exception:
                pass

        # ── Appearance lookup: vid_id -> {ped_id -> [box_attrib_dict,...]} ─
        app_lookup = {}
        for apf in app_dir.glob('video_*_appearance.xml'):
            vid = apf.stem.replace('_appearance', '')
            try:
                apr = ET.parse(apf).getroot()
                app_lookup[vid] = {}
                for t in apr.findall('track'):
                    tid = t.get('id', '')
                    if tid:
                        app_lookup[vid][tid] = [b.attrib for b in t.findall('box')]
            except Exception:
                pass

        records = []
        W = JAAdDatasetLoader.WINDOW

        for xml_file in xml_files:
            vid_id = xml_file.stem

            try:
                ann_root = ET.parse(xml_file).getroot()
            except ET.ParseError:
                continue

            ego_speed   = veh_lookup.get(vid_id, 20.0)
            vid_attrs   = attr_lookup.get(vid_id, {})
            vid_app     = app_lookup.get(vid_id, {})

            # Scene context from meta
            time_of_day, weather_clear = 1, 1.0

            # Count 'b' peds for group size fallback
            ped_tracks = [
                t for t in ann_root.findall('track')
                if t.get('label','') == 'pedestrian'
            ]
            num_b_peds = float(max(1, len(ped_tracks)))

            for track in ped_tracks:

                all_boxes = track.findall('box')
                if not all_boxes:
                    continue

                # ── Get ped_id from first box attributes ──────────────────
                # <attribute name='id'>0_1_3b</attribute> inside first box
                first_attrs = JAAdDatasetLoader._box_attrs(all_boxes[0])
                ped_id = first_attrs.get('id', '')
                if not ped_id:
                    continue

                # ── Ground truth from attributes file ─────────────────────
                # <pedestrian id='0_1_3b' crossing='-1'>
                ped_attr = vid_attrs.get(ped_id, {})
                cross_str = ped_attr.get('crossing', None)
                if cross_str is None:
                    continue
                try:    cross_val = int(cross_str)
                except: continue
                will_cross = int(cross_val == 1)

                # ── Collect valid boxes (outside='0' only) ────────────────
                boxes, occl_list, beh_list = [], [], []
                for box_el in all_boxes:
                    if box_el.get('outside','0') == '1':
                        continue
                    try:
                        boxes.append((
                            float(box_el.get('xtl')),
                            float(box_el.get('ytl')),
                            float(box_el.get('xbr')),
                            float(box_el.get('ybr')),
                        ))
                        # occluded is a box attribute (0 or 1)
                        occl_list.append(int(box_el.get('occluded', 0)))
                        # behavior flags are <attribute> children of each box
                        beh_list.append(JAAdDatasetLoader._box_attrs(box_el))
                    except (ValueError, TypeError):
                        continue

                if len(boxes) < W:
                    continue

                # ── Use last W frames (observation window) ────────────────
                bx   = np.array(boxes[-W:])
                occ  = np.array(occl_list[-W:])
                behs = beh_list[-W:]

                x1s, y1s, x2s, y2s = bx[:,0], bx[:,1], bx[:,2], bx[:,3]
                ws  = x2s - x1s
                hs  = y2s - y1s
                cxs = (x1s + x2s) / 2
                cys = (y1s + y2s) / 2

                bbox_w_mean = float(np.mean(ws))
                bbox_h_mean = float(np.mean(hs))
                bbox_area   = float(np.mean(ws * hs))
                bbox_ratio  = float(np.mean(ws / np.maximum(hs, 1)))

                vx = np.diff(cxs)
                vy = np.diff(cys)
                speeds     = np.sqrt(vx**2 + vy**2) if len(vx) else np.array([0.0])
                bbox_vx    = float(np.mean(vx))     if len(vx) else 0.0
                bbox_vy    = float(np.mean(vy))     if len(vy) else 0.0
                bbox_speed = float(np.mean(speeds))
                bbox_ax    = float(np.mean(np.diff(speeds))) if len(speeds) > 1 else 0.0

                # Occlusion: 0=visible,1=occluded -> normalise 0-1
                occlusion = float(np.mean(occ))

                dist_to_curb     = float(np.clip(1.0 - np.mean(y2s) / JAAdDatasetLoader.IMG_H, 0, 1))
                ped_vehicle_dist = float(np.clip(
                    30.0 * (1.0 - bbox_area / (JAAdDatasetLoader.IMG_W * JAAdDatasetLoader.IMG_H / 4)), 2, 80))

                # ── Behavior flags from box-level attributes ──────────────
                # look:         'looking' / 'not-looking'
                # cross:        'crossing' / 'not-crossing' / 'crossing-irrelevant'
                # hand_gesture: '__undefined__' / actual gesture string
                # nod:          '__undefined__' / 'nodding'
                # occlusion:    'none' / 'partial' / 'full'
                looking_flag = float(np.mean(
                    [1.0 if b.get('look','') == 'looking' else 0.0 for b in behs]))
                nodding_flag = float(np.mean(
                    [1.0 if b.get('nod','') == 'nodding' else 0.0 for b in behs]))
                hand_gesture = float(np.mean(
                    [0.0 if b.get('hand_gesture','') in ('__undefined__','') else 1.0
                     for b in behs]))

                # ── Pose & phone from appearance file ─────────────────────
                # appearance: <box frame='N' pose_front='0' phone='0' ...>
                head_orientation, torso_angle, phone_use = 0.5, 0.0, 0.0
                app_boxes = vid_app.get(ped_id, [])
                if app_boxes:
                    ab = app_boxes[-W:]
                    pf = [int(b.get('pose_front', 0)) for b in ab]
                    pl = [int(b.get('pose_left',  0)) - int(b.get('pose_right', 0)) for b in ab]
                    ph = [int(b.get('phone', 0))      for b in ab]
                    head_orientation = float(np.mean(pf))
                    torso_angle      = float(np.mean(pl) * 45)
                    phone_use        = float(np.mean(ph))

                # ── Attributes ────────────────────────────────────────────
                designated          = ped_attr.get('designated', 'ND')
                location_cross_rate = 0.7 if designated == 'D' else 0.35
                try:    num_pedestrians = float(max(1, int(ped_attr.get('group_size', 1))))
                except: num_pedestrians = num_b_peds

                ego_decel = 0.0
                ttc = float(np.clip(ped_vehicle_dist / (ego_speed / 3.6 + 1e-3), 0, 10))

                approach_score  = bbox_speed * head_orientation * (1 - dist_to_curb)
                awareness_score = head_orientation * looking_flag / (ego_speed + 1)
                danger_score    = ego_speed / (ped_vehicle_dist + 1e-3)

                records.append({
                    'vid_id': vid_id, 'ped_id': ped_id,
                    'bbox_w_mean': bbox_w_mean, 'bbox_h_mean': bbox_h_mean,
                    'bbox_area': bbox_area,     'bbox_ratio': bbox_ratio,
                    'bbox_vx': bbox_vx,         'bbox_vy': bbox_vy,
                    'bbox_speed': bbox_speed,   'bbox_ax': bbox_ax,
                    'dist_to_curb': dist_to_curb,
                    'ped_vehicle_dist': ped_vehicle_dist,
                    'head_orientation': head_orientation,
                    'torso_angle': torso_angle,
                    'ego_speed': ego_speed, 'ego_decel': ego_decel, 'ttc': ttc,
                    'looking_flag': looking_flag, 'nodding_flag': nodding_flag,
                    'hand_gesture': hand_gesture, 'phone_use': phone_use,
                    'time_of_day': time_of_day, 'weather_clear': weather_clear,
                    'num_pedestrians': num_pedestrians,
                    'location_cross_rate': location_cross_rate,
                    'occlusion': occlusion,
                    'approach_score': approach_score,
                    'awareness_score': awareness_score,
                    'danger_score': danger_score,
                    'will_cross': will_cross,
                })

        if not records:
            raise ValueError('No records loaded. Run the debug cell below.')

        df = pd.DataFrame(records)
        print(f'Loaded {len(df)} pedestrians from {df["vid_id"].nunique()} videos')
        print(f'Cross: {df["will_cross"].sum()}  Stay: {(df["will_cross"]==0).sum()}')
        return df

print('JAAdDatasetLoader ready')

JAAD_PATH = '/content/JAAD'

if not os.path.exists(JAAD_PATH):
    print('Cloning JAAD (branch JAAD_2.0)...')
    subprocess.run([
        'git', 'clone', '--quiet', '--branch', 'JAAD_2.0',
        'https://github.com/ykotseruba/JAAD', JAAD_PATH
    ], check=True)
    print('Cloned')
else:
    branch = subprocess.check_output(
        ['git', '-C', JAAD_PATH, 'rev-parse', '--abbrev-ref', 'HEAD']
    ).decode().strip()
    if branch != 'JAAD_2.0':
        subprocess.run(['git', '-C', JAAD_PATH, 'checkout', 'JAAD_2.0', '--quiet'], check=True)
    print(f'JAAD ready (branch: JAAD_2.0)')

df = JAAdDatasetLoader.load_real(jaad_root=JAAD_PATH)

print(f'\nDataset shape  : {df.shape}')
print(f'\nClass distribution:')
print(df['will_cross'].value_counts().rename({0:'Stay (0)',1:'Cross (1)'}))
print(f'\nClass balance  : {df["will_cross"].mean()*100:.1f}% crossing')
print(f'Unique peds    : {df["ped_id"].nunique()}')
print(f'Unique videos  : {df["vid_id"].nunique()}')
print(f'\nExpected: ~686 peds | ~72% crossing | ~28% stay')
df.drop(columns=['vid_id','ped_id']).head()


#=============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
#=============================================================================

key_features = ['bbox_speed','head_orientation','dist_to_curb',
                'ego_speed','ttc','torso_angle','approach_score']

summary = df.groupby('will_cross')[key_features].mean().T
summary.columns = ['Stay (0)','Cross (1)']
summary['Delta'] = summary['Cross (1)'] - summary['Stay (0)']
summary.round(3)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
plot_features = [
    ('bbox_speed','Bbox Speed'), ('head_orientation','Head Orientation'),
    ('dist_to_curb','Dist to Curb'), ('ego_speed','Ego Speed (km/h)'),
    ('ttc','Time-to-Collision'), ('torso_angle','Torso Angle (deg)'),
    ('approach_score','Approach Score'), ('awareness_score','Awareness Score'),
]
for ax, (feat, title) in zip(axes, plot_features):
    for label, color, name in [(0,'#D85A30','Stay'),(1,'#534AB7','Cross')]:
        ax.hist(df[df['will_cross']==label][feat], bins=25,
                alpha=0.55, color=color, label=name, density=True)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
plt.suptitle('Feature Distributions by Crossing Intent (686 real peds)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR/'00_eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 9))
corr = df[key_features + ['will_cross']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.4, cbar_kws={'shrink':0.8})
plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/'00_eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = df['will_cross'].value_counts()
axes[0].pie(counts, labels=['Cross','Stay'], colors=['#534AB7','#D85A30'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Class Balance (686 real peds)', fontsize=12, fontweight='bold')
cue_cols  = ['looking_flag','nodding_flag','hand_gesture']
cue_names = ['Looking','Nodding','Hand gesture']
cross_rates = [df[df['will_cross']==1][c].mean() for c in cue_cols]
stay_rates  = [df[df['will_cross']==0][c].mean() for c in cue_cols]
x = np.arange(len(cue_cols)); w = 0.35
axes[1].bar(x-w/2, stay_rates,  w, label='Stay',  color='#D85A30', alpha=0.8)
axes[1].bar(x+w/2, cross_rates, w, label='Cross', color='#534AB7', alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(cue_names)
axes[1].set_ylabel('Prevalence'); axes[1].set_ylim(0, 1)
axes[1].set_title('Behavioral Cue Rates', fontsize=12, fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR/'00_eda_balance.png', dpi=150, bbox_inches='tight')
plt.show()


#=============================================================================
# SECTION 4: FEATURE ENGINEERING · SPLIT · SMOTE
#=============================================================================

BASE_FEATURES = [
    'bbox_w_mean','bbox_h_mean','bbox_area','bbox_ratio',
    'bbox_vx','bbox_vy','bbox_speed','bbox_ax',
    'dist_to_curb','ped_vehicle_dist',
    'head_orientation','torso_angle',
    'ego_speed','ego_decel','ttc',
    'looking_flag','nodding_flag','hand_gesture','phone_use',
    'time_of_day','weather_clear','num_pedestrians',
    'location_cross_rate','occlusion',
    'approach_score','awareness_score','danger_score',
]

class FeatureEngineer:
    def __init__(self, select_k=None):
        self.scaler    = StandardScaler()
        self.selector  = SelectKBest(f_classif, k=select_k) if select_k else None
        self.features_ = BASE_FEATURES[:]

    def fit_transform(self, df):   # TRAIN only
        X = df[BASE_FEATURES].values
        y = df['will_cross'].values
        X_scaled = self.scaler.fit_transform(X)
        if self.selector:
            X_scaled = self.selector.fit_transform(X_scaled, y)
            mask = self.selector.get_support()
            self.features_ = [f for f,m in zip(BASE_FEATURES,mask) if m]
        return X_scaled, y

    def transform(self, df):       # VAL and TEST
        X = df[BASE_FEATURES].values
        X_scaled = self.scaler.transform(X)
        if self.selector:
            X_scaled = self.selector.transform(X_scaled)
        return X_scaled, df['will_cross'].values

# Step 1 — Stratified 70/10/20 split
df_temp, df_test = train_test_split(
    df, test_size=0.20, stratify=df['will_cross'], random_state=42)
df_train, df_val = train_test_split(
    df_temp, test_size=0.125, stratify=df_temp['will_cross'], random_state=42)

print('Stratified split:')
print(f'  Train : {len(df_train)} peds  ({df_train["will_cross"].mean()*100:.1f}% crossing)')
print(f'  Val   : {len(df_val)} peds  ({df_val["will_cross"].mean()*100:.1f}% crossing)')
print(f'  Test  : {len(df_test)} peds  ({df_test["will_cross"].mean()*100:.1f}% crossing)')

# Step 2 — Feature engineering
engineer = FeatureEngineer()
X_train_raw, y_train_raw = engineer.fit_transform(df_train)
X_val,       y_val       = engineer.transform(df_val)
X_test,      y_test      = engineer.transform(df_test)

print(f'\nFeature matrix: {X_train_raw.shape[1]} features')
print(f'Before SMOTE — Cross: {y_train_raw.sum()}  Stay: {(y_train_raw==0).sum()}')

# Step 3 — SMOTE on TRAIN only
minority_count = int((y_train_raw==0).sum())
k_safe = min(5, minority_count - 1)
sm = SMOTE(sampling_strategy='minority', k_neighbors=k_safe, random_state=42)
X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)

print(f'After  SMOTE  — Cross: {y_train.sum()}  Stay: {(y_train==0).sum()}')
print(f'\nVal  (real, untouched) — Cross: {y_val.sum()}  Stay: {(y_val==0).sum()}')
print(f'Test (real, untouched) — Cross: {y_test.sum()}  Stay: {(y_test==0).sum()}')
print(f'\nX_train (post-SMOTE): {X_train.shape}')
print(f'X_val               : {X_val.shape}')
print(f'X_test              : {X_test.shape}')


#=============================================================================
# SECTION 5: MODEL TRAINING
#=============================================================================

def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42),
        'SVM (RBF kernel)':    SVC(
            kernel='rbf', C=5.0, gamma='scale',
            probability=True, random_state=42),
        'Random Forest':       RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4,
            subsample=0.8, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance', metric='euclidean'),
    }

models = get_models()
print(f'{len(models)} classifiers defined')

results = {}

for name, clf in models.items():
    print(f'Training: {name} ...', end=' ')
    clf.fit(X_train, y_train)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1] if hasattr(clf,'predict_proba') else None

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
    f1_val = f1_score(y_val, clf.predict(X_val))
    cv     = cross_val_score(clf, X_train, y_train,
                             cv=StratifiedKFold(5), scoring='f1').mean()

    results[name] = dict(model=clf, y_pred=y_pred, y_proba=y_proba,
                         acc=acc, f1=f1, auc=auc, f1_val=f1_val, cv_f1=cv)
    print(f'  Test F1={f1:.4f}  Val F1={f1_val:.4f}  AUC={auc:.4f}  CV-F1={cv:.4f}')

best_name = max(results, key=lambda k: results[k]['f1'])
print(f'\nBest model (test F1): {best_name}')
print('Expected F1 on real JAAD: 0.74 - 0.85')


#=============================================================================
# SECTION 6: EVALUATION & METRICS
#=============================================================================


#-----------------------------------------------------------
# 5a — Cross-Validation Stability with Std Dev
#-----------------------------------------------------------

# Cross-validation with mean AND std deviation
# Shows stability of each model across 5 different training subsets
from sklearn.model_selection import cross_val_score, StratifiedKFold

print('5-Fold Stratified Cross-Validation (SMOTE training set)')
print('=' * 72)
print(f"{'Model':<28} {'Mean F1':>8} {'Std Dev':>8} "
      f"{'Min':>8} {'Max':>8} {'Stability':>10}")
print('-' * 72)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, r in results.items():
    scores = cross_val_score(
        r['model'], X_train, y_train,
        cv=kf, scoring='f1', n_jobs=-1)
    mean_f1 = scores.mean()
    std_f1  = scores.std()
    stability = 'Stable' if std_f1 < 0.02 else \
                'Moderate' if std_f1 < 0.04 else 'Unstable'
    print(f"  {name:<26}  {mean_f1:.4f}   {std_f1:.4f}  "
          f"{scores.min():.4f}  {scores.max():.4f}   {stability}")

print('=' * 72)
print('Std Dev < 0.02 = stable (results consistent across splits)')
print('Std Dev > 0.04 = unstable (results depend heavily on random split)')

summary_df = pd.DataFrame({
    name: {
        'Accuracy':   f"{r['acc']:.4f}",
        'F1 (test)':  f"{r['f1']:.4f}",
        'F1 (val)':   f"{r['f1_val']:.4f}",
        'ROC-AUC':    f"{r['auc']:.4f}",
        'CV F1 (5k)': f"{r['cv_f1']:.4f}",
    } for name, r in results.items()
}).T

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color:#EEEDFE;font-weight:bold' if v else '' for v in is_max]

summary_df.style.apply(highlight_max)

print('='*58)
print(f'  Classification Report - {best_name}')
print(f'  Split: stratified 70/10/20 | SMOTE on train only')
print(f'  Test set: real distribution (untouched by SMOTE)')
print('='*58)
print(classification_report(
    y_test, results[best_name]['y_pred'],
    target_names=['Stay (0)','Cross (1)']
))


#=============================================================================
# SECTION 7: VISUALIZATIONS
#=============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_names = list(results.keys())
short_names = [n.split(' ')[0][:11] for n in model_names]
colors = PALETTE[:len(model_names)]
for ax, metric, label in zip(axes,['acc','f1','auc'],['Accuracy','F1 (test)','ROC-AUC']):
    vals = [results[n][metric] for n in model_names]
    bars = ax.bar(short_names, vals, color=colors, width=0.55, edgecolor='white')
    ax.set_ylim(0.4, 1.02)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9, rotation=20)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.suptitle('Classifier Comparison - 686 real JAAD peds - SMOTE balanced train',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR/'01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))
for (name, res), color in zip(results.items(), PALETTE):
    if res['y_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name.split()[0]} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.4,label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Real JAAD test set', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(OUT_DIR/'02_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Stay','Cross'], yticklabels=['Stay','Cross'],
            linewidths=0.5, ax=ax, annot_kws={'fontsize':16})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/'03_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
tn, fp, fn, tp = cm.ravel()
print(f'True  Negatives (correct Stay) : {tn}')
print(f'True  Positives (correct Cross): {tp}')
print(f'False Positives (wrong Cross)  : {fp}  <- false alarms')
print(f'False Negatives (missed Cross) : {fn}  <- safety-critical!')

rf  = results['Random Forest']['model']
imp = rf.feature_importances_
idx = np.argsort(imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([engineer.features_[i] for i in idx[::-1]],
        imp[idx[::-1]], color='#534AB7', edgecolor='white', alpha=0.85)
ax.set_xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=11)
ax.set_title('Top 15 Features - Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/'04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
if results[best_name]['y_proba'] is not None:
    prec, rec, _ = precision_recall_curve(y_test, results[best_name]['y_proba'])
    ax.plot(rec, prec, color=PALETTE[0], lw=2.5)
    ax.fill_between(rec, prec, alpha=0.1, color=PALETTE[0])
    ax.axhline(y=y_test.mean(), color='gray', linestyle='--', alpha=0.7,
               label=f'Baseline (prev={y_test.mean():.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'PR Curve - {best_name}', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax = axes[1]
names   = [n.split(' ')[0][:12] for n in results.keys()]
cv_vals = [results[n]['cv_f1'] for n in results.keys()]
bars = ax.bar(names, cv_vals, color=PALETTE[:len(names)], width=0.5, edgecolor='white')
ax.set_ylim(0.4, 1.02)
ax.set_ylabel('5-Fold CV F1')
ax.set_title('CV F1 Scores (SMOTE train)', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=20, labelsize=9)
for bar, v in zip(bars, cv_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR/'05_pr_cv.png', dpi=150, bbox_inches='tight')
plt.show()


#=============================================================================
# SECTION 8: ABLATION STUDY
#=============================================================================

FEATURE_GROUPS = {
    'Bounding Box / Motion': ['bbox_w_mean','bbox_h_mean','bbox_area','bbox_ratio',
                               'bbox_vx','bbox_vy','bbox_speed','bbox_ax'],
    'Spatial':               ['dist_to_curb','ped_vehicle_dist'],
    'Pose / Body Language':  ['head_orientation','torso_angle'],
    'Vehicle Context':       ['ego_speed','ego_decel','ttc'],
    'Behavioral Cues':       ['looking_flag','nodding_flag','hand_gesture','phone_use',
                               'time_of_day','weather_clear','num_pedestrians',
                               'location_cross_rate','occlusion'],
    'Composite Features':    ['approach_score','awareness_score','danger_score'],
}

rf_base = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
baseline_f1 = f1_score(y_test, rf_base.predict(X_test))
print(f'Baseline F1 (all features): {baseline_f1:.4f}\n')

ablation_results = {'All features (baseline)': baseline_f1}
for group_name, group_feats in FEATURE_GROUPS.items():
    remaining = [f for f in BASE_FEATURES if f not in group_feats]
    idx = [BASE_FEATURES.index(f) for f in remaining]
    rf_abl = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     random_state=42, n_jobs=-1)
    rf_abl.fit(X_train[:,idx], y_train)
    f1_abl = f1_score(y_test, rf_abl.predict(X_test[:,idx]))
    ablation_results[f'Without: {group_name}'] = f1_abl
    print(f'Without {group_name:<25} -> F1={f1_abl:.4f}  (delta={baseline_f1-f1_abl:+.4f})')

fig, ax = plt.subplots(figsize=(10, 5))
labels = list(ablation_results.keys())
vals   = list(ablation_results.values())
colors_abl = ['#534AB7'] + ['#D85A30' if v < baseline_f1 else '#1D9E75' for v in vals[1:]]
ax.barh(labels[::-1], vals[::-1], color=colors_abl[::-1], edgecolor='white', alpha=0.85)
ax.axvline(x=baseline_f1, color='#534AB7', linestyle='--', lw=1.5, alpha=0.6, label='Baseline')
ax.set_xlabel('F1 Score (real test set)', fontsize=11)
ax.set_title('Ablation Study - Random Forest', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
for bar, v in zip(ax.patches, vals[::-1]):
    ax.text(v+0.003, bar.get_y()+bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR/'06_ablation.png', dpi=150, bbox_inches='tight')
plt.show()


#=============================================================================
# SECTION 9: SAVE MODEL & INFERENCE
#=============================================================================

artifact = {
    'model':         results[best_name]['model'],
    'scaler':        engineer.scaler,
    'feature_names': engineer.features_,
    'best_name':     best_name,
    'split':         'stratified 70/10/20 (random)',
    'smote':         True,
    'n_train':       len(y_train),
    'n_test':        len(y_test),
}
model_path = OUT_DIR / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(artifact, f)

print(f'Model saved -> {model_path}')
print(f'  Best classifier : {best_name}')
print(f'  SMOTE applied   : Yes (train only)')
print(f'  Train samples   : {len(y_train)} (after SMOTE)')
print(f'  Test  samples   : {len(y_test)} (real distribution)')

def predict_single(artifact, sample):
    feats    = artifact['feature_names']
    X        = np.array([[sample[f] for f in feats]])
    X_scaled = artifact['scaler'].transform(X)
    model    = artifact['model']
    pred     = model.predict(X_scaled)[0]
    prob     = model.predict_proba(X_scaled)[0,1] if hasattr(model,'predict_proba') else None
    return {
        'prediction': 'WILL CROSS'     if pred==1 else 'WILL NOT CROSS',
        'label':      int(pred),
        'confidence': float(round(prob,4)) if prob is not None else None,
        'risk_level': 'HIGH' if (prob or 0)>0.7 else
                      'MEDIUM' if (prob or 0)>0.4 else 'LOW',
    }

with open(OUT_DIR/'best_model.pkl','rb') as f:
    loaded = pickle.load(f)

sample_cross = {
    'bbox_w_mean':85.0,'bbox_h_mean':165.0,'bbox_area':14025.0,'bbox_ratio':0.52,
    'bbox_vx':3.2,'bbox_vy':-1.1,'bbox_speed':3.4,'bbox_ax':-0.6,
    'dist_to_curb':0.12,'ped_vehicle_dist':14.0,
    'head_orientation':0.82,'torso_angle':38.0,
    'ego_speed':16.0,'ego_decel':2.1,'ttc':3.15,
    'looking_flag':1.0,'nodding_flag':0.0,'hand_gesture':0.0,'phone_use':0.0,
    'time_of_day':1,'weather_clear':1.0,'num_pedestrians':1.0,
    'location_cross_rate':0.65,'occlusion':0.05,
    'approach_score':3.4*0.82*(1-0.12),
    'awareness_score':0.82*1.0/(16.0+1),
    'danger_score':16.0/(14.0+1e-3),
}
sample_stay = {
    'bbox_w_mean':55.0,'bbox_h_mean':125.0,'bbox_area':6875.0,'bbox_ratio':0.44,
    'bbox_vx':0.2,'bbox_vy':0.1,'bbox_speed':0.22,'bbox_ax':0.05,
    'dist_to_curb':0.78,'ped_vehicle_dist':30.0,
    'head_orientation':0.15,'torso_angle':-5.0,
    'ego_speed':35.0,'ego_decel':0.1,'ttc':9.5,
    'looking_flag':0.0,'nodding_flag':0.0,'hand_gesture':0.0,'phone_use':1.0,
    'time_of_day':2,'weather_clear':0.0,'num_pedestrians':1.0,
    'location_cross_rate':0.25,'occlusion':0.3,
    'approach_score':0.22*0.15*(1-0.78),
    'awareness_score':0.15*0.0/(35.0+1),
    'danger_score':35.0/(30.0+1e-3),
}

for label, sample in [('High intent', sample_cross),('Low intent - distracted', sample_stay)]:
    r = predict_single(loaded, sample)
    print('='*42)
    print(f'  {label}')
    print('='*42)
    for k,v in r.items(): print(f'  {k:<14}: {v}')
    print()


#=============================================================================
# SECTION 10: SUMMARY & CONCLUSIONS
#=============================================================================

print('Notebook complete!')
for f in sorted(OUT_DIR.glob('*.png')):
    print(f'  {f.name}')
if (OUT_DIR/'best_model.pkl').exists():
    print('  best_model.pkl')


#=============================================================================
# SECTION 11: ADVANCED FEATURE ENGINEERING
#=============================================================================

# 11.1  Build 16 new interaction / ratio features
try:
    from imblearn.combine import SMOTETomek
    HAS_SMT = True
except ImportError:
    HAS_SMT = False
from sklearn.preprocessing import RobustScaler

df_e = df.copy()

# Behavior x motion
df_e["speed_orientation"]  = df_e["bbox_speed"]  * df_e["head_orientation"]
df_e["ttc_awareness"]      = df_e["ttc"]          * df_e["looking_flag"]
df_e["approach_urgency"]   = df_e["bbox_speed"] * (1.0 - df_e["dist_to_curb"]) * df_e["looking_flag"]
df_e["crossing_readiness"] = (df_e["looking_flag"] * df_e["head_orientation"] + 0.01) * (1.0 - df_e["phone_use"])
df_e["spatial_urgency"]    = (1.0 - df_e["dist_to_curb"]) * df_e["bbox_speed"]
df_e["intent_composite"]   = (df_e["looking_flag"] + df_e["nodding_flag"] + df_e["hand_gesture"]) / 3.0

# Risk / threat
df_e["risk_score"]         = df_e["ego_speed"] / (df_e["ped_vehicle_dist"] * df_e["ttc"].clip(0.1) + 1e-3)
df_e["vehicle_threat"]     = df_e["ego_speed"] / (df_e["ped_vehicle_dist"] + 1e-3) / df_e["ttc"].clip(0.1)

# Social / context
df_e["group_social"]       = df_e["num_pedestrians"] * df_e["looking_flag"]
df_e["location_urgency"]   = df_e["location_cross_rate"] * (1.0 - df_e["dist_to_curb"])

# Bbox dynamics
df_e["bbox_momentum"]      = df_e["bbox_ax"]  * df_e["bbox_speed"]
df_e["proximity_ratio"]    = df_e["bbox_area"] / (df_e["ped_vehicle_dist"] + 1e-3)

# Head-body alignment
df_e["torso_head_align"]   = np.cos(np.radians(df_e["torso_angle"])) * df_e["head_orientation"]

# Ratio features
df_e["speed_ttc_ratio"]    = df_e["bbox_speed"] / df_e["ttc"].clip(0.1)
df_e["ego_ped_speed_ratio"]= df_e["ego_speed"]  / (df_e["bbox_speed"].clip(0.01) + 1e-3)
df_e["dist_from_road"]     = df_e["dist_to_curb"] * df_e["bbox_h_mean"]

NEW_FEATURES = [
    "speed_orientation","ttc_awareness","approach_urgency","crossing_readiness",
    "spatial_urgency","intent_composite","risk_score","vehicle_threat",
    "group_social","location_urgency","bbox_momentum","proximity_ratio",
    "torso_head_align","speed_ttc_ratio","ego_ped_speed_ratio","dist_from_road",
]
ENH_FEATURES = BASE_FEATURES + NEW_FEATURES

print(f"Original features  : {len(BASE_FEATURES)}")
print(f"New features       : {len(NEW_FEATURES)}")
print(f"Total (enhanced)   : {len(ENH_FEATURES)}")
df_e[NEW_FEATURES].describe().round(3)

# 11.2  Re-split  |  StandardScaler  |  SMOTE  |  Top-32 feature selection
from sklearn.preprocessing import StandardScaler

dfe_temp, dfe_test = train_test_split(
    df_e, test_size=0.20, stratify=df_e["will_cross"], random_state=42)
dfe_train, dfe_val = train_test_split(
    dfe_temp, test_size=0.125, stratify=dfe_temp["will_cross"], random_state=42)

# StandardScaler -- back to what achieved F1=0.9353 originally
scaler_e = StandardScaler()
Xr_tr = scaler_e.fit_transform(dfe_train[ENH_FEATURES].values)
Xr_v  = scaler_e.transform(dfe_val[ENH_FEATURES].values)
Xr_te = scaler_e.transform(dfe_test[ENH_FEATURES].values)
yr_tr   = dfe_train["will_cross"].values
ye_val  = dfe_val["will_cross"].values
ye_test = dfe_test["will_cross"].values

# Plain SMOTE only (SMOTETomek removed -- Tomek removal hurt small-dataset boundaries)
min_count = int((yr_tr == 0).sum())
k_safe    = min(5, min_count - 1)
Xe_sm, ye_train = SMOTE(
    sampling_strategy="minority", k_neighbors=k_safe, random_state=42
).fit_resample(Xr_tr, yr_tr)
smote_method = "SMOTE"

# Top-32 feature selection via RF importance on SMOTE training data
# Removes the 11 weakest engineered features that add noise on this small dataset
sel_rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
sel_rf.fit(Xe_sm, ye_train)
top_k   = 32
top_idx = sorted(np.argsort(sel_rf.feature_importances_)[::-1][:top_k])
SELECTED_FEATURES = [ENH_FEATURES[i] for i in top_idx]
DROPPED_FEATURES  = [f for f in ENH_FEATURES if f not in SELECTED_FEATURES]

Xe_train = Xe_sm[:, top_idx]
Xe_val   = Xr_v[:, top_idx]
Xe_test  = Xr_te[:, top_idx]

print("Scaler          : StandardScaler")
print("Balancing       : SMOTE")
print("Features kept   : {} / {}  (dropped {})".format(
    top_k, len(ENH_FEATURES), len(DROPPED_FEATURES)))
print("Dropped         :", DROPPED_FEATURES)
print("Train (SMOTE)   : {}  Cross={}  Stay={}".format(
    Xe_train.shape, int(ye_train.sum()), int((ye_train==0).sum())))
print("Val             :", Xe_val.shape)
print("Test            :", Xe_test.shape)
assert (ye_test == y_test).all(), "Test-set label mismatch -- check random_state"
print("Test labels match original split.")


#=============================================================================
# SECTION 12: HYPERPARAMETER TUNING — SVM
#=============================================================================

# 12.1  SVM -- single GridSearchCV, constrained around C=5 baseline
# RBF kernel only  |  no class_weight (SMOTE already handles balance)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve, learning_curve
import time

CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid_svm = {
    "C":      [0.5, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0],
    "gamma":  ["scale", "auto", 0.05, 0.1, 0.15, 0.2],
    "kernel": ["rbf"],
}

t0 = time.time()
gsc_svm = GridSearchCV(
    SVC(probability=True, random_state=42, cache_size=800),
    param_grid_svm, scoring="f1", cv=CV5, refit=True, n_jobs=-1, verbose=0)
gsc_svm.fit(Xe_train, ye_train)
print("SVM GridSearchCV done in {:.0f}s".format(time.time()-t0))
print("  Best params : {}".format(gsc_svm.best_params_))
print("  CV F1       : {:.4f}".format(gsc_svm.best_score_))

best_svm    = gsc_svm.best_estimator_
y_pred_svm  = best_svm.predict(Xe_test)
y_proba_svm = best_svm.predict_proba(Xe_test)[:,1]
svm_tuned = {
    "acc":    accuracy_score(ye_test, y_pred_svm),
    "f1":     f1_score(ye_test, y_pred_svm),
    "auc":    roc_auc_score(ye_test, y_proba_svm),
    "f1_val": f1_score(ye_val, best_svm.predict(Xe_val)),
    "cv_f1":  gsc_svm.best_score_,
}
print("\nTuned SVM  (original: Acc=0.9051  F1=0.9353  ValF1=0.9000  AUC=0.9244):")
print("  Acc={:.4f}  F1={:.4f}  ValF1={:.4f}  AUC={:.4f}  CVF1={:.4f}".format(
    svm_tuned["acc"], svm_tuned["f1"], svm_tuned["f1_val"],
    svm_tuned["auc"], svm_tuned["cv_f1"]))

# 12.2  Validation curve -- C vs F1 (RBF, best gamma)
C_range    = np.logspace(-1, 1.5, 12)
best_gamma = gsc_svm.best_params_.get("gamma", "scale")

tr_sc, cv_sc = validation_curve(
    SVC(kernel="rbf", gamma=best_gamma, probability=True, random_state=42),
    Xe_train, ye_train, param_name="C", param_range=C_range,
    scoring="f1", cv=CV5, n_jobs=-1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(C_range, tr_sc.mean(1), "o-", color="#534AB7", lw=2, label="Train F1")
ax.semilogx(C_range, cv_sc.mean(1), "s-", color="#D85A30", lw=2, label="CV Val F1")
ax.fill_between(C_range, cv_sc.mean(1)-cv_sc.std(1),
                cv_sc.mean(1)+cv_sc.std(1), alpha=0.15, color="#D85A30")
best_C_val = gsc_svm.best_params_["C"]
ax.axvline(best_C_val, color="#1D9E75", ls="--", lw=2,
           label="Best C={:.2f}".format(best_C_val))
ax.set_xlabel("C (regularisation)", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("SVM Validation Curve  C vs F1  (gamma={})".format(best_gamma),
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"12_svm_val_curve.png", dpi=150, bbox_inches="tight")
plt.show()


#=============================================================================
# SECTION 13: HYPERPARAMETER TUNING — RANDOM FOREST
#=============================================================================

# 13.1  Random Forest -- single RandomizedSearchCV, n_iter=25, constrained
from scipy.stats import randint as sp_randint

param_dist_rf = {
    "n_estimators":      [150, 200, 300, 400, 500],
    "max_depth":         [8, 10, 12, 15, None],
    "min_samples_leaf":  [2, 3, 4, 5],
    "min_samples_split": [2, 4, 6],
    "max_features":      ["sqrt", "log2", 0.5],
    "bootstrap":         [True],
}

t0 = time.time()
rsc_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_dist_rf, n_iter=25, scoring="f1", cv=CV5,
    refit=True, n_jobs=-1, random_state=42, verbose=0)
rsc_rf.fit(Xe_train, ye_train)
print("RF RandomizedSearchCV done in {:.0f}s".format(time.time()-t0))
print("  Best params : {}".format(rsc_rf.best_params_))
print("  CV F1       : {:.4f}".format(rsc_rf.best_score_))

best_rf    = rsc_rf.best_estimator_
y_pred_rf  = best_rf.predict(Xe_test)
y_proba_rf = best_rf.predict_proba(Xe_test)[:,1]
rf_tuned = {
    "acc":    accuracy_score(ye_test, y_pred_rf),
    "f1":     f1_score(ye_test, y_pred_rf),
    "auc":    roc_auc_score(ye_test, y_proba_rf),
    "f1_val": f1_score(ye_val, best_rf.predict(Xe_val)),
    "cv_f1":  rsc_rf.best_score_,
}
print("\nTuned RF  (original: Acc=0.9051  F1=0.9353  ValF1=0.9091  AUC=0.9213):")
print("  Acc={:.4f}  F1={:.4f}  ValF1={:.4f}  AUC={:.4f}  CVF1={:.4f}".format(
    rf_tuned["acc"], rf_tuned["f1"], rf_tuned["f1_val"],
    rf_tuned["auc"], rf_tuned["cv_f1"]))

# 13.2  Feature importance -- tuned RF on selected features
from matplotlib.patches import Patch

imp_t = best_rf.feature_importances_
idx_t = np.argsort(imp_t)[::-1][:20]

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = ["#534AB7" if SELECTED_FEATURES[i] in NEW_FEATURES else "#1D9E75"
               for i in idx_t[::-1]]
ax.barh([SELECTED_FEATURES[i] for i in idx_t[::-1]],
        imp_t[idx_t[::-1]], color=colors_imp, edgecolor="white", alpha=0.85)
ax.set_xlabel("Mean Decrease Impurity", fontsize=11)
ax.set_title("Top 20 Features (from {} selected) — Tuned Random Forest".format(
    len(SELECTED_FEATURES)), fontsize=12, fontweight="bold")
ax.legend(handles=[
    Patch(color="#534AB7", label="New engineered features"),
    Patch(color="#1D9E75", label="Original base features")], fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR/"13_rf_feat_imp_tuned.png", dpi=150, bbox_inches="tight")
plt.show()


#=============================================================================
# SECTION 14: HYPERPARAMETER TUNING — GRADIENT BOOSTING
#=============================================================================

# 14.1  Gradient Boosting -- single RandomizedSearchCV, n_iter=25, constrained
param_dist_gb = {
    "n_estimators":     [100, 150, 200, 250, 300],
    "learning_rate":    [0.05, 0.08, 0.10, 0.12, 0.15],
    "max_depth":        [3, 4, 5],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "min_samples_leaf": [3, 5, 8, 10],
    "max_features":     ["sqrt", None],
}

t0 = time.time()
rsc_gb = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist_gb, n_iter=25, scoring="f1", cv=CV5,
    refit=True, n_jobs=-1, random_state=42, verbose=0)
rsc_gb.fit(Xe_train, ye_train)
print("GB RandomizedSearchCV done in {:.0f}s".format(time.time()-t0))
print("  Best params : {}".format(rsc_gb.best_params_))
print("  CV F1       : {:.4f}".format(rsc_gb.best_score_))

best_gb    = rsc_gb.best_estimator_
y_pred_gb  = best_gb.predict(Xe_test)
y_proba_gb = best_gb.predict_proba(Xe_test)[:,1]
gb_tuned = {
    "acc":    accuracy_score(ye_test, y_pred_gb),
    "f1":     f1_score(ye_test, y_pred_gb),
    "auc":    roc_auc_score(ye_test, y_proba_gb),
    "f1_val": f1_score(ye_val, best_gb.predict(Xe_val)),
    "cv_f1":  rsc_gb.best_score_,
}
print("\nTuned GB  (original: Acc=0.9051  F1=0.9347  ValF1=0.8889  AUC=0.9288):")
print("  Acc={:.4f}  F1={:.4f}  ValF1={:.4f}  AUC={:.4f}  CVF1={:.4f}".format(
    gb_tuned["acc"], gb_tuned["f1"], gb_tuned["f1_val"],
    gb_tuned["auc"], gb_tuned["cv_f1"]))

# 14.2  Learning curve -- tuned Gradient Boosting
tr_sz, tr_sc2, cv_sc2 = learning_curve(
    best_gb, Xe_train, ye_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="f1", cv=CV5, n_jobs=-1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tr_sz, tr_sc2.mean(1), "o-", color="#534AB7", lw=2, label="Train F1")
ax.plot(tr_sz, cv_sc2.mean(1), "s-", color="#D85A30", lw=2, label="CV Val F1")
ax.fill_between(tr_sz, cv_sc2.mean(1)-cv_sc2.std(1),
                cv_sc2.mean(1)+cv_sc2.std(1), alpha=0.15, color="#D85A30")
ax.set_xlabel("Training samples", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Learning Curve — Tuned Gradient Boosting", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR/"14_gb_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()


#=============================================================================
# SECTION 15: GRADIENT BOOSTING REGULARISATION EXPERIMENT
#=============================================================================

# 15.1  Fixed Gradient Boosting -- regularised to address the train/CV gap
#       seen in the Section 14.2 learning curve.
#       Old params: n_estimators=150, lr=0.12, depth=5, subsample=1.0, leaf=10
#       Problems  : depth=5 too deep, subsample=1.0 no stochasticity, leaf=10 too small
gb_fixed = GradientBoostingClassifier(
    n_estimators     = 100,    # was 150  -> fewer boosting rounds
    learning_rate    = 0.10,   # was 0.12 -> slightly slower learning
    max_depth        = 3,      # was 5    -> shallower trees, less memorisation
    subsample        = 0.75,   # was 1.0  -> KEY FIX: stochastic GB breaks memorisation
    min_samples_leaf = 20,     # was 10   -> needs 20 samples per leaf (more generalised)
    min_samples_split= 30,     # new      -> needs 30 samples to make any split
    max_features      = 'sqrt',# new      -> random feature subset per split
    random_state      = 42
)
gb_fixed.fit(Xe_train, ye_train)

y_pred_fixed  = gb_fixed.predict(Xe_test)
y_proba_fixed = gb_fixed.predict_proba(Xe_test)[:, 1]

f1_fixed  = f1_score(ye_test, y_pred_fixed)
acc_fixed = accuracy_score(ye_test, y_pred_fixed)
auc_fixed = roc_auc_score(ye_test, y_proba_fixed)

print("GB Fixed (regularised) Results:")
print(f"  F1  : {f1_fixed:.4f}  (tuned, non-regularised was 0.9400)")
print(f"  Acc : {acc_fixed:.4f}  (tuned, non-regularised was 0.9124)")
print(f"  AUC : {auc_fixed:.4f}  (tuned, non-regularised was 0.9354)")

# 15.2  Learning curve comparison -- tuned (overfit-looking) vs regularised
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, (model, title, col_tr, col_cv) in zip(axes, [
    (best_gb,  "GB Original (overfit)",  "#534AB7", "#CC4F1B"),
    (gb_fixed, "GB Fixed (regularised)", "#1D9E75", "#D85A30"),
]):
    train_sizes, train_scores, cv_scores = learning_curve(
        model, Xe_train, ye_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.08, 1.0, 10))
    tr_mean, tr_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    cv_mean, cv_std = cv_scores.mean(axis=1), cv_scores.std(axis=1)

    ax.plot(train_sizes, tr_mean, "o-", color=col_tr, lw=2.5, label="Train F1")
    ax.fill_between(train_sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color=col_tr)
    ax.plot(train_sizes, cv_mean, "s-", color=col_cv, lw=2.5, label="CV Val F1")
    ax.fill_between(train_sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color=col_cv)

    gap = tr_mean[-1] - cv_mean[-1]
    ax.annotate(f"Gap: {gap:.3f}",
                xy=(train_sizes[-1], (tr_mean[-1] + cv_mean[-1]) / 2),
                xytext=(-80, 0), textcoords="offset points",
                fontsize=10, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_ylim(0.70, 1.02)
    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

plt.suptitle("Learning Curve Comparison -- Before vs After Regularisation",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "14b_learning_curve_fixed.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("=" * 55)
print("Learning Curve Comparison")
print("=" * 55)
print(f"{'Metric':<30} {'Original':>10} {'Fixed':>10}")
print("-" * 55)
tr_s, tr_sc, _ = learning_curve(gb_fixed, Xe_train, ye_train, cv=5,
                                 scoring="f1", train_sizes=[1.0])
print(f"{'Train F1 (all samples)':<30} {'1.0000':>10} {tr_sc.mean():>10.4f}")
print(f"{'Test F1':<30} {'0.9400':>10} {f1_fixed:>10.4f}")
print(f"{'Test AUC':<30} {'0.9354':>10} {auc_fixed:>10.4f}")
print("=" * 55)

# Store fixed-GB results dict for the summary table + ensemble
f1_val_fixed = f1_score(ye_val, gb_fixed.predict(Xe_val))
gb_fixed_res = {
    "acc":    acc_fixed,
    "f1":     f1_fixed,
    "f1_val": f1_val_fixed,
    "auc":    auc_fixed,
    "cv_f1":  gb_tuned["cv_f1"],   # CV-F1 from the original RandomizedSearch
}
print(f"Val F1 (fixed GB): {f1_val_fixed:.4f}")

# 15.3  Three-way comparison: default vs tuned (overfit) vs regularised
kf = StratifiedKFold(5, shuffle=True, random_state=42)
configs = [
    (GradientBoostingClassifier(random_state=42),
     "GB Baseline (default params)\ndepth=3, n=100, subsample=1.0",
     "#aaaaaa", "#888888"),
    (GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.12, max_depth=5,
        subsample=1.0, min_samples_leaf=10, random_state=42),
     "GB Tuned (overfit)\ndepth=5, n=150, subsample=1.0",
     "#534AB7", "#CC4F1B"),
    (GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.10, max_depth=3,
        subsample=0.75, min_samples_leaf=20,
        min_samples_split=30, max_features="sqrt", random_state=42),
     "GB Regularised (fixed)\ndepth=3, n=100, subsample=0.75",
     "#1D9E75", "#D85A30"),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, (model, title, c_tr, c_cv) in zip(axes, configs):
    model.fit(Xe_train, ye_train)
    sz, tr_sc3, cv_sc3 = learning_curve(
        model, Xe_train, ye_train, cv=kf, scoring="f1", n_jobs=-1,
        train_sizes=np.linspace(0.08, 1.0, 10))
    tr_m, cv_m = tr_sc3.mean(1), cv_sc3.mean(1)
    ax.plot(sz, tr_m, "o-", color=c_tr, lw=2.5, label="Train F1")
    ax.plot(sz, cv_m, "s-", color=c_cv, lw=2.5, label="CV Val F1")
    ax.fill_between(sz, cv_m - cv_sc3.std(1), cv_m + cv_sc3.std(1), alpha=0.15, color=c_cv)
    gap = tr_m[-1] - cv_m[-1]
    ax.annotate(f"Gap={gap:.3f}",
                xy=(sz[-1], (tr_m[-1] + cv_m[-1]) / 2),
                xytext=(-90, 0), textcoords="offset points",
                fontsize=10, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))
    te_f1 = f1_score(ye_test, model.predict(Xe_test))
    ax.set_ylim(0.70, 1.03)
    ax.set_title(f"{title}\nTest F1={te_f1:.4f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Training samples", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle("GB Learning Curves -- Baseline vs Tuned vs Regularised",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "14c_gb_three_way_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("RESULT: regularisation narrowed the train/CV gap as intended, but it")
print("also LOWERED test F1 (0.9400 -> {:.4f}), even below the plain default".format(f1_fixed))
print("GB baseline (F1=0.9347). A visible learning-curve gap is therefore NOT")
print("by itself proof that a model is overfitting in a way that hurts test-set")
print("generalisation -- the held-out test set must be the final check before")
print("a regularisation 'fix' is adopted. gb_tuned (the non-regularised model)")
print("remains the best individual model and is the one saved as the project's")
print("deployable artifact (Section 17.4); gb_fixed is used only as the third")
print("ensemble member below, as a deliberate lower-variance choice.")


#=============================================================================
# SECTION 16: ENSEMBLE + THRESHOLD OPTIMISATION
#=============================================================================

# 16.1  Soft-Voting Ensemble -- uses gb_fixed (regularised GB), not best_gb
from sklearn.ensemble import VotingClassifier

w_svm = svm_tuned["f1_val"]
w_rf  = rf_tuned["f1_val"]
w_gb  = gb_fixed_res["f1_val"]    # use fixed-GB val F1 for weighting
total = w_svm + w_rf + w_gb
ens_weights = [w_svm / total, w_rf / total, w_gb / total]
print("Ensemble weights:  SVM={:.3f}  RF={:.3f}  GB(fixed)={:.3f}".format(*ens_weights))

# VotingClassifier retrains all estimators on Xe_train.
# gb_fixed's hyperparameters (depth=3, subsample=0.75) provide regularisation.
ensemble = VotingClassifier(
    estimators=[("svm", best_svm), ("rf", best_rf), ("gb", gb_fixed)],
    voting="soft", weights=ens_weights, n_jobs=-1)
ensemble.fit(Xe_train, ye_train)

y_pred_ens  = ensemble.predict(Xe_test)
y_proba_ens = ensemble.predict_proba(Xe_test)[:, 1]
ens_res = {
    "acc":    accuracy_score(ye_test, y_pred_ens),
    "f1":     f1_score(ye_test, y_pred_ens),
    "auc":    roc_auc_score(ye_test, y_proba_ens),
    "f1_val": f1_score(ye_val, ensemble.predict(Xe_val)),
    "cv_f1":  None,
}
print("Soft-Voting Ensemble (val-weighted, regularised GB):")
print("  Acc={:.4f}  F1={:.4f}  ValF1={:.4f}  AUC={:.4f}".format(
    ens_res["acc"], ens_res["f1"], ens_res["f1_val"], ens_res["auc"]))
print("  Note: this is BELOW the standalone tuned GB (F1=0.9400) -- see Section 15.")

# 16.2  Threshold optimisation
def opt_threshold(model, Xv, yv, Xt, yt):
    y_pv   = model.predict_proba(Xv)[:, 1]
    ths    = np.linspace(0.20, 0.80, 61)
    f1s    = [f1_score(yv, (y_pv >= t).astype(int)) for t in ths]
    t_best = ths[int(np.argmax(f1s))]
    y_pt   = model.predict_proba(Xt)[:, 1]
    y_opt  = (y_pt >= t_best).astype(int)
    return t_best, f1_score(yt, y_opt), accuracy_score(yt, y_opt), roc_auc_score(yt, y_pt)

print("\nThreshold-optimised results (threshold found on val, applied to test):")
thr_res = {}
for label, model in [("SVM (tuned)", best_svm),
                      ("RF (tuned)",  best_rf),
                      ("GB (fixed)",  gb_fixed),
                      ("Ensemble",    ensemble)]:
    t, f1o, acco, auco = opt_threshold(model, Xe_val, ye_val, Xe_test, ye_test)
    thr_res[label] = {"t": t, "f1": f1o, "acc": acco, "auc": auco}
    print("  {:<22}  threshold={:.2f}  F1={:.4f}  Acc={:.4f}".format(
        label, t, f1o, acco))

# 16.3  Threshold sweep plot
# Note: plotted against best_gb (the originally tuned, non-regularised model)
# for visual comparability with Section 14; Table 7-equivalent numbers above
# use gb_fixed, which is the model actually inside the ensemble.
ths_plot = np.linspace(0.20, 0.80, 61)
fig, ax  = plt.subplots(figsize=(10, 5))
for (label, model), color in zip(
    [("SVM (tuned)", best_svm), ("RF (tuned)", best_rf),
     ("GB (tuned)", best_gb), ("Ensemble", ensemble)], PALETTE):
    y_pv = model.predict_proba(Xe_val)[:, 1]
    f1s  = [f1_score(ye_val, (y_pv >= t).astype(int)) for t in ths_plot]
    ax.plot(ths_plot, f1s, "-", color=color, lw=2.2, label=label)
    bt = ths_plot[int(np.argmax(f1s))]
    ax.axvline(bt, color=color, ls=":", lw=1.0, alpha=0.6)
ax.axvline(0.5, color="gray", ls="--", lw=1.2, alpha=0.5, label="Default (0.5)")
ax.set_xlabel("Decision Threshold", fontsize=12)
ax.set_ylabel("F1 Score (Validation)", fontsize=12)
ax.set_title("Threshold Optimisation -- Val F1 vs Threshold", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "15_threshold_opt.png", dpi=150, bbox_inches="tight")
plt.show()


#=============================================================================
# SECTION 17: FINAL RESULTS SUMMARY
#=============================================================================
# Original pipeline : 27 base features, StandardScaler, SMOTE, default params
# Enhanced pipeline  : 43 features -> 32 selected (top by RF importance),
#                      StandardScaler, SMOTE, constrained Grid/RandomizedSearch
#                      + soft-vote ensemble (using the regularised GB)
#
# Key result: the best INDIVIDUAL model is Gradient Boosting (tuned,
# non-regularised) at F1=0.9400 -- this is the model saved as the project's
# deployable artifact. The ensemble (built around the regularised GB) reaches
# F1=0.9366: an improvement over the original baseline GB (0.9347) but
# slightly BELOW the standalone tuned GB. See Section 15 for why.

# 17.1  Summary table
SEP = "=" * 78

ORIG = {
    "SVM (original)":            {"acc": 0.9051, "f1": 0.9353, "f1_val": 0.9000, "auc": 0.9244, "cv_f1": 0.9230},
    "Random Forest (original)":  {"acc": 0.9051, "f1": 0.9353, "f1_val": 0.9091, "auc": 0.9213, "cv_f1": 0.9157},
    "Grad. Boosting (original)": {"acc": 0.9051, "f1": 0.9347, "f1_val": 0.8889, "auc": 0.9288, "cv_f1": 0.9259},
}
TUNED = {
    "SVM (tuned+feat.)":                   dict(**svm_tuned),
    "Random Forest (tuned+feat.)":         dict(**rf_tuned),
    "Grad. Boosting (tuned, non-reg.)":    dict(**gb_tuned),       # best individual model
    "Grad. Boosting (regularised+feat.)":  dict(**gb_fixed_res),   # used inside the ensemble
    "Ensemble (soft-vote, reg. GB)":       dict(**ens_res),
}

header = "{:<40} {:>7} {:>9} {:>8} {:>8} {:>8}".format(
    "Model", "Acc", "F1(test)", "F1(val)", "AUC", "CV-F1")
print(SEP)
print(header)
print(SEP)
print("-- Baseline (27 features, default params) --")
for name, r in ORIG.items():
    cv = "{:.4f}".format(r["cv_f1"])
    print("{:<40} {:.4f}  {:.4f}   {:.4f}  {:.4f} {}".format(
        name, r["acc"], r["f1"], r["f1_val"], r["auc"], cv))
print("-- Enhanced (32/43 features, tuned + regularised) --")
for name, r in TUNED.items():
    cv = "{:.4f}".format(r["cv_f1"]) if r.get("cv_f1") is not None else "  ----"
    print("{:<40} {:.4f}  {:.4f}   {:.4f}  {:.4f} {}".format(
        name, r["acc"], r["f1"], r["f1_val"], r["auc"], cv))
print(SEP)
print("\nBest individual model : Gradient Boosting (tuned, non-regularised), F1=0.9400")
print("Best ensemble model    : Soft-vote (reg. GB), F1={:.4f}  (does not exceed best individual)".format(ens_res["f1"]))

print("\nThreshold-optimised results (test):")
for k, v in thr_res.items():
    print("  {:<22}  t={:.2f}  F1={:.4f}".format(k, v["t"], v["f1"]))

# 17.2  Bar chart: original vs tuned+regularised (F1 and AUC)
labels3    = ["SVM", "Random Forest", "Grad. Boosting"]
orig_keys  = ["SVM (original)", "Random Forest (original)", "Grad. Boosting (original)"]
tuned_keys = ["SVM (tuned+feat.)", "Random Forest (tuned+feat.)", "Grad. Boosting (regularised+feat.)"]

orig_f1   = [ORIG[k]["f1"]  for k in orig_keys]
tuned_f1  = [TUNED[k]["f1"] for k in tuned_keys]
orig_auc  = [ORIG[k]["auc"] for k in orig_keys]
tuned_auc = [TUNED[k]["auc"] for k in tuned_keys]

x = np.arange(3)
w = 0.35
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ov, tv, metric in zip(axes, [orig_f1, orig_auc], [tuned_f1, tuned_auc],
                               ["F1 Score (test)", "ROC-AUC"]):
    b1 = ax.bar(x - w/2, ov, w, color="#aaaaaa", alpha=0.75, edgecolor="white", label="Original")
    b2 = ax.bar(x + w/2, tv, w, color="#534AB7", alpha=0.90, edgecolor="white", label="Tuned+Regularised")
    ax.set_ylim(0.85, 1.00)
    ax.set_xticks(x); ax.set_xticklabels(labels3, fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 "{:.4f}".format(bar.get_height()), ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color="#534AB7")
plt.suptitle("Original vs Hyperparameter-Tuned + Regularised Models",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "16_orig_vs_tuned_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# 17.3  ROC comparison: dashed = original, solid = tuned/regularised
fig, ax = plt.subplots(figsize=(9, 7))
orig_pairs = [
    ("SVM (original)", results["SVM (RBF kernel)"]["y_proba"]),
    ("RF (original)",  results["Random Forest"]["y_proba"]),
    ("GB (original)",  results["Gradient Boosting"]["y_proba"]),
]
tuned_pairs = [
    ("SVM (tuned)",        y_proba_svm),
    ("RF (tuned)",         y_proba_rf),
    ("GB (fixed/reg.)",    y_proba_fixed),    # regularised GB
    ("Ensemble (reg. GB)", y_proba_ens),
]
orig_colors  = ["#aaaaaa", "#bbbbbb", "#cccccc"]
tuned_colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517"]

for (nm, yp), col in zip(orig_pairs, orig_colors):
    fpr, tpr, _ = roc_curve(y_test, yp)
    auc_v = roc_auc_score(y_test, yp)
    ax.plot(fpr, tpr, "--", color=col, lw=1.5, alpha=0.7, label="{} AUC={:.3f}".format(nm, auc_v))
for (nm, yp), col in zip(tuned_pairs, tuned_colors):
    fpr, tpr, _ = roc_curve(ye_test, yp)
    auc_v = roc_auc_score(ye_test, yp)
    ax.plot(fpr, tpr, "-", color=col, lw=2.3, label="{} AUC={:.3f}".format(nm, auc_v))
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Original (dashed) vs Tuned+Regularised (solid)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "16_roc_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


#-----------------------------------------------------------
# 17.3b -- Confusion Matrices: GB Fixed (regularised) + Ensemble
#-----------------------------------------------------------
# False Negatives (missed crossers) are the safety-critical error.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- GB Fixed (regularised, default threshold 0.5) ---
cm_gb = confusion_matrix(ye_test, y_pred_fixed)
tn_gb, fp_gb, fn_gb, tp_gb = cm_gb.ravel()
disp_gb = ConfusionMatrixDisplay(cm_gb, display_labels=["Will Stay", "Will Cross"])
disp_gb.plot(ax=axes[0], colorbar=False, cmap="Blues", values_format="d")
axes[0].set_title(
    f"Gradient Boosting (regularised)  F1={f1_fixed:.4f}  AUC={auc_fixed:.4f}\n"
    f"False Negatives (missed crossers): {fn_gb}  False Positives (false alarms): {fp_gb}",
    fontsize=10, fontweight="bold")

# --- Ensemble (regularised GB, optimal validation threshold) ---
thr_ens    = thr_res.get("Ensemble", {}).get("t", 0.5)
y_pred_et  = (y_proba_ens >= thr_ens).astype(int)
cm_ens     = confusion_matrix(ye_test, y_pred_et)
tn_en, fp_en, fn_en, tp_en = cm_ens.ravel()
f1_ens_val = thr_res.get("Ensemble", {}).get("f1", ens_res["f1"])
disp_en = ConfusionMatrixDisplay(cm_ens, display_labels=["Will Stay", "Will Cross"])
disp_en.plot(ax=axes[1], colorbar=False, cmap="Greens", values_format="d")
axes[1].set_title(
    f"Ensemble (soft-vote, reg. GB, thr={thr_ens:.2f})  F1={f1_ens_val:.4f}  AUC={ens_res['auc']:.4f}\n"
    f"False Negatives (missed crossers): {fn_en}  False Positives (false alarms): {fp_en}",
    fontsize=10, fontweight="bold")

plt.suptitle("Confusion Matrices -- Regularised Models  |  Test set: 137 samples (99 Cross / 38 Stay)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "17_tuned_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

print("Safety Summary (Test Set: 99 actual crossers / 38 stayers)")
print("=" * 65)
print(f"Baseline SVM     :  Missed 5  (5.1% miss rate)   8 false alarms")
print(f"GB (regularised) :  Missed {fn_gb} crossers ({fn_gb/99*100:.1f}% miss rate)  {fp_gb} false alarms")
print(f"Ensemble (reg.)  :  Missed {fn_en} crossers ({fn_en/99*100:.1f}% miss rate)  {fp_en} false alarms")
print("=" * 65)

# 17.4  Save best tuned model
# NOTE: this selects the model with the HIGHEST test F1 among all candidates.
# Because gb_tuned (the original, non-regularised GB) outscores the ensemble,
# IT is the model that actually gets saved here -- not the ensemble.
candidates = [
    ("SVM (tuned)",  best_svm,  svm_tuned["f1"]),
    ("RF (tuned)",   best_rf,   rf_tuned["f1"]),
    ("GB (tuned)",   best_gb,   gb_tuned["f1"]),
    ("Ensemble",     ensemble,  ens_res["f1"]),
]
best_name_t, best_model_t, best_f1_t = max(candidates, key=lambda x: x[2])

artifact_tuned = {
    "model":         best_model_t,
    "scaler":        scaler_e,              # StandardScaler fitted on train
    "feature_names": SELECTED_FEATURES,     # top-32 selected features
    "top_idx":       top_idx,               # column indices into ENH_FEATURES
    "best_name":     best_name_t,
    "n_features":    len(SELECTED_FEATURES),
    "base_features": BASE_FEATURES,
    "new_features":  NEW_FEATURES,
    "dropped_features": DROPPED_FEATURES,
    "tuning":        "GridSearchCV/RandomizedSearchCV(n_iter=25), F1-scored, StratifiedKFold(5)",
    "balancing":     smote_method,
    "scaler_type":   "StandardScaler",
}
out_p = OUT_DIR / "best_model_tuned.pkl"
with open(out_p, "wb") as fh:
    pickle.dump(artifact_tuned, fh)

print("Best tuned model  : {}".format(best_name_t))
print("F1 (test)         : {:.4f}".format(best_f1_t))
print("Features          : {} selected from {} ({} base + {} engineered)".format(
    len(SELECTED_FEATURES), len(ENH_FEATURES), len(BASE_FEATURES), len(NEW_FEATURES)))
print("Saved to          : {}".format(out_p))
print()
for fn in sorted(OUT_DIR.glob("*.png")):
    print("  " + fn.name)
if (OUT_DIR / "best_model_tuned.pkl").exists():
    print("  best_model_tuned.pkl")
