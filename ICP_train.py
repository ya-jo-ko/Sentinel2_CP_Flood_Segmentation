from __future__ import print_function

import os
import math
import numpy as np
import argparse
import pickle
from datetime import datetime
import skimage.io as io

import tensorflow as tf
from tensorflow.keras.models import save_model
from sklearn.model_selection import KFold

from models import get_model
from data_generators import get_generators_s2

# ------------------------------ Argument Parser ------------------------------
parser = argparse.ArgumentParser(description="Inductive Conformal Prediction Training Pipeline")
parser.add_argument('--path', type=str, required=True, help='Path to Ombria dataset')
parser.add_argument('--temperature_scaling', action='store_true', help='Apply temperature scaling')
parser.add_argument('--perc', type=float, default=0.9, help='Percentage of training data (e.g. 0.9 for 90%)')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and scores')
args = parser.parse_args()

# --------------------------- Load Binary Labels ---------------------------
def load_binary_masks(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = io.imread(os.path.join(folder, filename)) / 255.0
        images.append(np.where(img > 0, 1, 0))
    return images

# --------------------------- Setup Dataset ---------------------------
dataset_type = 'Ombria'
path = args.path
model_type = 'S2'

mask_images = load_binary_masks(os.path.join(path, 'MASK'))
after_images = sorted(os.listdir(os.path.join(path, 'AFTER')))
before_images = sorted(os.listdir(os.path.join(path, 'BEFORE')))
mask_filenames = sorted(os.listdir(os.path.join(path, 'MASK')))

assert all(a.split('_')[-1] == b.split('_')[-1] == m.split('_')[-1]
           for a, b, m in zip(after_images, before_images, mask_filenames)), \
           "Mismatch in filenames between BEFORE, AFTER, and MASK folders."

# --------------------------- Train/Test Split Helper ---------------------------
def get_k_from_percent(train_percent, tolerance=1e-6):
    val_percent = 1 - train_percent
    k_float = 1 / val_percent
    k_int = round(k_float)
    if abs(k_float - k_int) > tolerance:
        raise ValueError(f"Train percent {train_percent} doesn't yield valid K. Got {k_float:.4f}")
    return k_int

# --------------------------- Training Function ---------------------------
def train_icp(file_splits, batch_size, epochs, ts, nc_score='res'):
    train_idx, val_idx = file_splits[0]
    print(f"\nTraining on {len(train_idx)} samples, validating on {len(val_idx)} samples")

    train_gen, val_gen = get_generators_s2(path, train_idx, val_idx, batch_size=batch_size, dataset_type='Ombria')

    model = get_model(ts)
    model.fit(
        train_gen,
        steps_per_epoch=len(train_idx) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=max(1, len(val_idx) // batch_size),
        verbose=1
    )

    steps = math.ceil(len(val_idx) / batch_size)
    y_true = []
    for _, y in val_gen.take(steps):
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true).reshape(-1, 1)

    y_pred = model.predict(val_gen, steps=steps).reshape(-1, 1)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if nc_score == 'res':
        nc_scores = np.abs(y_pred - y_true)
    else:
        raise ValueError("Invalid nc_score. Only 'res' supported.")

    print("Validation samples:", nc_scores.shape[0] / (256 * 256))
    return model, [nc_scores]

# --------------------------- Save Function ---------------------------
def save_outputs(model, scores, save_dir, perc, epochs):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    with open(os.path.join(save_dir, f"Ombria_models_ICP{perc}_ep{epochs}_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(save_dir, f"Ombria_scores_ICP{perc}_ep{epochs}_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(scores, f)
    print(f"Saved model and scores to {save_dir}")

# --------------------------- Main Training Loop ---------------------------
K = get_k_from_percent(args.perc)
kf = KFold(n_splits=K, shuffle=True, random_state=42)
file_splits = [next(iter(kf.split(after_images)))]

model, val_scores = train_icp(
    file_splits=file_splits,
    batch_size=8,
    epochs=args.epochs,
    ts=args.temperature_scaling
)

save_outputs(model, val_scores, args.save_dir, args.perc, args.epochs)
