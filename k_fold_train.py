from __future__ import print_function
import os
import numpy as np
import math
import argparse
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import KFold
import pickle
from skimage import io

from utils import *
from models import get_model
from data_generators import get_generators_s2

# ------------------------------ Argument Parser ------------------------------
parser = argparse.ArgumentParser(description="K-Fold Conformal Prediction Training Pipeline")
parser.add_argument('--path', type=str, required=True, help='Path to Ombria dataset')
parser.add_argument('--temperature_scaling', action='store_true')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

# ------------------------------ Load Labels ------------------------------
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = io.imread(os.path.join(folder, filename)) / 255.0
        images.append(np.where(img > 0, 1, 0))
    return images

# ------------------------------ Dataset Setup ------------------------------
model_selection = 'S2'
dataset_type = 'Ombria'
path = args.path

mask_images = load_images_from_folder(os.path.join(path, 'MASK'))
after_images = sorted(os.listdir(os.path.join(path, 'AFTER')))
before_images = sorted(os.listdir(os.path.join(path, 'BEFORE')))
mask_files = sorted(os.listdir(os.path.join(path, 'MASK')))

assert all(a.split('_')[-1] == b.split('_')[-1] == m.split('_')[-1]
           for a, b, m in zip(after_images, before_images, mask_files)), "Filename mismatch"

# ------------------------------ Training ------------------------------
def train_k_fold(file_splits, K, batch_size, epochs, ts):
    models, val_scores, best_temps = [], [], []
    for fold_idx, (train_idx, val_idx) in enumerate(file_splits):
        print(f"\nTraining Fold {fold_idx+1}/{K}")
        train_gen, val_gen = get_generators_s2(path, train_idx, val_idx, batch_size=batch_size)
        model = get_model(model_selection, ts)

        model.fit(
            train_gen,
            steps_per_epoch=len(train_idx) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=max(1, len(val_idx) // batch_size),
            verbose=1
        )

        steps = math.ceil(len(val_idx) / batch_size)
        y_val = [y.numpy() for _, y in val_gen.take(steps)]
        y_val = np.concatenate(y_val).reshape(-1, 1)

        y_pred = model.predict(val_gen, steps=steps).reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if ts:
            logits = inverse_sigmoid(y_pred)
            best_T = find_best_temperature(np.squeeze(logits, -1), y_val)
            print("Best temperature:", best_T)
            best_temps.append(best_T)
            y_pred = tf.sigmoid(temperature_scaling(logits, best_T)).numpy()

        nc_scores = np.abs(y_pred - y_val)
        models.append(model)
        val_scores.append(nc_scores)

    total_samples = sum(score.shape[0] for score in val_scores) / (256 * 256)
    print("Total validation samples:", total_samples)
    assert total_samples == len(after_images), "Mismatch in sample count"
    return models, val_scores, best_temps

# ------------------------------ Save ------------------------------
def save_models(models, scores, temps, save_dir, K, epochs, ts):
    os.makedirs(save_dir, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")

    base = f"Ombria"
    if ts:
        suffix = f"TS_models_K{K}_ep{epochs}_{date}.pkl"
        temp_file = f"{base}temps_K{K}_ep{epochs}_{date}.pkl"
        with open(os.path.join(save_dir, temp_file), 'wb') as f:
            pickle.dump(temps, f)
    else:
        suffix = f"models_K{K}_ep{epochs}_{date}.pkl"

    with open(os.path.join(save_dir, suffix), 'wb') as f:
        pickle.dump(models, f)
    with open(os.path.join(save_dir, suffix.replace("models", "scores")), 'wb') as f:
        pickle.dump(scores, f)
    print("Saved models, scores, and temps")

# ------------------------------ Run ------------------------------
kf = KFold(n_splits=args.K, shuffle=True, random_state=42)
splits = list(kf.split(after_images))

models, val_scores, best_temps = train_k_fold(
    splits,
    K=args.K,
    batch_size=8,
    epochs=args.epochs,
    ts=args.temperature_scaling
)

save_models(models, val_scores, best_temps, args.save_dir, args.K, args.epochs, args.temperature_scaling)
