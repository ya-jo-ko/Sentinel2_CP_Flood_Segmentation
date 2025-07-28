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
from data_generators import (
    get_generators_s1s2,
    get_generators_s1,
    get_generators_s2
)

# ------------------------------ Argument Parser ------------------------------
parser = argparse.ArgumentParser(description="K-Fold Conformal Prediction Training Pipeline")
parser.add_argument('--dataset', type=str, required=True, choices=['S1GFloods', 'Ombria'])
parser.add_argument('--model_type', type=str, choices=['S1', 'S2', 'S1_S2'], default=None)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--path2', type=str, default=None)
parser.add_argument('--temperature_scaling', action='store_true')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()


# ------------------------- Helper: Load Labels -------------------------
def load_images_from_folder(folder):
    images = []
    img_list = sorted(os.listdir(folder))
    for filename in img_list:
        img = io.imread(os.path.join(folder, filename)) / 255.0
        images.append(np.where(img > 0, 1, 0))
    return images

# --------------------------- Dataset Setup ---------------------------
if args.dataset == 'S1GFloods':
    dataset_type = 'S1GFloods'
    path1 = path2 = args.path
    model_selection = 'S1'
    mask_images = load_images_from_folder(os.path.join(path1, 'label'))
    mask_images_filename = sorted(os.listdir(os.path.join(path1, 'label')))
    after_images = sorted(os.listdir(os.path.join(path1, 'A')))
    before_images = sorted(os.listdir(os.path.join(path1, 'B')))
elif args.dataset == 'Ombria':
    dataset_type = 'Ombria'
    model_selection = args.model_type
    path1 = args.path
    path2 = args.path2 if args.model_type == 'S1_S2' else path1
    mask_images = load_images_from_folder(os.path.join(path1, 'MASK'))
    mask_images_filename = sorted(os.listdir(os.path.join(path1, 'MASK')))
    after_images = sorted(os.listdir(os.path.join(path1, 'AFTER')))
    before_images = sorted(os.listdir(os.path.join(path1, 'BEFORE')))
else:
    raise ValueError("Invalid dataset type")



# ------------------------- Training Function -------------------------
def train_k_fold(file_splits, K, batch_size, epochs, ts, nc_score='res'):
    models, val_scores = [], []
    best_temps = []
    for fold_idx, (train_idx, val_idx) in enumerate(file_splits):
        print(f"\nTraining Fold {fold_idx+1}/{K}")
        # Load model and data generator based on model type
        if model_selection == 'S1_S2':
            train_gen, val_gen = get_generators_s1s2(path1, path2, train_idx, val_idx, batch_size=batch_size)
        elif model_selection == 'S2':
            train_gen, val_gen = get_generators_s2(path1, train_idx, val_idx, batch_size=batch_size)
        else:
            train_gen, val_gen = get_generators_s1(path1, train_idx, val_idx, batch_size=batch_size, dataset_type=dataset_type)

        model = get_model(model_selection, ts)
        segmodel = model.fit(
            train_gen,
            steps_per_epoch=len(train_idx) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=max(1, len(val_idx) // batch_size),
            verbose=1
        )

        steps = math.ceil(len(val_idx) / batch_size)
        # Collect predictions and calculate nonconformity scores
        #steps = max(1, len(val_idx) // batch_size)
        y_val = []
        for _, y in val_gen.take(steps):
            y_val.append(y.numpy())
        y_val = np.concatenate(y_val).reshape(-1, 1)

        y_pred = model.predict(val_gen, steps=steps).reshape(-1, 1)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if nc_score == 'res':
            if ts == 1:
                y_true = y_val
                # Convert probabilities to logits
                logits = inverse_sigmoid(y_pred)
                logits_ = np.squeeze(logits,axis=-1)
                labels = y_true
                best_T = find_best_temperature(logits_, labels)
                T = best_T #1.8
                #best_temps.append(best_T)
                print("Best temperature:", T)
                best_temps.append(best_T)
                # Convert back to probabilities
                y_pred = tf.sigmoid(temperature_scaling(logits,T)).numpy()
            nc_scores = np.abs(y_pred - y_val)
        else:
            raise ValueError("Invalid nc_score specified. Available option: 'res'")

        models.append(model)
        val_scores.append(nc_scores)

    total_val_scores = sum(score.shape[0] for score in val_scores)/(256*256)
    print("Total validation samples:", total_val_scores)
    print("Expected total:", len(after_images))
    assert total_val_scores == len(after_images), "Mismatch in validation sample count!"

    return models, val_scores, best_temps

# ------------------------- Save Results -------------------------
def save_models(dataset_type, models, scores, save_dir, K, epochs, ts):
    os.makedirs(save_dir, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    if ts == 1:
        with open(os.path.join(save_dir, f"{dataset_type}_TS_models_K{K}_ep{epochs}_{date}_r2.pkl"), 'wb') as f:
            pickle.dump(models, f)
        with open(os.path.join(save_dir, f"{dataset_type}_TS_scores_K{K}_ep{epochs}_{date}_r2.pkl"), 'wb') as f:
            pickle.dump(scores, f)
        with open(os.path.join(save_dir, f"{dataset_type}temps_K{K}_ep{epochs}_{date}_r2.pkl"), 'wb') as f:
            pickle.dump(best_temps, f)
    else:
        with open(os.path.join(save_dir, f"{dataset_type}_models_K{K}_ep{epochs}_{date}_r2.pkl"), 'wb') as f:
            pickle.dump(models, f)
        with open(os.path.join(save_dir, f"{dataset_type}_scores_K{K}_ep{epochs}_{date}_r2.pkl"), 'wb') as f:
            pickle.dump(scores, f)
    print("Saved models and scores.")

# ------------------------- Run Training -------------------------
kf = KFold(n_splits=args.K, shuffle=True, random_state=42)
splits = list(kf.split(after_images))

assert all(a.split('_')[-1] == b.split('_')[-1] == m.split('_')[-1]
           for a, b, m in zip(after_images, before_images, mask_images_filename))

if args.temperature_scaling == 1:
    models, val_scores, best_temps = train_k_fold(
        splits,
        K=args.K,
        batch_size=8,
        epochs=args.epochs,
        ts=args.temperature_scaling
    )
    save_models(dataset_type, models, val_scores, args.save_dir, args.K, args.epochs, args.temperature_scaling)
else:
    models, val_scores, _ = train_k_fold(
        splits,
        K=args.K,
        batch_size=8,
        epochs=args.epochs
    )
    save_models(dataset_type, models, val_scores, args.save_dir, args.K, args.epochs, args.temperature_scaling)
