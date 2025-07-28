from __future__ import print_function

import os
import math
import numpy as np
import argparse
import pickle
from datetime import datetime
import skimage.io as io

from keras.models import *
from keras.layers import LeakyReLU
from keras import backend as keras
import tensorflow as tf
from tensorflow.keras.models import save_model, load_model
from sklearn.model_selection import KFold

from models import get_model
from data_generators import (
    get_generators_s1s2,
    get_generators_s1,
    get_generators_s2
)

# ------------------------------ Argument Parser ------------------------------
parser = argparse.ArgumentParser(description="Inductive Conformal Prediction Training Pipeline")
parser.add_argument('--dataset', type=str, required=True, choices=['S1GFloods', 'Ombria'])
parser.add_argument('--model_type', type=str, choices=['S1', 'S2', 'S1_S2'], default=None)
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--path2', type=str, default=None)
parser.add_argument('--temperature_scaling', action='store_true')
parser.add_argument('--perc', type=float, default=0.9)
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


def train_percent_to_k(train_percent, tolerance=1e-6):
    val_percent = 1 - train_percent
    k_exact = 1 / val_percent
    k_rounded = round(k_exact)

    if abs(k_exact - k_rounded) > tolerance:
        raise ValueError(f"train_percent={train_percent} does not yield an integer K. Got K={k_exact:.4f}")

    return k_rounded


# ------------------------- Training Function -------------------------
def train_icp(file_splits, batch_size, epochs, ts, nc_score='res'):
    val_scores = []    
    assert len(file_splits) == 1, "file_splits should contain only one (train_idx, val_idx) split."
    train_idx, val_idx = file_splits[0]
    print(f"\nTraining on {len(train_idx)} samples, validating on {len(val_idx)} samples")

    # Select generators based on model type
    if model_selection == 'S1_S2':
        train_gen, val_gen = get_generators_s1s2(path1, path2, train_idx, val_idx, batch_size=batch_size)
    elif model_selection == 'S2':
        train_gen, val_gen = get_generators_s2(path1, train_idx, val_idx, batch_size=batch_size)
    else:
        train_gen, val_gen = get_generators_s1(path1, train_idx, val_idx, batch_size=batch_size, dataset_type=dataset_type)

    # Get and train model
    model = get_model(model_selection, ts)
    model.fit(
        train_gen,
        steps_per_epoch=len(train_idx) // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=max(1, len(val_idx) // batch_size),
        verbose=1
    )

    # Compute NC scores on validation set
    steps = math.ceil(len(val_idx) / batch_size)
    y_val = []
    for _, y in val_gen.take(steps):
        y_val.append(y.numpy())
    y_val = np.concatenate(y_val).reshape(-1, 1)

    y_pred = model.predict(val_gen, steps=steps).reshape(-1, 1)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if nc_score == 'res':
        nc_scores = np.abs(y_pred - y_val)
    else:
        raise ValueError("Invalid nc_score specified. Available option: 'res'")
        
    val_scores.append(nc_scores)
    total_val_scores = sum(score.shape[0] for score in val_scores)/(256*256)
    print("Total validation samples:", total_val_scores)
    return model, val_scores

# ------------------------- Save Results -------------------------
def save_models(dataset_type, models, scores, save_dir, perc, epochs):
    os.makedirs(save_dir, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    with open(os.path.join(save_dir, f"{dataset_type}_models_ICP{perc}_ep{epochs}_{date}.pkl"), 'wb') as f:
        pickle.dump(models, f)
    with open(os.path.join(save_dir, f"{dataset_type}_scores_ICP{perc}_ep{epochs}_{date}.pkl"), 'wb') as f:
        pickle.dump(scores, f)
    print("Saved models and scores.")

# ------------------------- Run Training -------------------------
K = train_percent_to_k(args.perc)
kf = KFold(n_splits=K, shuffle=True, random_state=42)
file_splits = [next(iter(kf.split(after_images)))]  # just one split (e.g. 90% train, 10% val)

assert all(a.split('_')[-1] == b.split('_')[-1] == m.split('_')[-1]
           for a, b, m in zip(after_images, before_images, mask_images_filename))


model, val_scores = train_icp(
    file_splits,
    batch_size=8,
    epochs=args.epochs,
    ts=args.temperature_scaling
)
save_models(dataset_type, model, val_scores, args.save_dir, args.perc, args.epochs)