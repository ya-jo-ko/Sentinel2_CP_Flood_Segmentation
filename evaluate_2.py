import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from skimage import io
import tensorflow as tf
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope

from data_generators import test_generator_s2
from utils import *

# --------------------------- Argument Parser ---------------------------
parser = argparse.ArgumentParser(description="Evaluate S2 ICP Models on Ombria Test Set")
parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.pkl)')
parser.add_argument('--score_path', type=str, required=True, help='Path to validation scores (.pkl)')
parser.add_argument('--test_path', type=str, required=True, help='Path to Ombria test set')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--temperature_scaling', action='store_true')
args = parser.parse_args()

# --------------------------- Load Models & Scores ---------------------------
with custom_object_scope({'LeakyReLU': LeakyReLU}):
    with open(args.model_path, 'rb') as f:
        models = pickle.load(f)

with open(args.score_path, 'rb') as f:
    validation_scores = pickle.load(f)

if args.temperature_scaling:
    with open('./saved_models/S2temps_ombria.pkl', 'rb') as f:
        best_temps = pickle.load(f)

# --------------------------- Prepare Test Dataset ---------------------------
test_dataset = test_generator_s2(args.test_path, dataset_type='Ombria')

# --------------------------- Load Ground Truth Labels ---------------------------
def load_labels(label_path):
    images = []
    img_list = sorted(os.listdir(label_path))
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255.0
        images.append(img)
    return images

test_labels = load_labels(os.path.join(args.test_path, 'MASK'))

# --------------------------- Run Conformal Prediction ---------------------------
prediction_set, coverage, inefficiency, predictions, thresh = k_cv_cp_test(
    test_dataset=test_dataset,
    test_labels=test_labels,
    models=models,
    validation_scores=validation_scores,
    alpha=args.alpha,
    nc_score='res',
    ts=args.temperature_scaling
)

# --------------------------- Plot Prediction Outputs ---------------------------
output_dir = "./prediction_plots_S2"
os.makedirs(output_dir, exist_ok=True)

y_preds = [model.predict(test_dataset, verbose=0) for model in models]
y_pred_mean = np.mean(np.array(y_preds), axis=0)
y_pred_mean = np.squeeze(y_pred_mean, axis=-1)

threshold = 1 - thresh
num_samples = len(test_labels)

for idx in range(num_samples):
    pred_image = y_pred_mean[idx]
    gt_image = test_labels[idx]

    pred_binary = (pred_image >= 0.5).astype(np.uint8)
    non_flood_pred = 1 - pred_image
    flood_mask = pred_image >= threshold
    non_flood_mask = non_flood_pred >= threshold

    prediction_map = np.zeros_like(pred_image, dtype=int)
    prediction_map[flood_mask & non_flood_mask] = 2
    prediction_map[flood_mask & ~non_flood_mask] = 1
    prediction_map[~flood_mask & non_flood_mask] = 0

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(pred_binary, cmap='gray')
    plt.title("Binary Prediction")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction_map, cmap='viridis')
    plt.title("Conformal Prediction Set")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_image, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f"prediction_{idx:03d}.png"))
    plt.close()
    print(f"Saved: prediction_{idx:03d}.png")
