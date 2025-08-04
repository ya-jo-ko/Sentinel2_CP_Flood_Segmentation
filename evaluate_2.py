import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from skimage import io
import tensorflow as tf
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
#from keras.utils.generic_utils import get_custom_objects
from keras.utils import custom_object_scope

from data_generators import test_generator_s2
from utils import *

# --------------------------- Argument Parser ---------------------------
parser = argparse.ArgumentParser(description="Evaluate Trained Models on Test Set")
parser.add_argument('--model_path', type=str, required=True, help='Path to saved model pickle (.pkl)')
parser.add_argument('--model_path2', type=str, required=True, help='Path to saved scores pickle (.pkl)')
parser.add_argument('--dataset', type=str, required=True, choices=['Ombria'])
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--test_path2', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--temperature_scaling', action='store_true')
args = parser.parse_args()

# Register LeakyReLU for deserialization
#get_custom_objects().update({'LeakyReLU': LeakyReLU})
# --------------------------- Load Model ---------------------------
# For new env
with custom_object_scope({'LeakyReLU': LeakyReLU}):
    with open(args.model_path, 'rb') as f:
        models = pickle.load(f)
# For old env 
'''
with open(args.model_path, 'rb') as f:
    models = pickle.load(f)
'''
#with open(args.model_path.replace("models_", "scores_"), 'rb') as f:
#    validation_scores = pickle.load(f)
with open(args.model_path2, 'rb') as f:
    validation_scores = pickle.load(f)
    
# --------------------------- Load Test Labels ---------------------------
def load_labels(label_path):
    images = []
    img_list = os.listdir(label_path)
    img_list.sort()
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255
        images.append(img)
    return images


# --------------------------- Conformal Prediction Evaluation ---------------------------
def k_cv_cp_test(test_dataset, test_labels, models, validation_scores, alpha=0.1, nc_score='res', ts=0):
    prediction_intervals = []
    candidate_labels = np.unique(test_labels)
    print("Unique Labels: ", candidate_labels)
    test_labels_flat = np.array(test_labels).reshape(-1, 1)
    print("Labels flat shape: ",test_labels_flat.shape)
    prediction_set = np.zeros((len(test_labels_flat), len(candidate_labels)))
    all_val_scores = np.concatenate(validation_scores, axis=0)
    N = all_val_scores.shape[0]
    #candidate_qs = np.linspace(0, np.max(all_val_scores), num=1000)
    candidate_qs = all_val_scores
    #min_valid_q = None
    '''
    for q in candidate_qs:
        miscoverage_sum = 0
        total_points = 0
        for val_fold_scores in validation_scores:
            fold_miscoverage = np.mean(val_fold_scores > q)
            miscoverage_sum += fold_miscoverage
            total_points += 1  # could weight by len(val_fold_scores) if uneven
        cv_risk = miscoverage_sum / total_points
        if cv_risk <= alpha:
            min_valid_q = q
            break
    '''
    epsilon = 0 #1e-4
    # Start from quantile threshold
    initial_q = np.quantile(all_val_scores, np.ceil((N + 1) * (1 - alpha)) / N, interpolation="higher")

    # Sort and remove duplicates for efficient scan
    candidate_qs = np.unique(np.sort(all_val_scores))

    # Find index of the first q <= initial_q
    start_idx = np.searchsorted(candidate_qs, initial_q, side="left")

    # Scan downward until coverage falls below 1 - alpha
    min_valid_q = None
    for i in tqdm(range(start_idx, -1, -1), desc="Refining threshold"):
        q = candidate_qs[i]
        #cv_risk = np.mean(all_val_scores > q)  # empirical miscoverage
        #print(cv_risk)
        #print(q)
        miscoverage_flags = []
        for val_fold_scores in validation_scores:
            fold_miscoverage = np.mean(val_fold_scores > q)
            miscoverage_flags.append(fold_miscoverage)
        cv_risk = np.mean(miscoverage_flags)
        if cv_risk <= alpha + epsilon:
            min_valid_q = q
        else:
            break  # stop at first q with too much risk (coverage < 1 - alpha)

    if min_valid_q is None:
        raise ValueError("No valid threshold found that satisfies the risk constraint.")

    thresholds_val = min_valid_q

    print(f"CV-Estimated Threshold: {thresholds_val}")
                
    for label_idx, label in enumerate(candidate_labels):
        test_nc_scores = []
        predictions = []
        K=0
        for model in models:
            K += 1
            y_pred = model.predict(test_dataset)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            predictions.append(y_pred)
            if nc_score == 'res':
                nc_score_k = np.abs(y_pred - label)
            else:
                raise ValueError("Invalid nc_score specified. Available option: 'res'")
            test_nc_scores.append(nc_score_k)

        test_av = np.min(test_nc_scores, axis=0)
        print(test_av.shape)
        for i in tqdm(range(int(test_labels_flat.shape[0])), desc="Processing Test Samples"):
            if test_av[i] <= thresholds_val:
                prediction_set[i,label_idx] = 1

    empirical_coverage_count = 0
    total_prediction_set_size = 0
    for idx in range(int(test_labels_flat.shape[0])):
        true_label = test_labels_flat[idx]
        if prediction_set[idx,int(true_label)]==1:
            empirical_coverage_count += 1
        total_prediction_set_size += np.sum(prediction_set[idx,:])

    empirical_coverage = empirical_coverage_count / int(test_labels_flat.shape[0])
    inefficiency = total_prediction_set_size / int(test_labels_flat.shape[0])
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Empirical Coverage: {empirical_coverage:.4f}")
    print(f"Inefficiency: {inefficiency:.4f}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    dir_name = f"metrics_K{K}_alpha{alpha}_e"#epochs{epochs}"
    os.makedirs(dir_name, exist_ok=True)

    np.save(os.path.join(dir_name, "prediction_set.npy"), prediction_set)
    np.save(os.path.join(dir_name, "predictions.npy"), predictions)
    np.save(os.path.join(dir_name, "test_labels.npy"), test_labels_flat)

    metrics = {
        "empirical_coverage": empirical_coverage,
        "inefficiency": inefficiency
    }
    np.save(os.path.join(dir_name, "metrics.npy"), metrics)

    print(f"\nResults saved to '{dir_name}'")

    return prediction_set, empirical_coverage, inefficiency, predictions, thresholds_val


# --------------------------- Run Evaluation ---------------------------
if args.dataset == 'Ombria':
    test_labels = load_labels(os.path.join(args.test_path, 'MASK'))
else:
    raise ValueError("Unsupported dataset type. Use 'Ombria'.")

prediction_set, coverage, inefficiency, predictions, thresh = k_cv_cp_test(
    test_dataset=test_dataset,
    test_labels=test_labels,
    models=models,
    validation_scores=validation_scores,
    alpha=args.alpha,
    nc_score='res',
    ts = args.temperature_scaling
)


# Binarize predictions before evaluation
#avg_preds = np.mean([pred[..., 0] for pred in predictions], axis=0)

# Binarize the averaged predictions
#binary_preds = np.where(avg_preds > 0.5, 1, 0).reshape(256,256,)

#test_labels_np = np.array(test_labels)#.reshape(-1)

# Evaluate against ground truths
#segm_metrics = compute_segmentation_metrics(binary_preds, test_labels_np)

#print("\nSegmentation Metrics:")
#print(f"Pixel Accuracy: {segm_metrics['pixel_accuracy']:.4f}")
#print(f"Mean IoU: {segm_metrics['mean_IoU']:.4f}")
#print(f"Frequency Weighted IoU: {segm_metrics['frequency_weighted_IoU']:.4f}")



def plot_prediction_distributions(predictions, save_dir="plots"):
    """
    Generate and save diagnostic plots for a list of model prediction arrays.
    Each prediction is expected to be of shape (N, H, W, 1).
    """
    os.makedirs(save_dir, exist_ok=True)

    K = len(predictions)
    flattened_preds = [pred.ravel() for pred in predictions]

    # Plot individual histograms
    plt.figure(figsize=(12, 6))
    for i, flat in enumerate(flattened_preds):
        plt.hist(flat, bins=50, alpha=0.5, label=f'Model {i}', density=True)
    plt.title("Prediction Distributions per Model")
    plt.xlabel("Predicted Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_prediction_histograms.png"))
    plt.close()

    # Aggregate statistics across models
    stacked_preds = np.stack([pred.squeeze() for pred in predictions], axis=0)  # (K, N, H, W)
    min_preds = np.min(stacked_preds, axis=0).ravel()
    max_preds = np.max(stacked_preds, axis=0).ravel()
    mean_preds = np.mean(stacked_preds, axis=0).ravel()

    # Plot overlay of min, max, mean
    plt.figure(figsize=(12, 6))
    plt.hist(min_preds, bins=50, alpha=0.5, label='Min', density=True)
    plt.hist(mean_preds, bins=50, alpha=0.5, label='Mean', density=True)
    plt.hist(max_preds, bins=50, alpha=0.5, label='Max', density=True)
    plt.title("Aggregated Prediction Statistics (Min, Mean, Max)")
    plt.xlabel("Predicted Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "aggregated_prediction_stats.png"))
    plt.close()

    # Plot difference between max and min per pixel
    uncertainty_map = max_preds - min_preds
    plt.figure(figsize=(12, 4))
    plt.hist(uncertainty_map, bins=50, alpha=0.8, color='purple', density=True)
    plt.title("Prediction Spread (Max - Min) Histogram")
    plt.xlabel("Prediction Spread")
    plt.ylabel("Density")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_spread_histogram.png"))
    plt.close()

example_predictions = predictions
#plot_prediction_distributions(example_predictions)

# Output directory for saving plots
output_dir = "./prediction_plots_K10"
os.makedirs(output_dir, exist_ok=True)

# Average predictions across models
y_preds = [model.predict(test_dataset, verbose=0) for model in models]
y_pred_mean = np.mean(np.array(y_preds), axis=0)  # Shape: (N, 256, 256, 1)
y_pred_mean = np.squeeze(y_pred_mean, axis=-1)    # Shape: (N, 256, 256)

threshold = 1 - thresh  # thresh returned by ICP function
num_samples = len(test_labels)

for idx in range(num_samples):
    pred_image = y_pred_mean[idx]                # (256, 256)
    gt_image = test_labels[idx]                  # (256, 256)

    # Thresholded binary prediction
    pred_binary = (pred_image >= 0.5).astype(np.uint8)

    # Conformal prediction set (flood, non-flood, both)
    non_flood_pred = 1 - pred_image
    flood_mask = pred_image >= threshold
    non_flood_mask = non_flood_pred >= threshold

    prediction_set = np.zeros_like(pred_image, dtype=int)
    prediction_set[flood_mask & non_flood_mask] = 2  # Both
    prediction_set[flood_mask & ~non_flood_mask] = 1  # Flood only
    prediction_set[~flood_mask & non_flood_mask] = 0  # Non-flood only

    # Plot
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(pred_binary, cmap='gray')
    plt.title("Predicted (Thr=0.5)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction_set, cmap='viridis', interpolation='nearest')
    plt.title("Conformal Prediction Set")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gt_image, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # Save figure
    output_path = os.path.join(output_dir, f"prediction_{idx:03d}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")