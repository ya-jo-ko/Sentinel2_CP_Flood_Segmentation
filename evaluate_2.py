import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from skimage import io
import tensorflow as tf
from keras.layers import LeakyReLU
#from keras.utils.generic_utils import get_custom_objects
from keras.utils import custom_object_scope

from data_generators import test_generator_s1, test_generator_s2, test_generator_s1s2
from utils import compute_segmentation_metrics

# --------------------------- Argument Parser ---------------------------
parser = argparse.ArgumentParser(description="Evaluate Trained Models on Test Set")
parser.add_argument('--model_path', type=str, required=True, help='Path to saved model pickle (.pkl)')
parser.add_argument('--model_path2', type=str, required=True, help='Path to saved scores pickle (.pkl)')
parser.add_argument('--dataset', type=str, required=True, choices=['S1GFloods', 'Ombria'])
parser.add_argument('--model_type', type=str, choices=['S1', 'S2', 'S1&S2'], default=None)
parser.add_argument('--test_path', type=str, required=True)
parser.add_argument('--test_path2', type=str, default=None)
parser.add_argument('--alpha', type=float, default=0.01)
args = parser.parse_args()

# Register LeakyReLU for deserialization
#get_custom_objects().update({'LeakyReLU': LeakyReLU})
# --------------------------- Load Model ---------------------------
with custom_object_scope({'LeakyReLU': LeakyReLU}):
    with open(args.model_path, 'rb') as f:
        models = pickle.load(f)

#with open(args.model_path, 'rb') as f:
#    models = pickle.load(f)

#with open(args.model_path.replace("models_", "scores_"), 'rb') as f:
#    validation_scores = pickle.load(f)
with open(args.model_path2, 'rb') as f:
    validation_scores = pickle.load(f)


# --------------------------- Prepare Test Dataset ---------------------------
if args.model_type == 'S1&S2':
    test_dataset = test_generator_s1s2(args.test_path, args.test_path2)
elif args.model_type == 'S2':
    test_dataset = test_generator_s2(args.test_path, dataset_type=args.dataset)
else:
    test_dataset = test_generator_s1(args.test_path, dataset_type=args.dataset)
    
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
def k_cv_cp_test(test_dataset, test_labels, models, validation_scores, alpha=0.1, nc_score='res'):
    prediction_intervals = []
    candidate_labels = np.unique(test_labels)
    print("Unique Labels: ", candidate_labels)
    test_labels_flat = np.array(test_labels).reshape(-1, 1)
    print("Labels flat shape: ",test_labels_flat.shape)
    prediction_set = np.zeros((len(test_labels_flat), len(candidate_labels)))

    all_val_scores = np.concatenate(validation_scores, axis=0)
    candidate_qs = np.linspace(0, np.max(all_val_scores), num=1000)
    min_valid_q = None

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
            predictions.append(y_pred)
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
            if nc_score == 'res':
                nc_score_k = np.abs(y_pred - label)
            else:
                raise ValueError("Invalid nc_score specified. Available option: 'res'")
            test_nc_scores.append(nc_score_k)

        test_av = np.min(test_nc_scores, axis=0)
        #test_av = np.mean(test_nc_scores, axis=0)
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

    return prediction_set, empirical_coverage, inefficiency, predictions


# --------------------------- Run Evaluation ---------------------------
if args.dataset == 'Ombria':
    test_labels = load_labels(os.path.join(args.test_path, 'MASK'))
elif args.dataset == 'S1GFloods':
    test_labels = load_labels(os.path.join(args.test_path, 'label'))
    test_labels = [np.where(img > 0, 1, 0) for img in test_labels]
else:
    raise ValueError("Unsupported dataset type. Use 'Ombria' or 'S1GFloods'.")

prediction_set, coverage, inefficiency, predictions = k_cv_cp_test(
    test_dataset=test_dataset,
    test_labels=test_labels,
    models=models,
    validation_scores=validation_scores,
    alpha=args.alpha,
    nc_score='res'
)


# Binarize predictions before evaluation
avg_preds = np.mean([pred[..., 0] for pred in predictions], axis=0)

# Binarize the averaged predictions
binary_preds = np.where(avg_preds > 0.5, 1, 0)

# Evaluate against ground truths
segm_metrics = compute_segmentation_metrics(binary_preds, test_labels)

print("\nSegmentation Metrics:")
print(f"Pixel Accuracy: {segm_metrics['pixel_accuracy']:.4f}")
print(f"Mean IoU: {segm_metrics['mean_IoU']:.4f}")
print(f"Frequency Weighted IoU: {segm_metrics['frequency_weighted_IoU']:.4f}")