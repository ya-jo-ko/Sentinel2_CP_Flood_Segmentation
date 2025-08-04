import os
import argparse
import numpy as np
from skimage import io
import tensorflow as tf
from keras.layers import LeakyReLU
from keras.utils import custom_object_scope
import pickle
import matplotlib.pyplot as plt

from data_generators import test_generator_s2
from utils import compute_segmentation_metrics


def load_labels(label_path):
    images = []
    img_list = sorted(os.listdir(label_path))
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255.0
        images.append(np.where(img > 0.5, 1, 0))  # Binarize
    return images


def evaluate_icp(model_path, val_scores_path, test_dataset, test_labels, alpha, save_dir):
    with custom_object_scope({'LeakyReLU': LeakyReLU}):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    with open(val_scores_path, 'rb') as f:
        validation_scores = pickle.load(f)

    all_val_scores = np.concatenate(validation_scores).reshape(-1)
    n = len(all_val_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(all_val_scores, q_level, interpolation='higher')

    print(f"Computed q_hat from validation scores: {q_hat:.6f}")

    results = model.predict(test_dataset)
    results_vector = results.reshape(results.shape[0] * 256 * 256)
    probability_array = np.column_stack((1 - results_vector, results_vector))
    prediction_set = probability_array >= (1 - q_hat)

    test_labels_flat = np.array(test_labels).reshape(-1, 1)
    empirical_coverage_count = 0
    total_prediction_set_size = 0
    for idx in range(test_labels_flat.shape[0]):
        true_label = test_labels_flat[idx]
        if prediction_set[idx, int(true_label)] == 1:
            empirical_coverage_count += 1
        total_prediction_set_size += np.sum(prediction_set[idx, :])

    empirical_coverage = empirical_coverage_count / test_labels_flat.shape[0]
    inefficiency = total_prediction_set_size / test_labels_flat.shape[0]

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Empirical Coverage: {empirical_coverage:.4f}")
    print(f"Inefficiency: {inefficiency:.4f}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "prediction_sets.npy"), prediction_set)
    np.save(os.path.join(save_dir, "results.npy"), results)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Alpha: {alpha:.4f}\n")
        f.write(f"q_hat: {q_hat:.6f}\n")
        f.write(f"Empirical Coverage: {empirical_coverage:.4f}\n")
        f.write(f"Inefficiency: {inefficiency:.4f}\n")

    binary_preds = (results > 0.5).astype(np.uint8).squeeze()
    test_labels_np = np.array(test_labels)

    segm_metrics = compute_segmentation_metrics(binary_preds, test_labels_np)

    print("\nSegmentation Metrics:")
    print(f"Pixel Accuracy: {segm_metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {segm_metrics['mean_IoU']:.4f}")
    print(f"Frequency Weighted IoU: {segm_metrics['frequency_weighted_IoU']:.4f}")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ICP Model on Ombria S2 Test Set")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.pkl)')
    parser.add_argument('--score_path', type=str, required=True, help='Path to validation scores (.pkl)')
    parser.add_argument('--test_path', type=str, required=True, help='Path to Ombria test data')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./saved_icp_results/', help='Directory to save outputs')
    args = parser.parse_args()

    test_dataset = test_generator_s2(args.test_path, dataset_type='Ombria')
    test_labels = load_labels(os.path.join(args.test_path, 'MASK'))

    evaluate_icp(
        model_path=args.model_path,
        val_scores_path=args.score_path,
        test_dataset=test_dataset,
        test_labels=test_labels,
        alpha=args.alpha,
        save_dir=args.save_dir
    )
