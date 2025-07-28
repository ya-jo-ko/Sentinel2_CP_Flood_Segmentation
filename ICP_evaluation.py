import os
import argparse
import numpy as np
from skimage import io
import tensorflow as tf
from keras.models import load_model
from keras.layers import LeakyReLU
from keras.utils import custom_object_scope
import pickle
import matplotlib.pyplot as plt


from data_generators import test_generator_s1, test_generator_s2, test_generator_s1s2
from utils import compute_segmentation_metrics

def load_labels(label_path):
    images = []
    img_list = os.listdir(label_path)
    img_list.sort()
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255
        images.append(img)
    return images

def evaluate_icp(model_path, val_scores_path, test_dataset, test_labels, alpha, save_dir):
    # Load model from pickle
    with custom_object_scope({'LeakyReLU': LeakyReLU}):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    # Load validation nonconformity scores
    with open(val_scores_path, 'rb') as f:
        validation_scores = pickle.load(f)
    all_val_scores = np.concatenate(validation_scores).reshape(-1)

    # Calculate quantile and threshold
    n = len(all_val_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(all_val_scores, q_level, interpolation='higher')

    print(f"Computed q_hat from validation scores: {q_hat:.6f}")

    # Predict
    results = model.predict(test_dataset)
    results_vector = results.reshape(results.shape[0] * 256 * 256)
    probability_array = np.column_stack((1 - results_vector, results_vector))

    # Form prediction sets
    prediction_set = probability_array >= (1 - q_hat)

    test_labels_flat = np.array(test_labels).reshape(-1, 1)    
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
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "prediction_sets.npy"), prediction_set)
    np.save(os.path.join(save_dir, "results.npy"), results)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Alpha: {alpha:.4f}\n")
        f.write(f"q_hat: {q_hat:.6f}\n")
        f.write(f"Empirical Coverage: {empirical_coverage:.4f}\n")
        f.write(f"Inefficiency: {inefficiency:.4f}\n")
    
    # Binarize the predictions
    binary_preds = np.where(results > 0.5, 1, 0)
    binary_preds = np.squeeze(binary_preds, axis=-1)
    test_labels_ar = np.array(test_labels)

    # Evaluate against ground truths
    segm_metrics = compute_segmentation_metrics(binary_preds, test_labels_ar)

    print("\nSegmentation Metrics:")
    print(f"Pixel Accuracy: {segm_metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {segm_metrics['mean_IoU']:.4f}")
    print(f"Frequency Weighted IoU: {segm_metrics['frequency_weighted_IoU']:.4f}")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Evaluate Trained Models on Test Set")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model pickle (.pkl)')
    parser.add_argument('--model_path2', type=str, required=True, help='Path to saved scores pickle (.pkl)')
    parser.add_argument('--dataset', type=str, required=True, choices=['S1GFloods', 'Ombria'])
    parser.add_argument('--model_type', type=str, choices=['S1', 'S2', 'S1_S2'], default=None)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--test_path2', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()

    
    # --------------------------- Prepare Test Dataset ---------------------------
    if args.model_type == 'S1_S2':
        test_dataset = test_generator_s1s2(args.test_path, args.test_path2)
    elif args.model_type == 'S2':
        test_dataset = test_generator_s2(args.test_path, dataset_type=args.dataset)
    else:
        test_dataset = test_generator_s1(args.test_path, dataset_type=args.dataset)

    
    # Load test images and labels
    if args.dataset == 'Ombria':
        test_labels = load_labels(os.path.join(args.test_path, 'MASK'))
        test_labels = [np.where(img > 0.5, 1, 0) for img in test_labels]
    elif args.dataset == 'S1GFloods':
        test_labels = load_labels(os.path.join(args.test_path, 'label'))
        test_labels = [np.where(img > 0, 1, 0) for img in test_labels]
    else:
        raise ValueError("Unsupported dataset type. Use 'Ombria' or 'S1GFloods'.")

    save_dir = './saved_ICP_models/'
    evaluate_icp(
        model_path=args.model_path,
        val_scores_path=args.model_path2,
        test_dataset=test_dataset,
        test_labels=test_labels,
        alpha=args.alpha,
        save_dir=save_dir
    )
