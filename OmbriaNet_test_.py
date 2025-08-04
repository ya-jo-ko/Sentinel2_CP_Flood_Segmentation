import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_generators import test_generator_s2
from models import get_model
from tensorflow.keras.models import load_model
from keras.layers import LeakyReLU
from utils import compute_segmentation_metrics
import skimage.io as io


def load_labels(label_path):
    images = []
    img_list = sorted(os.listdir(label_path))
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255.0
        images.append(np.where(img > 0.5, 1, 0))
    return images


def test_model_s2(
    model_path: str,
    test_path: str,
    save_results_path: str,
    test_labels: list
):
    tf.random.set_seed(42)

    test_gen = test_generator_s2(test_path, dataset_type='Ombria')

    model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})
    print("Running predictions for model: S2")

    predictions = model.predict(test_gen)
    print(f"Predictions completed. Shape: {predictions.shape}")

    # Binarize predictions
    binary_preds = np.where(predictions > 0.5, 1, 0).squeeze(-1)
    test_labels = np.array(test_labels)

    print(f"Binary preds shape: {binary_preds.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    metrics = compute_segmentation_metrics(binary_preds, test_labels)

    print("\nSegmentation Metrics:")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_IoU']:.4f}")
    print(f"Frequency Weighted IoU: {metrics['frequency_weighted_IoU']:.4f}")

    os.makedirs(save_results_path, exist_ok=True)
    np.save(os.path.join(save_results_path, "predictions.npy"), predictions)
    np.save(os.path.join(save_results_path, "binary_predictions.npy"), binary_preds)
    np.save(os.path.join(save_results_path, "test_labels.npy"), test_labels)

    with open(os.path.join(save_results_path, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Results saved to: {save_results_path}")


if __name__ == "__main__":
    test_path = '/path/to/Ombria/test/'

    test_labels = load_labels(os.path.join(test_path, 'MASK'))

    test_model_s2(
        model_path='./baseline_models/S2_fold1_ts1_20ep_20250804/model_S2.keras',
        test_path=test_path,
        save_results_path='./baseline_models/S2_fold1_ts1_20ep_20250804/results',
        test_labels=test_labels
    )
