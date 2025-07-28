import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_generators import * #get_generators_s1, get_generators_s2, get_generators_s1s2
from models import get_model
from tensorflow.keras.models import load_model
from keras.layers import LeakyReLU
from utils import compute_segmentation_metrics
import skimage.io as io
import pickle
from keras.utils import custom_object_scope

def load_labels(label_path):
    images = []
    img_list = os.listdir(label_path)
    img_list.sort()
    for filename in img_list:
        img = io.imread(os.path.join(label_path, filename)) / 255
        images.append(img)
    return images


def test_model(
    model_path: str,
    test_path1: str,
    test_path2: str,
    model_type: str,
    save_results_path: str,
    test_labels: None
):
    tf.random.set_seed(42)
    dataset_type = 'S1GFloods'
    if model_type == 'S1':
        test_gen = test_generator_s1(test_path1, dataset_type=dataset_type)
    elif model_type == 'S2':
        test_gen = test_generator_s2(test_path1, dataset_type=dataset_type)
    elif model_type == 'S1_S2':
        if not path_s1 or not path_s2:
            raise ValueError("For 'S1_S2' model, both path_s1 and path_s2 must be provided.")
        test_gen = test_generator_s1s2(path_s1, path_s2)
    else:
        raise ValueError("Unsupported model type. Choose from: 'S1', 'S2', 'S1_S2'")

    model = load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})
    print(f"Running predictions for model: {model_type}")

    '''
    with custom_object_scope({'LeakyReLU': LeakyReLU}):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    '''
    predictions = model.predict(test_gen)
    
    #y_pred =np.where(results[:,:,:,0]>.5,1,0)
    #predictions = np.concatenate(predictions, axis=0)
    print(f"Predictions completed. Shape: {predictions.shape}")


    # Binarize and evaluate
    #avg_preds = np.mean(predictions[..., 0], axis=0)
    binary_preds = np.where(predictions > 0.5, 1, 0).squeeze(-1)
    print(binary_preds.shape)
    print(len(test_labels))
    test_labels = np.array(test_labels)
    print((test_labels.shape))
    metrics = compute_segmentation_metrics(binary_preds, test_labels)
    print("\nSegmentation Metrics:")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_IoU']:.4f}")
    print(f"Frequency Weighted IoU: {metrics['frequency_weighted_IoU']:.4f}")



if __name__ == "__main__":    
    test_path1='/mnt/home/ikonidakes/ikonidakis/S1GFloods/test/'
    dataset_ = 'S1GFloods'
    if dataset_ == 'Ombria':
        test_labels = load_labels(os.path.join(test_path1, 'MASK'))
    elif dataset_ == 'S1GFloods':
        test_labels = load_labels(os.path.join(test_path1, 'label'))
        test_labels = [np.where(img > 0, 1, 0) for img in test_labels]
    else:
        raise ValueError("Unsupported dataset type. Use 'Ombria' or 'S1GFloods'.")

    test_model(
        model_path='./baseline_models/S1_fold1_ts1_20ep_20250703_102215/model_S1.keras', #'./saved_ICP_models/S1GFloods_models_ICP0.95_ep20_2025-07-03.pkl', #
        test_path1='/mnt/home/ikonidakes/ikonidakis/S1GFloods/test/',
        test_path2='/mnt/home/ikonidakes/ikonidakis/S1GFloods/test/',
        model_type='S1',
        save_results_path='S1_fold1_ts1_20ep_20250625_131823/results', #'./saved_ICP_models/results', #
        test_labels = test_labels
    )