import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_generators import get_generators_s1, get_generators_s2, get_generators_s1s2
from models import get_model


def train_model(
    model_type: str,
    dataset_type: str,
    base_path: str,
    path_s1: str = None,
    path_s2: str = None,
    epochs: int = 20,
    batch_size: int = 8,
    k_folds: int = 1,
    save_base: str = './saved_models',
    ts: int = 0
):
    # Set seed for reproducibility
    tf.random.set_seed(42)

    # Get data size from one of the folders
    sample_folder = os.path.join(base_path, 'AFTER' if dataset_type == 'Ombria' else 'A')
    num_samples = len(os.listdir(sample_folder))
    all_indices = np.arange(num_samples)

    # Initialize K-Fold or single split
    if k_folds > 1:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = list(kf.split(all_indices))
    else:
        val_split = int(num_samples * 0.01)
        splits = [(all_indices[val_split:], all_indices[:val_split])]

    # Create base model save path
    os.makedirs(save_base, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_idx+1}/{len(splits)} ---")

        # Create data generators based on model type
        if model_type == 'S1':
            train_gen, val_gen = get_generators_s1(base_path, train_idx, val_idx, batch_size=batch_size, dataset_type=dataset_type)
        elif model_type == 'S2':
            train_gen, val_gen = get_generators_s2(base_path, train_idx, val_idx, batch_size=batch_size, dataset_type=dataset_type)
        elif model_type == 'S1_S2':
            if not path_s1 or not path_s2:
                raise ValueError("For 'S1_S2' model, both path_s1 and path_s2 must be provided.")
            train_gen, val_gen = get_generators_s1s2(path_s1, path_s2, train_idx, val_idx, batch_size=batch_size)
        else:
            raise ValueError("Unsupported model type. Choose from: 'S1', 'S2', 'S1_S2'")

        model = get_model(model_type, ts=ts)

        steps_per_epoch = len(train_idx) // batch_size
        val_steps = len(val_idx) // batch_size

        print(f"Training for {epochs} epochs | {steps_per_epoch} steps/epoch | {val_steps} validation steps")

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps
        )

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_base, f"{model_type}_fold{fold_idx+1}_ts{ts}_{epochs}ep_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{model_type}.keras")
        model.save(model_path)
        print(f"Model saved at: {model_path}")


if __name__ == '__main__':
    # Example usage
    train_model(
        model_type='S1',
        dataset_type='S1GFloods',
        base_path='/mnt/home/ikonidakes/ikonidakis/S1GFloods/train_/',
        epochs=20,
        batch_size=8,
        k_folds=1,
        save_base='./baseline_models',
        ts=1
    )
