import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from data_generators import get_generators_s2
from models import get_model
from sklearn.model_selection import KFold


def train_model(
    base_path: str,
    epochs: int = 20,
    batch_size: int = 8,
    k_folds: int = 1,
    save_base: str = './saved_models',
    ts: int = 0
):
    # Set seed for reproducibility
    tf.random.set_seed(42)

    # Get number of samples
    sample_folder = os.path.join(base_path, 'AFTER')
    num_samples = len(os.listdir(sample_folder))
    all_indices = np.arange(num_samples)

    # Setup train/val splits
    if k_folds > 1:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = list(kf.split(all_indices))
    else:
        val_split = int(num_samples * 0.01)
        splits = [(all_indices[val_split:], all_indices[:val_split])]

    os.makedirs(save_base, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold_idx+1}/{len(splits)} ---")

        # Load generators
        train_gen, val_gen = get_generators_s2(
            base_path,
            train_idx,
            val_idx,
            batch_size=batch_size,
            dataset_type='Ombria'
        )

        # Initialize model
        model = get_model('S2', ts=ts)

        steps_per_epoch = len(train_idx) // batch_size
        val_steps = len(val_idx) // batch_size

        print(f"Training for {epochs} epochs | {steps_per_epoch} steps/epoch | {val_steps} validation steps")

        model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_steps
        )

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(save_base, f"S2_fold{fold_idx+1}_ts{ts}_{epochs}ep_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model_S2.keras")
        model.save(model_path)
        print(f"Model saved at: {model_path}")


if __name__ == '__main__':
    # Example usage for Ombria S2
    train_model(
        base_path='/path/to/Ombria/train/',
        epochs=20,
        batch_size=8,
        k_folds=1,
        save_base='./baseline_models',
        ts=1
    )
