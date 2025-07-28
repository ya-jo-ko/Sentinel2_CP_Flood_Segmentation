from __future__ import print_function
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from skimage import io
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")

def get_generators_s1(base_path, train_idx, val_idx, img_height=256, img_width=256, batch_size=8, dataset_type='Ombria'):
    # Set directory names based on dataset
    if dataset_type == 'Ombria':
        dir11 = os.path.join(base_path, 'AFTER')
        dir12 = os.path.join(base_path, 'BEFORE')
        dir13 = os.path.join(base_path, 'MASK')
    elif dataset_type == 'S1GFloods':
        dir11 = os.path.join(base_path, 'A')
        dir12 = os.path.join(base_path, 'B')
        dir13 = os.path.join(base_path, 'label')
    else:
        raise ValueError("Unsupported dataset. Expected 'Ombria' or 'S1GFloods'.")

    
    after_images1 = sorted(os.listdir(dir11))
    before_images1 = sorted(os.listdir(dir12))
    mask_images1 = sorted(os.listdir(dir13))
    
    # Create lists of filenames for train and validation
    train_after1 = [after_images1[i] for i in train_idx]
    train_before1 = [before_images1[i] for i in train_idx]
    train_mask1 = [mask_images1[i] for i in train_idx]

    val_after1 = [after_images1[i] for i in val_idx]
    val_before1 = [before_images1[i] for i in val_idx]
    val_mask1 = [mask_images1[i] for i in val_idx]

    data_gen_args_for_training = dict(rotation_range=0.4, width_shift_range=0.05,height_shift_range=0.05,shear_range=0.2,
                    zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**data_gen_args_for_training)
    mask_image_datagen = ImageDataGenerator(**data_gen_args_for_training)

    # Create flow_from_directory or flow_from_dataframe generators dynamically
    def create_generator(generator, dir_path, file_list, color_mode, subset, batch_size):
        return generator.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': file_list}),
            subset = subset,
            directory=dir_path,
            x_col='filename',
            y_col=None,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode=color_mode,
            class_mode=None,
            shuffle=False
        )

    # Create training generators
    train_genX1 = create_generator(train_image_datagen, dir11, train_after1, 'grayscale', 'training', batch_size)
    train_genX2 = create_generator(train_image_datagen, dir12, train_before1, 'grayscale', 'training', batch_size)
    train_gen_mask = create_generator(mask_image_datagen, dir13, train_mask1, "grayscale", 'training', batch_size)

    # Create validation generators - 'training' tag does not matter, we do our own split
    val_genX1 = create_generator(train_image_datagen, dir11, val_after1, 'grayscale', 'training', batch_size)
    val_genX2 = create_generator(train_image_datagen, dir12, val_before1, 'grayscale', 'training', batch_size)
    val_gen_mask = create_generator(mask_image_datagen, dir13, val_mask1, "grayscale", 'training', batch_size)

    # Define generators that yield batches
    def fold_generator(genX1, genX2, gen_mask):
        while True:
            X1i = next(genX1) / 255.0
            X2i = next(genX2) / 255.0
            maski = next(gen_mask) / 255.0

            if dataset_type == 'Ombria':
                maski[maski > 0.5] = 1
                maski[maski <= 0.5] = 0
            else:
                maski[maski > 0] = 1 # S1GFloods masks have values 0 and 0.333

            # Convert all to tf.float32 tensors
            X1i = tf.convert_to_tensor(X1i, dtype=tf.float32)
            X2i = tf.convert_to_tensor(X2i, dtype=tf.float32)
            gen_maski = tf.convert_to_tensor(maski, dtype=tf.float32)
            yield ((X1i,X2i), gen_maski)  #Yield both images and their mutual label

    # Define input signature for TensorFlow dataset
    input_signature = (
        (
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32), 
        ),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32) # Mask
    )
    # Create the training dataset using the generator
    traingenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(train_genX1, train_genX2, train_gen_mask),
        output_signature=(input_signature)
    )
    # Create the validation dataset using the generator
    validationgenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(val_genX1, val_genX2, val_gen_mask),
        output_signature=(input_signature)
    )
    # Return training and validation generators
    return traingenerator, validationgenerator


def get_generators_s2(base_path, train_idx, val_idx, img_height=256, img_width=256, batch_size=8, dataset_type='Ombria'):
    if dataset_type == 'Ombria':
        dir11 = os.path.join(base_path, 'AFTER')
        dir12 = os.path.join(base_path, 'BEFORE')
        dir13 = os.path.join(base_path, 'MASK')
    else:
        raise ValueError("Unsupported dataset. Expected 'Ombria' or 'S1GFloods'.")

    
    after_images1 = sorted(os.listdir(dir11))
    before_images1 = sorted(os.listdir(dir12))
    mask_images1 = sorted(os.listdir(dir13))
    
    # Create lists of filenames for train and validation
    train_after1 = [after_images1[i] for i in train_idx]
    train_before1 = [before_images1[i] for i in train_idx]
    train_mask1 = [mask_images1[i] for i in train_idx]

    val_after1 = [after_images1[i] for i in val_idx]
    val_before1 = [before_images1[i] for i in val_idx]
    val_mask1 = [mask_images1[i] for i in val_idx]

    
    data_gen_args_for_training = dict(rotation_range=0.4, width_shift_range=0.05,height_shift_range=0.05,shear_range=0.2,
                    zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

    train_image_datagen = ImageDataGenerator(**data_gen_args_for_training)
    mask_image_datagen = ImageDataGenerator(**data_gen_args_for_training)

    # Create flow_from_directory or flow_from_dataframe generators dynamically
    def create_generator(generator, dir_path, file_list, color_mode, subset, batch_size):
        return generator.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': file_list}),
            subset = subset,
            directory=dir_path,
            x_col='filename',
            y_col=None,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode=color_mode,
            class_mode=None,
            shuffle=False
        )

    # Create training generators
    train_genX1 = create_generator(train_image_datagen, dir11, train_after1, 'rgb', 'training', batch_size)
    train_genX2 = create_generator(train_image_datagen, dir12, train_before1, 'rgb', 'training', batch_size)
    train_gen_mask = create_generator(mask_image_datagen, dir13, train_mask1, "grayscale", 'training', batch_size)

    # Create validation generators - 'training' tag does not matter, we do our own split
    val_genX1 = create_generator(train_image_datagen, dir11, val_after1, 'rgb', 'training', batch_size)
    val_genX2 = create_generator(train_image_datagen, dir12, val_before1, 'rgb', 'training', batch_size)
    val_gen_mask = create_generator(mask_image_datagen, dir13, val_mask1, "grayscale", 'training', batch_size)

    # Define generators that yield batches
    def fold_generator(genX1, genX2, gen_mask):
        while True:
            X1i = next(genX1) / 255.0
            X2i = next(genX2) / 255.0
            maski = next(gen_mask) / 255.0

            if dataset_type == 'Ombria':
                maski[maski > 0.5] = 1
                maski[maski <= 0.5] = 0
            else:
                maski[maski > 0] = 1 # S1GFloods masks have values 0 and 0.333

            # Convert all to tf.float32 tensors
            X1i = tf.convert_to_tensor(X1i, dtype=tf.float32)
            X2i = tf.convert_to_tensor(X2i, dtype=tf.float32)
            gen_maski = tf.convert_to_tensor(maski, dtype=tf.float32)
            yield ((X1i,X2i), gen_maski)  #Yield both images and their mutual label

    # Define input signature for TensorFlow dataset
    input_signature = (
        (
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32), 
        ),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32) # Mask
    )
    # Create the training dataset using the generator
    traingenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(train_genX1, train_genX2, train_gen_mask),
        output_signature=(input_signature)
    )
    # Create the validation dataset using the generator
    validationgenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(val_genX1, val_genX2, val_gen_mask),
        output_signature=(input_signature)
    )
    # Return training and validation generators
    return traingenerator, validationgenerator



#### ONLY FOR OMBRIA ####
def get_generators_s1s2(path1, path2, train_idx, val_idx, img_height=256, img_width=256, batch_size=8):
    # Define paths
    dir11 = os.path.join(path1, 'AFTER')
    dir12 = os.path.join(path1, 'BEFORE')
    dir13 = os.path.join(path1, 'MASK')
    dir21 = os.path.join(path2, 'AFTER')
    dir22 = os.path.join(path2, 'BEFORE')

    # Load file names
    after_images1 = sorted(os.listdir(dir11))
    before_images1 = sorted(os.listdir(dir12))
    mask_images1 = sorted(os.listdir(dir13))
    after_images2 = sorted(os.listdir(dir21))
    before_images2 = sorted(os.listdir(dir22))

    # Create lists of filenames for train and validation
    train_after1 = [after_images1[i] for i in train_idx]
    train_before1 = [before_images1[i] for i in train_idx]
    train_mask1 = [mask_images1[i] for i in train_idx]
    train_after2 = [after_images2[i] for i in train_idx]
    train_before2 = [before_images2[i] for i in train_idx]

    val_after1 = [after_images1[i] for i in val_idx]
    val_before1 = [before_images1[i] for i in val_idx]
    val_after2 = [after_images2[i] for i in val_idx]
    val_before2 = [before_images2[i] for i in val_idx]
    val_mask1 = [mask_images1[i] for i in val_idx]

    # Create data generators
    data_gen_args = dict(rotation_range=0.4, width_shift_range=0.05, height_shift_range=0.05,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    train_image_datagen = ImageDataGenerator(**data_gen_args)
    mask_image_datagen = ImageDataGenerator(**data_gen_args)


    # Create flow_from_directory or flow_from_dataframe generators dynamically
    def create_generator(generator, dir_path, file_list, color_mode, subset, batch_size):
        return generator.flow_from_dataframe(
            dataframe=pd.DataFrame({'filename': file_list}),
            subset = subset,
            directory=dir_path,
            x_col='filename',
            y_col=None,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode=color_mode,
            class_mode=None,
            shuffle=False
        )

    # Create training generators
    train_genX1 = create_generator(train_image_datagen, dir11, train_after1, 'rgb', 'training', batch_size)
    train_genX2 = create_generator(train_image_datagen, dir12, train_before1, 'rgb', 'training', batch_size)
    train_genX3 = create_generator(train_image_datagen, dir21, train_after2, 'grayscale', 'training', batch_size)
    train_genX4 = create_generator(train_image_datagen, dir22, train_before2, 'grayscale', 'training', batch_size)
    train_gen_mask = create_generator(mask_image_datagen, dir13, train_mask1, "grayscale", 'training', batch_size)


    # Create validation generators
    val_genX1 = create_generator(train_image_datagen, os.path.join(path1, 'AFTER'), val_after1, 'rgb', 'training', batch_size)
    val_genX2 = create_generator(train_image_datagen, os.path.join(path1, 'BEFORE'), val_before1, 'rgb', 'training', batch_size)
    val_genX3 = create_generator(train_image_datagen, os.path.join(path2, 'AFTER'), val_after2, 'grayscale', 'training', batch_size)
    val_genX4 = create_generator(train_image_datagen, os.path.join(path2, 'BEFORE'), val_before2, 'grayscale', 'training', batch_size)
    val_gen_mask = create_generator(mask_image_datagen, os.path.join(path1, 'MASK'), val_mask1, "grayscale", 'training', batch_size) # 1 for jk

    # Define generators that yield batches
    def fold_generator(genX1, genX2, genX3, genX4, gen_mask):
        while True:
            X1i = next(genX1) / 255.0
            X2i = next(genX2) / 255.0
            X3i = next(genX3) / 255.0
            X4i = next(genX4) / 255.0
            maski = next(gen_mask) / 255.0
            maski[maski > 0.5] = 1
            maski[maski <= 0.5] = 0
            # Convert all to tf.float32 tensors
            X1i = tf.convert_to_tensor(X1i, dtype=tf.float32)
            X2i = tf.convert_to_tensor(X2i, dtype=tf.float32)
            X3i = tf.convert_to_tensor(X3i, dtype=tf.float32)
            X4i = tf.convert_to_tensor(X4i, dtype=tf.float32)
            gen_maski = tf.convert_to_tensor(maski, dtype=tf.float32)
            yield ((X1i,X2i,X3i,X4i), gen_maski)  #Yield both images and their mutual label

    # Define input signature for TensorFlow dataset
    input_signature = (
        (
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),  # X1i: RGB
            tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),  # X2i: RGB
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),  # X1i: RGB
            tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),  # X2i: RGB
        ),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32)        # Mask
    )
    # maybe change batch size to 12 at some point...
    # Create the TensorFlow dataset using the generator
    inputgenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(train_genX1, train_genX2, train_genX3, train_genX4, train_gen_mask),
        output_signature=(input_signature)
    )
    # Create the TensorFlow dataset using the generator
    validationgenerator = tf.data.Dataset.from_generator(
        lambda: fold_generator(val_genX1, val_genX2, val_genX3, val_genX4, val_gen_mask),
        output_signature=(input_signature)
    )
    # Return training and validation generators
    return inputgenerator, validationgenerator

def test_generator_s1(base_test_path, dataset_type='Ombria'):

    def testGenerator(base_test_path, dataset_type='Ombria'):
        if dataset_type == 'Ombria':
            dir_before = os.path.join(base_test_path, 'BEFORE')
            dir_after = os.path.join(base_test_path, 'AFTER')
        elif dataset_type == 'S1GFloods':
            dir_before = os.path.join(base_test_path, 'B')
            dir_after = os.path.join(base_test_path, 'A')
        else:
            raise ValueError("Unsupported dataset type. Use 'Ombria' or 'S1GFloods'.")

        filenames_before = sorted(os.listdir(dir_before))
        filenames_after = sorted(os.listdir(dir_after))

        for i in range(len(filenames_before)):
            img_before = io.imread(os.path.join(dir_before, filenames_before[i]))
            img_after = io.imread(os.path.join(dir_after, filenames_after[i]))

            # For S1GFloods: images are RGB but only one channel is used
            if dataset_type == 'S1GFloods':
                img_before = np.expand_dims(img_before[:, :, 0], axis=-1)
                img_after = np.expand_dims(img_after[:, :, 0], axis=-1)
                img_before = tf.squeeze(img_before, axis=-1)
                img_after = tf.squeeze(img_after, axis=-1)

            # Normalize and reshape to batch dimension
            img_before = img_before / 255
            img_after = img_after / 255

            img_before = np.reshape(img_before, (1,) + img_before.shape)
            img_after = np.reshape(img_after, (1,) + img_after.shape)

            # Convert to tensors
            img_before = tf.convert_to_tensor(img_before, dtype=tf.float32)
            img_after = tf.convert_to_tensor(img_after, dtype=tf.float32)

            yield ((img_after, img_before),)

    # Define input signature for TensorFlow dataset
    output_signature = ((
        tf.TensorSpec(shape=(1, 256, 256), dtype=tf.float32),  # img_before
        tf.TensorSpec(shape=(1, 256, 256), dtype=tf.float32)   # img_after
    ),
    )
    # Create a tf.data.Dataset from the generator
    test_dataset = tf.data.Dataset.from_generator(
        lambda: testGenerator(base_test_path, dataset_type=dataset_type),
        output_signature=output_signature
    )
    return test_dataset


def test_generator_s2(base_test_path, dataset_type='Ombria'):
    def testGenerator(base_test_path, dataset_type='Ombria'):
        if dataset_type == 'Ombria':
            dir_before = base_test_path + '/AFTER'
            dir_after = base_test_path + '/BEFORE'
        else:
            raise ValueError("Unsupported dataset type. Use 'Ombria' or 'S1GFloods'.")

        filenames_before = os.listdir(dir_before)
        filenames_before.sort()
        filenames_after = os.listdir(dir_after)
        filenames_after.sort()

        for i in range(0,len(filenames_before)):
            img_before = io.imread(os.path.join(dir_before, filenames_before[i]))
            img_after = io.imread(os.path.join(dir_after, filenames_after[i]))

            # Normalize and reshape to batch dimension
            img_before = img_before / 255
            img_after = img_after / 255

            img_before = np.reshape(img_before, (1,) + img_before.shape)
            img_after = np.reshape(img_after, (1,) + img_after.shape)

            # Convert to tensors
            img_before = tf.convert_to_tensor(img_before, dtype=tf.float32)
            img_after = tf.convert_to_tensor(img_after, dtype=tf.float32)

            yield ((img_before, img_after),)

    # Define input signature for TensorFlow dataset
    output_signature = ((
        tf.TensorSpec(shape=(1, 256, 256, 3), dtype=tf.float32),  # img_before
        tf.TensorSpec(shape=(1, 256, 256, 3), dtype=tf.float32)   # img_after
    ),
    )
    # Create a tf.data.Dataset from the generator
    test_dataset = tf.data.Dataset.from_generator(
        lambda: testGenerator(base_test_path, dataset_type=dataset_type),
        output_signature=output_signature
    )
    return test_dataset

def test_generator_s1s2(test_path1, test_path2):
    def testGenerator(test_path1, test_path2):
        path_before1 =test_path1 + '/AFTER'
        path_after1 = test_path1 + '/BEFORE'
        path_before2 =test_path2 + '/AFTER'
        path_after2 = test_path2 + '/BEFORE'
        filename1 = os.listdir(path_before1)
        filename1.sort()
        filename2 = os.listdir(path_after1)
        filename2.sort()
        filename3 = os.listdir(path_before2)
        filename3.sort()
        filename4 = os.listdir(path_after2)
        filename4.sort()
        for i in range(0,len(filename1)):
            img_before1 = io.imread(os.path.join(path_before1,filename1[i]))
            img_after1 = io.imread(os.path.join(path_after1,filename2[i]))
            img_before2 = io.imread(os.path.join(path_before2,filename3[i]))
            img_after2 = io.imread(os.path.join(path_after2,filename4[i]))

            img_before1 = img_before1 / 255
            img_before1 = np.reshape(img_before1,(1,)+img_before1.shape)

            img_after1 = img_after1 / 255
            img_after1 = np.reshape(img_after1,(1,)+img_after1.shape)

            img_before2 = img_before2 / 255
            img_before2 = np.reshape(img_before2,(1,)+img_before2.shape)

            img_after2 = img_after2 / 255
            img_after2 = np.reshape(img_after2,(1,)+img_after2.shape)

            # Convert all to tf.float32 tensors
            img_before1 = tf.convert_to_tensor(img_before1, dtype=tf.float32)
            img_after1 = tf.convert_to_tensor(img_after1, dtype=tf.float32)
            img_before2 = tf.convert_to_tensor(img_before2, dtype=tf.float32)
            img_after2 = tf.convert_to_tensor(img_after2, dtype=tf.float32)
            yield ((img_before1, img_after1, img_before2, img_after2),)

    
    # Define input signature for TensorFlow dataset
    output_signature = ((
        tf.TensorSpec(shape=(1, 256, 256, 3), dtype=tf.float32),  # img_before
        tf.TensorSpec(shape=(1, 256, 256, 3), dtype=tf.float32),   # img_after
        tf.TensorSpec(shape=(1, 256, 256), dtype=tf.float32),  # img_before
        tf.TensorSpec(shape=(1, 256, 256), dtype=tf.float32)   # img_after
    ),
    )
    # Create a tf.data.Dataset from the generator
    test_dataset = tf.data.Dataset.from_generator(
        lambda: testGenerator(test_path1, test_path2),
        output_signature=output_signature
    )
    return test_dataset
