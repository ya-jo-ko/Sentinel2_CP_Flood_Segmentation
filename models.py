import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def unet_s2(input_size1=(256,256,3), input_size2=(256,256,3), ts=0):
    inputs1 = Input(input_size1)
    conv1 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(inputs1)
    conv1 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(pool1)
    conv2 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(pool2)
    conv3 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    inputs2 = Input(input_size2)
    conv1_2 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(inputs2)
    conv1_2 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(conv1_2)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2_2 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(pool1_2)
    conv2_2 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(conv2_2)
    pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3_2 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(pool2_2)
    conv3_2 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(conv3_2)
    pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    merged = concatenate([pool3,pool3_2])

    conv5 = Conv2D(512, 3, activation = LeakyReLU() , padding = 'same')(merged)
    conv5 = Conv2D(512, 3, activation = LeakyReLU() , padding = 'same')(conv5)
    drop5 = Dropout(0.2)(conv5)

    up7 = Conv2D(256, 2, activation = LeakyReLU() , padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge7 = concatenate([conv3,conv3_2,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(merge7)
    conv7 = Conv2D(256, 3, activation = LeakyReLU() , padding = 'same')(conv7)

    up8 = Conv2D(128, 2, activation = LeakyReLU() , padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,conv2_2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = LeakyReLU() , padding = 'same')(conv8)

    up9 = Conv2D(64, 2, activation = LeakyReLU() , padding = 'same', )(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,conv1_2,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(merge9)
    conv9 = Conv2D(64, 3, activation = LeakyReLU() , padding = 'same')(conv9)
    conv9 = Conv2D(2, 3, activation = LeakyReLU() , padding = 'same')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model([inputs1,inputs2],conv10)

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    return model


def get_model(model_type, ts=0):
    if model_type == 'S1':
        return unet_s1(input_size1=(256, 256, 1), input_size2=(256, 256, 1), ts=ts)
    elif model_type == 'S2':
        return unet_s2(input_size1=(256, 256, 3), input_size2=(256, 256, 3), ts=ts)
    elif model_type == 'S1_S2':
        return unet_s1s2(
            input_size1=(256, 256, 3), input_size2=(256, 256, 3),
            input_size3=(256, 256, 1), input_size4=(256, 256, 1),
            ts=ts
        )
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Available options: 'S1', 'S2', 'S1_S2'")

