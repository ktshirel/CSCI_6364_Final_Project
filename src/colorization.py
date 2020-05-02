import os
from pathlib import Path
import argparse
from skimage import color, io, transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

#tf.debugging.set_log_device_placement(True)

def convert_to_lab(im):

    im = color.rgb2lab(im).astype(np.float32)
    return im

def process_dataset_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    shape = img.shape
    [img,] = tf.numpy_function(convert_to_lab, [img], [tf.float32])
    img.set_shape(shape)
    l = img[:,:,0:1]
    ab = img[:,:,1:3]
    return l, ab

def create_dataset(image_dir, shuffle_buffer_size=1000, batch_size=32):

    image_dir = Path(os.path.join(os.getcwd(), Path(image_dir)))
    ds = tf.data.Dataset.list_files(str(image_dir/'*.jpg'))
    ds = ds.map(process_dataset_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache("./resources/cache/wikimedia_art.tfcache")
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

def split_dataset(dataset, split_percentage):

    # Inspiration from 
    # https://stackoverflow.com/questions/59669413/what-is-the-canonical-way-to-split-tf-dataset-into-test-and-validation-subsets

    validation_num = round(split_percentage * 100)
    ds = dataset.enumerate()
    train_ds = ds.filter(lambda i, data: i % 100 > validation_num)
    validation_ds =  ds.filter(lambda i, data: i % 100 <= validation_num)
    
    train_ds = train_ds.map(lambda i, data: data)
    validation_ds = validation_ds.map(lambda i, data: data)

    return train_ds, validation_ds

def viz_dataset_batch(l, ab, in_color):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        im = np.zeros((224,224,3))
        im[:,:,0:1] = l[n]
        if in_color:
            im[:,:,1:3] = ab[n]
        im = color.lab2rgb(im)
        plt.imshow(im)
        plt.axis("off")

def train_model(data, save_dir):

    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,1)),
        layers.Conv2D(32, (3,3), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.Conv2D(64, (3,3), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.Conv2D(128, (3,3), strides=2, padding="same", activation="relu"),
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(128, (3,3), padding="same", activation="relu"),
        layers.Conv2DTranspose(128, (3,3), strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, (3,3), padding="same", activation="relu"),
        layers.Conv2DTranspose(64, (3,3), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (3,3), padding="same", activation="relu"),
        layers.Conv2DTranspose(32, (3,3), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(2, (3,3), padding="same", strides=1)
    ])

    SAMPLES = 40300
    TRAIN_TEST_SPLIT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 100
    
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    model.summary()
    
    train_data, validation_data = split_dataset(data, TRAIN_TEST_SPLIT)

    steps_per_epoch = round(SAMPLES * (1 - TRAIN_TEST_SPLIT)) // BATCH_SIZE
    validation_steps = round(SAMPLES * TRAIN_TEST_SPLIT) // BATCH_SIZE

    history = model.fit(train_data, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_data=validation_data, validation_steps=validation_steps)
    model.save(save_dir)

def process_input(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    shape = img.shape
    img = convert_to_lab(img)
    temp = np.zeros((1,224,224,1))
    temp[:,:,:,0:1] = img[:,:,0:1]
    img = tf.convert_to_tensor(temp, dtype=tf.float32)
    l = img[:,:,:,0:1]
    ab = img[:,:,:,1:3]
    return l, ab

def viz_output(l, ab, in_color):
    im = np.zeros((224,224,3))
    im[:,:,0:1] = l
    if in_color:
        im[:,:,1:3] = ab
    im = color.lab2rgb(im)
    plt.imshow(im)
    plt.axis("off")

def save_outpt(l, ab, fname):
    im = np.zeros((224,224,3))
    im[:,:,0:1] = l
    im[:,:,1:3] = ab
    long_dim = max(im.shape)
    scale = round(512 / long_dim, 3)
    height = im.shape[0] * scale
    width = im.shape[1] * scale
    img = transform.resize(im, (int(height), int(width)))
    im = color.lab2rgb(im)
    io.imsave("{}.png".format(fname), im)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="The image to colorize")
    parser.add_argument("--output", help="The output filename")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    model_dir = "./models"
    img_dir = "./images/Wikimedia_Art"

    #ds = create_dataset(img_dir)
    #train_model(ds, "{}/wikimedia_100".format(model_dir))

    model = tf.keras.models.load_model("{}/wikimedia_100".format(model_dir))

    history = model.history()
    input_l, input_ab = process_input(args.input)
    output_ab = model.predict(input_l, batch_size=1)
    save_outpt(input_l, output_ab, args.output)