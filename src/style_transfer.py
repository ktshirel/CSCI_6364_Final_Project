import cv2
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import argparse

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = 512 / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = img[tf.newaxis, :]
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc, bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / (num_locations)

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, inputs):
        inputs = inputs * 255.0
        processed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(processed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name:value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value for style_name, value in zip(self.style_layers, style_outputs)}
        return {"content": content_dict, "style": style_dict}

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)
    
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_in", help="The style image.")
    parser.add_argument("--content_in", help="The content image to be stylized.")
    parser.add_argument("--stylized_out", help="The filename for the stylized output.")
    parser.add_argument("--checkpoint_img", help="Whether to save images after each epoch to see progress.", default=True)
    parser.add_argument("--epochs", help="Epochs to run.", type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()

    style_image = load_image(args.style_in)
    content_image = load_image(args.content_in)

    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1"
    ]
    style_weight = 1e-2
    content_weight = 1e4
    epochs = args.epochs
    steps_per_epoch = 100
    total_variation_weight = 30

    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]
    img = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    step = 0
    start = time.time()

    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            print(".", end="", flush=True)
            train_step(img)
        if args.checkpoint_img and n < epochs - 1:
            if epochs > 10 and n % 10 == 0:
                tensor_to_image(img).save("{}-{}.png".format(args.stylized_out, n))
            elif epochs <= 10:
                tensor_to_image(img).save("{}-{}.png".format(args.stylized_out, n))
        check_point = time.time()
        delta = check_point - start
        print_time = "{:.1f} seconds".format(delta) if delta < 60 else "{:.1f} minutes".format(delta/60)
        print(" Epoch: {} - time elapsed: {}".format(n, print_time))

    tensor_to_image(img).save("{}.png".format(args.stylized_out))