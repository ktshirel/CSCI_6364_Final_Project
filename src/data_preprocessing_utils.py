import os
import shutil
from pathlib import Path
import csv
import cv2
from skimage import io, transform
import numpy as np
import warnings
from PIL import Image

warnings.simplefilter("error", Image.DecompressionBombWarning)

BAD_STYLES = [
    "Pointillism",
    "Cubism",
    "Native Art (Primitivism)",
    "Mechanistic Cubism",
    "Post-Painterly Abstraction",
    "Color Field Painting",
    "Abstract Art",
    "Fauvism",
    "Futurism",
    "Abstract Expressionism",
    "Synthetic Cubism",
    "Purism",
    "Pop Art",
    "Neo-Expressionism",
    "Ukiyo-e",
    "Art Informel",
    "Ink and wash painting",
    "Lyrical Abstraction",
    "Cloisonnism",
    "Metaphysical art",
    "Minimalism",
    "Hard Edge Painting",
    "SÅsaku hanga",
    "Analytical Cubism",
    "Analytical Realism",
    "Art Brut",
    "Action painting",
    "Automatic Painting",
    "Byzantine",
    "Cartographic Art",
    "Concretism",
    "Constructivism",
    "Costumbrismo",
    "Cubo-Expressionism",
    "Cubo-Futurism",
    "Dada",
    "Divisionism",
    "Existential Art",
    "Ilkhanid",
    "Indian Space painting",
    "Intimism",
    "Kinetic Art",
    "Kitsch",
    "Modernismo",
    "Muralism",
    "Nanga (Bunjinga)",
    "Neo-Byzantine",
    "Neoplasticism",
    "Nihonga",
    "Northern Renaissance",
    "Op Art",
    "Orphism",
    "Poster Art Realism",
    "Precisionism",
    "Primitivism",
    "Rayonism",
    "Renaissance",
    "Shin-hanga",
    "Socialist Realism",
    "Spectralism",
    "Suprematism",
    "Surrealism",
    "Synchromism",
    "Synthetism",
    "Tachisme",
    "Tonalism",
    "Tubism",
    "Zen"
]

def create_npy(input_dir, output_dir):

    total = len([os.scandir(input_dir)])
    batch_size = 1000
    steps = 100
    count = 0
    num_errs = 0

    out = []

    for im in os.scandir(input_dir):
        count += 1
        if count % batch_size == 0:
            print(" - {} processed.".format(count))
        elif count - 1 == 0 or count + 1 == total or count % steps == 0:
            print("=", end="", flush=True)
        try:
            im = np.float32(cv2.imread(im.path))/255
            im = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
            out.append(im)
        except:
            num_errs += 1
            continue

    np.save(out_name, np.asarray(out))
    print("Finish with {} errors.".format(num_errs))

def make_art_dataset():

    train_image_dir = "/mnt/c/Users/Keegan/Downloads/train/train"
    test_image_dir = "/mnt/c/Users/Keegan/Downloads/test/test"
    dest_dir = "/home/ktshirel/machine-learning/final-project/Wikimedia_Art"

    with open("./all_data_info.csv") as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            date = row["date"].replace("c.", "")
            date = date.split(" ")[1] if date.find(" ") > 0 else date
            date = "0" if date.isalpha() or date == "" else date
            genre = row["style"]
            in_train = True if row["in_train"] == "True" else False
            fname = row["new_filename"]
            if int(float(date)) >= 1800 and int(float(date)) <= 1945:
                if genre not in BAD_STYLES:
                    if in_train:
                        image_path = Path(train_image_dir, fname)
                    elif not in_train:
                        image_path = Path(test_image_dir, fname)
                    dest = Path(dest_dir, fname)
                    if fname not in os.listdir(dest_dir):
                        shutil.move(image_path, dest)


def remove_bw_images():

    # Inspiration from
    # https://answers.opencv.org/question/173862/how-to-check-in-python-if-a-pixel-in-an-image-is-of-a-specific-color-other-than-black-and-white-for-grey-image/

    data_dir = "/home/ktshirel/machine-learning/final-project/Wikimedia_Art"
    bw_dir = "/home/ktshirel/machine-learning/final-project/Wikimedia_Art/bw_images"

    for image in os.scandir(data_dir):
        name = image.name
        image_path = image.path
        image = np.float32(cv2.imread(image_path))/255
        b,g,r = image[:,:,0], image[:,:,1], image[:,:,2]

        pix_count = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r = image[i,j,2]
                g = image[i,j,1]
                if abs(r-g) < 0.05:
                    pix_count += 1
        
        ratio = pix_count / (image.shape[0] * image.shape[1])
        if ratio > 0.9:
            shutil.move(image_path, Path(bw_dir, name))
            print(name)
    
def remove_broken_images():

    # Inspiration from
    # https://stackoverflow.com/questions/33548956/detect-avoid-premature-end-of-jpeg-in-cv2-python

    image_dir = "./images/Wikimedia_Art"
    broken_dir = "./images/Wikimedia_Art_broken_images"

    for im in os.scandir(image_dir):
        im_path = im.path
        im_name = im.name
        try:
            im = Image.open(im_path)
            im.verify()
            with open(im_path, "rb") as fhandle:
                data = fhandle.read()
                if data.startswith(b"BM") or data[-2:] != b"\xff\xd9":
                    #print("Bad file: {}".format(im_name))
                    shutil.move(im_path, Path(broken_dir, im_name))
        except (IOError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            print("Bad file: {}".format(im_name))
            shutil.move(im_path, Path(broken_dir, im_name))


#### Postprocessing

def resize_style_image():

    img_path = "./images/treasury_style_reference.jpg"
    img = io.imread(img_path)
    long_dim = max(img.shape)
    scale = round(512 / long_dim, 3)
    height = img.shape[0] * scale
    width = img.shape[1] * scale
    img = transform.resize(img, (int(height), int(width)))
    io.imsave("resized.jpg", img)

def resize_images_for_typesetting():
    
    image_dir = "./images/images_for_publication"

    for image in os.scandir(image_dir):
        img = io.imread(image.path)
        long_dim = max(img.shape)
        scale = round(512 / long_dim, 3)
        height = img.shape[0] * scale
        width = img.shape[1] * scale
        img = transform.resize(img, (int(height), int(width)))
        io.imsave(os.path.join(os.getcwd(), image.name), img)

if __name__ == "__main__":

    resize_images_for_typesetting()