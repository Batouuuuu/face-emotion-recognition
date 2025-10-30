"""Throught the script we will make data augmentation and image processing such as greyscaling, reshaping and normalization,  
in order to store the data into a numpy array with the X = image matrice and Y = label (emotion)"""

import os
import random
import cv2
import numpy as np
from typing import Dict, List, Tuple
import pickle




def get_image_path(input_folder_path : os.PathLike) -> Dict[str, List[str]]:
    """Create a dictionnary with the label as key and List with the asociated path file as value
    example : {"sadness" :  [data/train\\surprise\\Training_98899707.jpg,..]}" """

    files_dict = {}
    folders = [f for f in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, f))]

    for folder in folders:
        folder_path = os.path.join(input_folder_path, folder)
        files_dict[folder] = [os.path.join(folder_path, f) for f in \
                              os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
   
    return files_dict
         

##Data augmentation because surprise and disgust are too low samples 
## surprise : 3171, disgust : 436 so we will make few operation too increase their number to 4000 for both of them
def increase_data(dictionnary_path: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Iterate through the dictionary and add more images for the surprise and disgust labels
       Returns the updated dictionary including augmented images.
    """
    minor_classes = ["surprise", "disgust"]
    target_sizes = {"surprise": 4000, "disgust": 4000}
    transformations = {
        rotation_image: "_rotated",
        translation_image: "_translated",
        increase_brightness: "_bright",
        image_flip: "_flipped"
    }

    original_images = {classe: dictionnary_path[classe].copy() for classe in minor_classes}

    for classe in minor_classes:
        dictionnary_images_augmented = []
        total_images = len(dictionnary_path[classe])


        while total_images < target_sizes[classe]:
            for i, path_image in enumerate(original_images[classe]):
                if total_images >= target_sizes[classe]:
                    break

                img = cv2.imread(path_image)
                ## all the simple transformations
                for func, suffix in transformations.items():
                    if total_images >= target_sizes[classe]:
                        break

                    new_image, path_save = apply_and_save(func, img, path_image, suffix)
                    dictionnary_images_augmented.append(path_save)
                    total_images += 1

                ## application of combination of transformation
                if total_images < target_sizes[classe]:
                    new_image, path_save = apply_and_save(None, img, path_image, "_combo", combination=True)
                    dictionnary_images_augmented.append(path_save)
                    total_images += 1

        
        dictionnary_path[classe].extend(dictionnary_images_augmented)

    ## verif
    for classe in minor_classes:
        for _ in dictionnary_path: 
            assert len(dictionnary_path[classe]) >= 4000, f"Not enought image for {classe} : {dictionnary_path[classe]}"

    return dictionnary_path

              
def apply_and_save(transformation_func, image, path_image, extension, combination=False):
    """Make transformation on the image (rotation, flip, translation, bright) and saving the new image"""
    
    ## we try to combinate differente fonction
    transformations = [rotation_image, translation_image, increase_brightness, image_flip]
    img_modif = image.copy()
    number_of_modification = random.randint(2,4) ## minimum 2 transformation combination
    random_transformations = random.sample(transformations,number_of_modification)

    ## if we chose to make transformation combination
    if combination == True:
       for func in random_transformations:
           img_modif = func(img_modif)
    else:
        ## case when we already chose a transformation
        img_modif = transformation_func(image)

    path_save = build_new_path(path_image, extension)
    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    cv2.imwrite(path_save, img_modif)
    return img_modif, path_save



def rotation_image(image: np.ndarray) -> np.ndarray:
    """Rotate the image with a random angle between 45 and -45 and saving it"""

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(-45, 45) 
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def translation_image(image: np.ndarray) -> np.ndarray:
    """Deplacing the image throught a translation """

    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    height, width = image.shape[:2]
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image


def increase_brightness(image: np.ndarray) -> np.ndarray:
    """Changing the brightness and the contrast of the image with random values"""

    brightness = random.uniform(5, 15)  
    contrast = random.uniform(1.5, 2)  
    bright_image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    return bright_image

def image_flip(image: np.ndarray) -> np.ndarray:
    "Invert an image like a mirror"

    flipped_image = cv2.flip(image, 1)
    return flipped_image



def build_new_path(path_image : os.PathLike, extension: str) -> os.PathLike:
    """Construction of the new path in order to save the image 
    input : path_image = data/train/surprise/Training_10013223.jpg
    output : data/train/surprise/aygmented/Training_10013223_{extension}.jpg"""

    file_name = os.path.basename(path_image)
    name, ext = os.path.splitext(file_name)
    augmented_folder = os.path.join(os.path.dirname(path_image), "augmented")
    os.makedirs(augmented_folder, exist_ok=True)
    new_name = name + extension + ext
    new_path = os.path.join(augmented_folder, new_name)
    return new_path 



def preprocess_images(dictionnary_path : Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply operations (greysclaling, resizing, normalisation
      on the images in order to make sure they will fit our model """
    X = []
    y = []

    for sentiment, image_list in dictionnary_path.items():
        for image_path in image_list:  
            image = cv2.imread(image_path) 
            grey_image = greyscaling(image)
            resize_image = resizing(grey_image)
            normalized_image = normalization(resize_image)
            X.append(normalized_image)
            y.append(sentiment)

    return np.array(X), np.array(y)


def greyscaling(image : np.ndarray) -> np.ndarray:
    """Image is already in grey but has 3 canals, we only want 2-d dimensional matrix"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resizing(image: np.ndarray) -> np.ndarray:
    """Make sur every image 48x48 pxl, if not resize"""
    if image.shape[:2] != (48,48):
        return cv2.resize(image, (48,48))
    else:
        return image

def normalization(image: np.ndarray) -> np.ndarray:
    """Return the matrix with number between 0 and 1 to reduce their weight"""
    return (image / 255.0).astype(np.float32)

def save_arrays(X: np.ndarray, y: np.ndarray, output_file: os.PathLike) -> None:
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)


def main():

    ##train_set
    train_images_path = get_image_path("data/train")
    complete_dico = increase_data(train_images_path)
    X_train_array, y_train_array = preprocess_images(complete_dico)
    save_arrays(X_train_array, y_train_array, "data/train_arrays.pkl")

    ##test_set
    test_images_path = get_image_path("data/test")
    X_test_array, y_test_array = preprocess_images(test_images_path)
    save_arrays(X_test_array, y_test_array, "data/test_arrays.pkl")



if __name__ == "__main__":
    main()