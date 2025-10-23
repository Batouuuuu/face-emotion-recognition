"""Throught the script we will make data augmentation and image processing such as greyscaling, reshaping and normalization
in order to store the data into a numpy array with the X = image matrice and Y = label (emotion)"""

import os
import random
import cv2
import numpy as np
from typing import Dict, List



def get_image_path(input_folder_path : os.PathLike) -> Dict[str, List[str]]:
    """Create a dictionnary with the label as key and List with the asociated path file as value
    example : {"sadness" :  [data/train\\surprise\\Training_98899707.jpg,..]}" """

    files_dict = {"": []}
    folders = [f for f in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, f))]

    for folder in folders:
        folder_path = os.path.join(input_folder_path, folder)
        files_dict[folder] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
   
    return files_dict
         

##Data augmentation because surprise and disgust are too low samples 
## surprise : 3171, disgust : 436 so we will make few operation too increase their number to 4000 for both of them
def increase_data(dictionnary_path: Dict[str, List[str]]):
    """Iterate throught the dictionnary and add more image for the surprise and disgust label"""
    
    minor_classes = ["surprise"] #"disgust"]
    target_sizes = {"disgust": 4000, "surprise": 4000} ## limite of the samples we want
    ## None key is for when we want the combination of many transformation (check apply and save parameter)
    transformations = {rotation_image : "_rotated", translation_image : "_translated",
                      increase_brightness: "_bright", image_flip: "_flipped"}
    
    for classe in minor_classes:
        while len(dictionnary_path[classe]) < target_sizes[classe]:
            for path_image in dictionnary_path[classe].copy():
                img = cv2.imread(path_image)
                for func, suffix in transformations.items():
                    if len(dictionnary_path[classe]) >= target_sizes[classe]:
                        break
                    image, path_save = apply_and_save(func, img, path_image, suffix)
                    apply_and_save(None, img, path_image, "_combo", True)
                    dictionnary_path[classe].append(path_save)

              
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
    "Reverse an image like a mirror"

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



    
def greyscaling():
    """"""
def resizing():
    """Every picture is suppose to be 48x48 pxl, we verify if not we rezise"""

def normalization():
    """"""


def main():
    dico = get_image_path("data/train")
    increase_data(dico)


if __name__ == "__main__":
    main()