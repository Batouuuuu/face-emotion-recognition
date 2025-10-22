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
         

##Data augmentation because surprise and disgust are too low enchantillons 
def increase_data(dictionnary_path: Dict[str, List[str]]):
    """Iterate throught the dictionnary and add more picture for the surprise and disgust label"""
    
    minor_classes = ["surprise"] #"disgust"]
    for classe in minor_classes:
        for path_image in dictionnary_path[classe]:
            img = cv2.imread(path_image)
            rotation_image(img, path_image)
            translation_image(img, path_image)
            new_image = increase_brightness(img, path_image)
            cv2.imshow("Displayed Image", new_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
   


def rotation_image(image, path_image : os.PathLike):
    """Rotate the image with a random angle between 45 and -45"""

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(-45, 45) 
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    new_path = build_new_path(path_image, "_rotated")
    cv2.imwrite(new_path, rotated_image)
    return rotated_image


def translation_image(image, path_image : os.PathLike):
    """"""

    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    height, width = image.shape[:2]
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    new_path = build_new_path(path_image, "_translated")
    cv2.imwrite(new_path, translated_image)
    return translated_image



def increase_brightness(image, path_image : os.PathLike):
    """Changing the brightness and the contrast of the image with random values"""
    brightness = random.uniform(5, 15)  
    contrast = random.uniform(1.5, 2)  
    bright_image = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
    new_path = build_new_path(path_image, "_bright")
    cv2.imwrite(new_path, bright_image)
    return bright_image


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
    """"""
def normalization():
    """"""


def main():
    dico = get_image_path("data/train")
    increase_data(dico)


if __name__ == "__main__":
    main()