"""Throught the script we will make data augmentation and image processing such as greyscaling, reshaping and normalization
and store the data into a numpy array with the X = image matrice and Y = label (emotion)"""

import os

import glob
import cv2
import numpy as np
from typing import Dict, List
from matplotlib import pyplot as plt


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
    """"""
    
    minor_classes = ["surprise", "disgust"]
    for classe in minor_classes:
        for element in dictionnary_path[classe]:
            img = cv2.imread(element)
            rotation_image()
            translation_image()
            increase_brightness()
            save_new_images()


def rotation_image():
    """"""
def translation_image():
    """"""
def increase_brightness():
    """"""
def save_new_images():
    """"""
    
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