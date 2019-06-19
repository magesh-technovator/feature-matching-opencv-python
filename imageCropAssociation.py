# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:53:37 2019

@author: MAGESHWARAN
"""

import os
import sys
import configparser
import json
import cv2
from tqdm import tqdm
from dataExtraction import (createFolders, getFiles, getUrls, downloadImages,
                            getSampleTestset)
from modifiedFLANNAlgoWithTemplateMacthing import (ModifiedFLANN, findMinMax)


if __name__ == "__main__":

    base_dir = os.getcwd()

    # reading config file
    config = configparser.ConfigParser()
    config.read("modelConfig.inf")

    section = config.sections()

    # storing section names into a list
    dataset = config.options(section[0])
    filepath = config.options(section[1])
    algorithm = config.options(section[2])

    # ---------------------- Dataset Extrcation -------------------------------
    dataAvailable = config.getboolean(section[0], dataset[0])

    if not dataAvailable:
        print("Creating Directories!!!")
        # -------------------------- Create directories -----------------------

        data_folder = createFolders(base_dir, "Dataset")

        images_folder = createFolders(data_folder, "Images")

        crops_folder = createFolders(data_folder, "Crops")

        # -------------------------- Download files ---------------------------
        print("Downloading images.txt and crops.txt")

        files_url = ["https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/images.txt",
                     "https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/crops.txt"]


        getFiles(files_url[0], data_folder)
        getFiles(files_url[1], data_folder)

        # --------------------------- Download Dataset ------------------------
        image_url = getUrls("images.txt", data_folder)

        crop_url = getUrls("crops.txt", data_folder)

        print("Downloading Real Dataset and Test Dataset!!!")
        print("Please Wait this may take few mins depending on your internet connection")
        downloadImages(image_url, images_folder)
        downloadImages(crop_url, crops_folder)


        # ------------------------ Download Sample dataset --------------------
        sample_testset = ["https://bit.ly/2VoBYo1", "sample_testset.tar.gz"]
        getSampleTestset(sample_testset, data_folder)

    else:
        images_folder = config.get(section[1], filepath[0])
        crops_folder = config.get(section[1], filepath[1])

    modelResult = config.get(section[1], filepath[2])
    useTemplateMacthing = config.get(section[2], algorithm[0])

    # --------------------- FLANN Based Feature Detection ---------------------
    completeTracker = {}

    if not (os.path.isdir(images_folder) and os.path.isdir(crops_folder)):
        print("Please enter valid folder path in Config File")
        sys.exit()

    noAssociationCropImages = os.listdir(crops_folder)
    noAssociationImages = os.listdir(images_folder)

    for imagefile in tqdm(os.listdir(images_folder)):
        img = cv2.imread(os.path.join(images_folder, imagefile))
        imageTracker = []

        for cropfile in os.listdir(crops_folder):
            crop_img = cv2.imread(os.path.join(crops_folder,
                                               cropfile))

            # Using FLANN Match with SIFT Descriptor to find Association
            flannMatch, crop_border = ModifiedFLANN(crop_img, img,
                                                    useTemplateMacthing)

            if flannMatch:
                if crop_border is not None:
                    # Detect min and max values of x and y from bounding box
                    pts = findMinMax(crop_border[0])

                    # keep track of crop association for a particular image
                    imageTracker.append((cropfile.replace(".jpg", ""), pts))

                    # Remove associated images from the list
                    # helps us to record NA Crops
                    if cropfile in noAssociationCropImages:
                        noAssociationCropImages.remove(cropfile)

            else:
                if crop_border is not None:
                    # keep track of crop association for a particular image
                    imageTracker.append((cropfile.replace(".jpg", ""),
                                         crop_border))

                    # Remove associated images from the list
                    # helps us to record NA Crops
                    if cropfile in noAssociationCropImages:
                        noAssociationCropImages.remove(cropfile)

        # Store all the CropImages associated to an image in a dictionary
        completeTracker[imagefile.replace(".jpg", "")] = imageTracker


    NA_Crops = []

    for crop in noAssociationCropImages:
        NA_Crops.append([crop.replace(".jpg", ""), []])

    # Store all the Unassociated images to dictionary with key: "NA"
    completeTracker["NA"] = NA_Crops

    modelResult = os.path.join(base_dir, modelResult)

    # Dump the Dictionary to a json file in the current working directory
    with open(modelResult, "w") as f:
        json.dump(completeTracker, f, indent=4, sort_keys=True)


    print("Output Json File is generated")
