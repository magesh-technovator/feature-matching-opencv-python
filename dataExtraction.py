# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:40:26 2019

@author: MAGESHWARAN
"""
import os
from urllib import request
import tarfile
from tqdm import tqdm


def createFolders(base_folder, folder_name):
    """
        Creates New folder to store data

    Input:
        base_folder : path of the base file

        folder_name: Name of the folder to be created

    Output:
        Returns the path of new folder created

    """

    if not os.path.isdir(os.path.join(base_folder, folder_name)):
        os.mkdir(os.path.join(base_folder, folder_name))
        print("new ", folder_name, " created")

    folder = os.path.join(base_folder, folder_name)
    return folder

def getFiles(url, folder):
    """
        Downloads images.txt and crops.txt from internet

    Input:
        url : url of the file

        folder: Path of the folder where the files to be downloaded

    Output:
        Returns nothing

    """

    filename = url.split("/")[-1].strip()
    f = open(os.path.join(folder, filename), 'wb')
    f.write(request.urlopen(url).read())
    f.close()

def getUrls(txt_file, folder):
    """
        Extract the urls from images.txt and crops.txt

    Input:
        txt_file : txt file (images.txt or crops.txt)

        folder: Path of the txt file

    Output:
        Returns list of urls read from the input file

    """

    with open(os.path.join(folder, txt_file)) as imagefiles:
        urls = imagefiles.readlines()
    return urls

def downloadImages(url_list, folder):
    """
        Downloads images using the links provided in txt files

    Input:
        url_list : urls read from txt files

        folder: Path of the folder where the files to be downloaded

    Output:
        Returns nothing

    """

    for image in tqdm(url_list):
        filename = image.split("/")[-1].strip()
        f = open(os.path.join(folder, filename), 'wb')
        f.write(request.urlopen(image).read())
        f.close()

def getSampleTestset(sample_testset, folder):
    """
        Downloads sample_testset.tar and extract the data from it

    Input:
        sample_testset : list containing url and filename

        folder: Path of the folder where the files to be downloaded

    Output:
        Returns nothing

    """

    filename = os.path.join(folder, sample_testset[1])
    f = open(filename, "wb")
    f.write(request.urlopen(sample_testset[0]).read())
    f.close()

    if filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(folder)
        tar.close()

if __name__ == "__main__":

    # -------------------------- Create directories ---------------------------
    base_dir = os.getcwd()

    data_folder = createFolders(base_dir, "Dataset")

    images_folder = createFolders(data_folder, "Images")

    crops_folder = createFolders(data_folder, "Crops")

    # -------------------------- Download files -------------------------------

    files_url = ["https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/images.txt",
                 "https://s3.amazonaws.com/msd-cvteam/interview_tasks/crops_images_association_2/crops.txt"]


    getFiles(files_url[0], data_folder)
    getFiles(files_url[1], data_folder)

    # --------------------------- Download Dataset ----------------------------
    image_url = getUrls("images.txt", data_folder)

    crop_url = getUrls("crops.txt", data_folder)


    downloadImages(image_url, images_folder)
    downloadImages(crop_url, crops_folder)

    # ------------------------ Download Sample dataset ------------------------
    testset = ["https://bit.ly/2VoBYo1", "sample_testset.tar.gz"]
    getSampleTestset(testset, data_folder)
