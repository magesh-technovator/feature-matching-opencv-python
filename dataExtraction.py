# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:40:26 2019

@author: MAGESHWARAN
"""
import os
from urllib import request
from tqdm import tqdm
import tarfile

def createFolders(base_folder, folder_name):
    if not os.path.isdir(os.path.join(base_folder, folder_name)):
        os.mkdir(os.path.join(base_folder, folder_name))
        print("new ", folder_name, " created")

    folder = os.path.join(base_folder, folder_name)
    return folder

def getFiles(url):
    filename = url.split("/")[-1].strip()
    f = open(os.path.join(data_folder, filename),'wb')
    f.write(request.urlopen(url).read())
    f.close()

def getUrls(txt_file):
    with open(os.path.join(data_folder, txt_file)) as imagefiles:
        urls = imagefiles.readlines()
    return urls

def downloadImages(url_list, folder):
    for image in tqdm(url_list):
        filename = image.split("/")[-1].strip()
        f = open(os.path.join(folder, filename),'wb')
        f.write(request.urlopen(image).read())
        f.close()

def getSampleTestset(sample_testset):
    filename = os.path.join(data_folder, sample_testset[1])
    f = open(filename, "wb")
    f.write(request.urlopen(sample_testset[0]).read())
    f.close()

    if filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(data_folder)
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


    getFiles(files_url[0])
    getFiles(files_url[1])

    # --------------------------- Download Dataset ----------------------------
    image_url = getUrls("images.txt")

    crop_url = getUrls("crops.txt")


    downloadImages(image_url, images_folder)
    downloadImages(crop_url, crops_folder)

    # ------------------------ Download Sample dataset ------------------------
    sample_testset = ["https://bit.ly/2VoBYo1", "sample_testset.tar.gz"]
    getSampleTestset(sample_testset)