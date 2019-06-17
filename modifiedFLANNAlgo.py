# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:36:34 2019

@author: MAGESHWARAN
"""
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from tqdm import tqdm

base_dir = os.getcwd()
data_folder = os.path.join(base_dir, "Dataset")

images_folder = os.path.join(data_folder, "Images")
crops_folder = os.path.join(data_folder, "Crops")
sample_testset = os.path.join(data_folder, "sample_testset")

sample_json_result = os.path.join(sample_testset, "sample_result.json")
sample_images = os.path.join(sample_testset, "images")
sample_crops = os.path.join(sample_testset, "crops")

def ModifiedFLANN(img1, img2):
    
    mini_match_count = 10
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    
    search_params = dict(checks=10)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    orgBorder = None
    flannMatch = True
    
    if (des1 is None) or (des2 is None):
        flannMatch = False
        
        if (img2.shape[0] > img1.shape[0]) and (img2[1].shape[1] > img1.shape[1]):
            res = cv2.matchTemplate(img2,img1, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            h, w, c = img1.shape
            
            pts = [int(min_loc[0]), int(min_loc[0]) + w,
                   int(max_loc[0]), int(max_loc[1])]
            
            return flannMatch, pts

        return flannMatch, orgBorder
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    
    
    for match1, match2 in matches:
        if match1.distance < (0.7 * match2.distance):
            good_matches.append((match1))
    
    
    if len(good_matches) > mini_match_count:
        cropImg = []
        orgImg = []
        
        for m in good_matches:
            cropImg.append(kp1[m.queryIdx].pt)
            orgImg.append(kp2[m.trainIdx].pt)
        
        cropImg, orgImg = np.float32((cropImg, orgImg))
        
        H, status = cv2.findHomography(cropImg, orgImg, cv2.RANSAC, 3.0)

        if H is None:
            return flannMatch, orgBorder
        
        h, w, c = img1.shape
        
        cropBorder = np.float32([[[0,0], [0,h-1], [w-1,h-1], [w-1,0]]])
        
        orgBorder = cv2.perspectiveTransform(cropBorder, H)
        
        cv2.polylines(img2, [np.int32(orgBorder)], True, (0, 255, 0), 5)
   
    return flannMatch, orgBorder

def findMinMax(border):
    x, y = np.transpose(border)[0], np.transpose(border)[1]
    
    x1, x2 = int(x.min()), int(x.max())
    
    y1, y2 = int(y.min()), int(x.max())
    
    return [x1, y1, x2, y2]

completeTracker = {}

noAssociationCropImages = os.listdir(sample_crops)

for imagefile in tqdm(os.listdir(sample_images)):
    img = cv2.imread(os.path.join(sample_images,
                                   imagefile))
    imageTracker = []
    
    for cropfile in os.listdir(sample_crops):
        crop_img = cv2.imread(os.path.join(sample_crops,
                                           cropfile))

        flannMatch, crop_border = ModifiedFLANN(crop_img, img)
        
        if flannMatch:
            if crop_border is not None:
                pts = findMinMax(crop_border[0])
                imageTracker.append((cropfile.replace(".jpg", ""), pts))
                if cropfile in noAssociationCropImages:
                    noAssociationCropImages.remove(cropfile)
        else:
            if crop_border is not None:
                imageTracker.append((cropfile.replace(".jpg", ""), crop_border))
                if cropfile in noAssociationCropImages:
                    noAssociationCropImages.remove(cropfile)
    
    completeTracker[imagefile.replace(".jpg", "")] = imageTracker

NA_Crops = []

for crop in noAssociationCropImages:
    NA_Crops.append([crop.replace(".jpg", ""), []])
    
completeTracker["NA"] = NA_Crops

with open(sample_json_result, "w") as f:
    json.dump(completeTracker, f)


print("Output Json File is generated")