# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:36:34 2019

@author: MAGESHWARAN
"""
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

base_dir = os.getcwd()
data_folder = os.path.join(base_dir, "Dataset")

images_folder = os.path.join(data_folder, "Images")
crops_folder = os.path.join(data_folder, "Crops")
sample_testset = os.path.join(data_folder, "sample_testset")

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
    
    if (des1 is None) or (des2 is None):
        return orgBorder
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

        h, w, c = img1.shape
        
        cropBorder = np.float32([[[0,0], [0,h-1], [w-1,h-1], [w-1,0]]])
        orgBorder = cv2.perspectiveTransform(cropBorder, H)
        
        cv2.polylines(img2, [np.int32(orgBorder)], True, (0, 255, 0), 5)
    
    else:
        print("No Good Matches Found")
    
    matched = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.imshow(matched)
    plt.show()
    
    return orgBorder

def findMinMax(border):
    x, y = np.transpose(border)[0], np.transpose(border)[1]
    
    x1, x2 = x.min(), x.max()
    
    y1, y2 = y.min(), x.max()
    
    return (x1, y1, x2, y2)

completeTracker = {}

allImages = os.listdir(sample_images)
allcropImages = os.listdir(sample_crops)

for imagefile in os.listdir(sample_images):
    img = cv2.imread(os.path.join(sample_images,
                                   imagefile))
    imageTracker = []
    
    for cropfile in os.listdir(sample_crops):
        crop_img = cv2.imread(os.path.join(sample_crops,
                                           cropfile))

        crop_border = ModifiedFLANN(crop_img, img)
    
        if crop_border is None:
            print("Images are Not associated")
        
        else:
            print("images are Associated")
            pts = findMinMax(crop_border[0])
            imageTracker.append((cropfile.replace(".jpg", ""), pts))
    
    completeTracker[imagefile.replace(".jpg", "")] = imageTracker
# =============================================================================
# plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
# plt.show()
# 
# =============================================================================

