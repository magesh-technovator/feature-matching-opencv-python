# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:10:19 2019

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

def bruteForceSIFT(img1, img2):
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    print(len(matches))
    for match1, match2 in matches:
        if match1.distance < 0.75 * match2.distance:
            good.append([match1])
            
    sift_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    
    sift_matches = cv2.cvtColor(sift_matches, cv2.COLOR_BGR2RGB)
    
    print(len(good))
    plt.imshow(sift_matches)
    plt.show()
    return (kp1, kp2)

def bruteForceORB(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    orb_matches = cv2.drawMatches(img1, kp1, img2, kp2,matches[:10], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   
    orb_matches = cv2.cvtColor(orb_matches, cv2.COLOR_BGR2RGB)
    plt.imshow(orb_matches)
    plt.show()
    
    return matches[:10]

def FLANN(img1, img2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    
    search_params = dict(checks=10)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    matchesMask = [[0,0] for i in range(len(matches))]

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
    
    flann_matches = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    
    orb_matches = cv2.cvtColor(flann_matches, cv2.COLOR_BGR2RGB)
    plt.imshow(orb_matches)
    plt.show()
    
    return matches[:10]

def templateMatching(img1, img2, method):
    w, h, c = img1.shape
    img = img2.copy()
    
    res = cv2.matchTemplate(img, img1, eval(method))
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = min_loc
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img,top_left, bottom_right, 255, 2)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    
    return top_left, bottom_right
 
    
def ModifiedFLANN(img1, img2):
    
    mini_match_count = 10
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    
    search_params = dict(checks=10)
    
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    orgBorder = None
    
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


crop_img = cv2.imread(os.path.join(sample_crops,
                                   "3ef28b3a-cdd5-548b-8134-7c69393c89f2.jpg"))

img = cv2.imread(os.path.join(sample_images,
                                   "ea9b55cc-e010-5905-b696-9f8bacc50fe6.jpg"))

plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
plt.show()
# =============================================================================
# 
# matches_sift = bruteForceSIFT(crop_img, img)
# 
# matches_orb = bruteForceORB(crop_img, img)
# 
# matches_flann = FLANN(crop_img, img)
# 
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# 
# for method in methods:
#     x, y = templateMatching(crop_img, img, method)
# =============================================================================

crop_border = ModifiedFLANN(crop_img, img)