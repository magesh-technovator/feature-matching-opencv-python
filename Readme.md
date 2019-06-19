# Solutions to interview problems of MadStreetDen

## Approach:
Initially by visualizing the associations in sample test set and examining the cropped images, I have decided to go with Feature Matching(using keypoints) techinques instead of template matching. Following are the various methods I tried out with few samples that are manually picked from the test set:
1. BruteForce Matching with ORB # performed well but not very accurate compared to others
2. BruteForce Matching(knn) with SIFT # Very accurate
3. FLANN based Matching with SIFT Descriptors # Very accurate and Faster method.
4. **FLANN based Matching with SIFT Descriptors and Template Matching** for Samples where SIFT fails to detect keypoints(*Model Currently Used for ImageCrop Association*) # Better in terms of Accuracy and Faster than BruteForce Methods.

After Iterating over the sample testset with above methods, I found the last one to be more efficient.

## Algorithm:
<modifiedFLANNAlgoWithTemplateMacthing.py>
### Building of Algorithm Step-by-Step:
1. It uses **SIFT** method to get keypoint and Descriptors.
2. In some cases(solid color image) there will not be an image gradient, SIFT fails to find keypoints. So, I tried with template matching for these cases.
3. Then a **FLANN based KNN Matching** is done with default parameters and k=2 for KNN.
4. Best Features are selected by Ratio test based on Lowe's paper.
5. To detect the Four Keypoints, I spent some time in Understanding the **keypoints object and DMatch Object** with opencv documentations and .cpp files in opencv library.
6. I have used following DMatch attributes *DMatch.trainIdx* and *DMatch.queryIdx* and used those indexes in keypoints Object to get the points
7. Find Homography method with Random Sample Consensus(RANSAC with threshold=3.0)
