# Finding Patches of Query image associated to a set of training images using Feature Matching

## Approach:
Initially by visualizing the associations in sample test set and examining the cropped images, I have decided to go with Feature Matching(using keypoints) techinques instead of template matching. Following are the various methods I tried out with few samples that are manually picked from the test set:
1. BruteForce Matching with ORB. # performed well but not very accurate compared to others
2. BruteForce Matching(knn) with SIFT. # Very accurate
3. FLANN based Matching with SIFT Descriptors. # Very accurate and Faster method.
4. **FLANN based Matching with SIFT Descriptors and Template Matching** for Samples where SIFT fails to detect keypoints(*Model Currently Used for ImageCrop Association*). # Better in terms of Accuracy and Faster than BruteForce Methods.

After Iterating over the sample testset with above methods, I found the last one to be more efficient.

## Codebase Structure and Exectution:
1. **modelConfig.inf** facilitates **imagesCropAssociation.py**:
	- Gives option to set datasetAvailabiity in LocalMachine, if not available(set False) it triggers *dataExtraction.py*
	- Filepath of Images and Crops folder and Filename of the output json file
	- Options to use Template Matching or not.

2. **dataExtraction.py** has the code to automatically create directiories and download:
	* images.txt
	* crops.txt
	* sample_testset.tar and uncompress it
	* Dataset using urls from images.txt and crops.txt
	
3. **modifiedFLANNAlgoWithTemplateMacthing.py** has the core algorithm behind this task:
	- FLANN based Knn Matching with SIFT Descriptors,
	- Template Matching
	- Homography and Perspective transform
	- MinMax detection of bounding box

4. **imagesCropAssociation.py**:
	- **This is the file to be executed to run the algorithm and generate json ouput**
	- Verify the config file associated to this, before runnig this script.
	- If dataset is not locally available, set **datasetLocallyAvailable to False** in config file, model will automatically download it for you.
	
5. **evaluationConfig.inf**	facilitates **EvaluationMetrics.py**, has filepaths of ground truth json file and model outputted json file

6. **EvaluationMetrics.py**: Ground truth lables are comapred against the model generated output. It prints precision, Recall, F1_Score and Confusion Matrix.

7. **StaticCodeAnalysisReport** It has Static code analysis reports run with pylint for all the .py files.
	- invalid-name is disabled, since pylint follows UPPER Case Naming style.
	- no-member is disabled, since python fails to detect the components from opencv library.

8. **modelOutput.json** Model generated output file on the real Dataset.

9. **sampleOutput.json** Model generated output file on the Sample Dataset.

10. **requirements.txt** It has list of python packages and their versions, used for this project. 
	- **Makesure your python Installation comes os, json and configparser scripts.**
	- Code is written in Python3 and the version I have used is 3.6.5

## Algorithm:

### Building of Algorithm Step-by-Step:
**modifiedFLANNAlgoWithTemplateMacthing.py** --> ModifiedFLANN function
1. It uses **SIFT** method to get keypoint and Descriptors.
2. In some cases(solid color image) there will not be an image gradient, SIFT fails to find keypoints. So, I tried with template matching for these cases.
3. Then a **FLANN based KNN Matching** is done with default parameters and k=2 for KNN.
4. Best Features are selected by Ratio test based on Lowe's paper.
5. To detect the Four Keypoints, I spent some time in Understanding the **keypoints object and DMatch Object** with opencv documentations and .cpp files in opencv library.
6. I have used following DMatch attributes *DMatch.trainIdx* and *DMatch.queryIdx* and used those indexes in keypoints Object to get the points
7. Find Homography with Random Sample Consensus(RANSAC with threshold=3.0) to compute **Geometric Transformation Matrix**
8. **Perspective transform** to detect corners using the current view.
9. finMinMax function is used to detect min and max values of x and y.

## Merits and Other Possible Approaches:

### Merits:
1. Faster method compared to Sliding windows and BruteForce Matching when perfomed on larger datasets.
2. Very Accuracte in Predictin True Positives and Good Approximation of Image Boundaries.
3. Association of Crop images with Rotation and Shearing made easy.

### Other Possible Approaches:
1. To try with KAZE and AKAZE algorithms.
2. Use Sliding window technique.
3. Handling unassociated images(FN) with different set of algorithms.
4. Use specific algorithms for solid color images(easily be recognized with average pixel values or using thresholds)
5. Use Feature Maching with Bounding Distortion Algorithm[2].

## Accuracy Metrics and Error Analysis:

### F1_Score:
1. Associated Images are taken as TP and Unassociated images as well as Unassociated crop images are considered as TN.
2. Ground truth lables are comapred against the model generated output in <evaluationMetrics.py>. It outputs precision, Recall, F1_Score and Confusion Matrix.
3. I did AB testing for FLANN based matching with and without Template Matching with these metrics.

##### Observations:
* Precision:  0.41836734693877553
* recall:  0.6029411764705882
* F1 Score: 0.49397590361445787
* Confusion Matrix: [[ 82 114] [ 54   1]]
 
### Error Analysis:
1. Images with solid colors has no gradients and even templateMatching is not performing well on this.
2. Crop Images with black borders are not properly handled.
3. Homography fails to compute Transformation with image like(fc429b5f-429b-5b5a-8e10-93835f02db9d.jpg) reason might be Low Resolution or Radom salt and pepper noises in crop image.

## References:
[1]. [A Comparative Analysis of SIFT, SURF, KAZE, AKAZE, ORB, and BRISK](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8346440) <br />
[2]. [Feature Maching with Bounding Distortion Algorithm](http://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/lipman_et_al_-_feature_matching_with_bounded_distortion.pdf)
