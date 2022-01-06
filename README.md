# Histogram-Oriented-Gradients-and-pedestrian-detection-using-SVM-Classifier-

This project implements a HOG Feature extractor and makes use of the SVM classifier for 
pedestrian detection in the NICTA Pedestrian Dataset. The training set contains 1000 
positives samples (images contain pedestrians) and 2000 negatives samples (images do not 
contain pedestrians). The testing set includes 500 positive samples and 500 negative samples. In 
the implementation of HOG, the gradient(magnitude, orientation ) is found for the image. Then 
using a sliding window of 16x16 we get 105 overlapping blocks with 50 percent. For each of 
these blocks four 8x8 cells are taken and their histograms are calculated and concatenated . 
Finally, the concatenated Feature vector for all the blocks is obtained. This way the HOG 
descriptors for both the positive, negative samples of both the training and test data are found . 
The SVM classifier is trained on the training data with different kernels(linear, rbf, polynomial ) 
and the test data is tested on the classification model to depict the accuracy, false positive rate, 
and the miss rate. 
