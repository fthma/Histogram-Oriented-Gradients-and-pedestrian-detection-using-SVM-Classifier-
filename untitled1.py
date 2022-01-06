import numpy as np
import glob
import cv2

from numpy import linalg as LA
from sklearn.svm import SVC
import random


TestPositiveImages = [cv2.imread(file) for file in glob.glob("NICTA/TestSet/PositiveSamples/*.pnm")]
TestNegativeImages = [cv2.imread(file) for file in glob.glob("NICTA/TestSet/NegativeSamples/*.pnm")]

TrainPositiveImages = [cv2.imread(file) for file in glob.glob("NICTA/TrainSet/PositiveSamples/*.pnm")]
TrainNegativeImages = [cv2.imread(file) for file in glob.glob("NICTA/TrainSet/NegativeSamples/*.pnm")]

#function to get 16x16 overlapping blocks(sliding windows) or 8x8 cells 
def sliding_window(image,orientation, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]],orientation[y:y + windowSize[1], x:x + windowSize[0]])
            

            
#function to extract the HOG descriptor for the given image
def HOG_FeatureExtractor(I):
    
    hx=np.array([-1,0,1])
    hy=hx.reshape(3,1)
    
    
    
    (winW, winH) = (16, 16)
    HOGS=[]
    NormalizedHOG=[]
    
    I=cv2.resize(I,(128,64),interpolation = cv2.INTER_AREA)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = np.float32(I) / 255.0
    gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=1)
    #MagG=np.sqrt(np.power(gx,2) + np.power(gy,2))
    #ang=np.arctan2(gy,gx)
    #deg=(ang*180)/np.pi
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True) 
    
    for (x, y, window,orientation) in sliding_window(mag,angle, stepSize=8, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
       
        #code for hog
        Block_HOG=[]
    
        for (x, y, cell,cellOrient) in sliding_window(window,orientation, stepSize=8, windowSize=(8, 8)):
            if cell.shape[0] != 8 or cell.shape[1] != 8:
                continue
            bins=np.arange(0,200,20)
            h=np.histogram(cellOrient,bins,weights=cell)
            Block_HOG.extend(h[0])
           
        HOGS.append(Block_HOG)
    
    #normalizing the HOGs
    for hog in HOGS:
        l2norm=LA.norm(hog)
        l2normHog=hog/(l2norm+0.000001)
        NormalizedHOG.extend(l2normHog)
        
    return NormalizedHOG   


#function to extract the HOG descriptor for the all the images
def getHOGDescriptors(PositiveImages,NegativeImages):
    PostiveHOGDescriptors=[]
    NegativeHOGDescriptors=[]
    HOGDescriptors=[]
    TrainLabels=[]
    
    print("getting HOG for all positive samples")
    for image in PositiveImages:
        HOGDesc=HOG_FeatureExtractor(image)
        PostiveHOGDescriptors.append(HOGDesc)
        
    print("getting HOG for all negative samples")

    for image in NegativeImages:
        HOGDesc=HOG_FeatureExtractor(image)
        NegativeHOGDescriptors.append(HOGDesc)
        
    NegLabels=np.zeros(len(NegativeHOGDescriptors), dtype=int)
    PosLabels=np.ones(len(PostiveHOGDescriptors), dtype=int)
    
    TrainLabels=list(NegLabels)+list(PosLabels)
    HOGDescriptors=NegativeHOGDescriptors+PostiveHOGDescriptors
    
    #shuffling the positive and negative images
    #c=list(zip(HOGDescriptors,TrainLabels))
    #random.shuffle(c)
    
    #TrainData,TrainLabels=zip(*c)
    
    return list(HOGDescriptors),list(TrainLabels)


def SVM_Classification(TrainingData,TrainLabels,TestData,TestLabels,k='linear'):
    
    
    print("in the SVM classification function")
    from sklearn.metrics import confusion_matrix
    Train=np.asarray(TrainingData)
    Labels=np.asarray(TrainLabels)
    
    Test=np.asarray(TestData)
    TestLabels=np.asarray(TestLabels)
    
    clf=SVC(kernel=k,gamma='scale')
    clf.fit(Train,Labels)
    
    prediction=clf.predict(Test)
    
    #true positve, true negative, false positive, false negative
    tn, fp, fn, tp = confusion_matrix(TestLabels, prediction).ravel()
    #accuracy
    acc=(tp+tn)/(tp+tn+fp+fn)
    #false positive rate & miss rate
    fpr=fp/(fp+tn)
    missrate=fn/(tp+fn)
    
    print("For SVM classified with kernel={}".format(k))
    print("Accuracy:{}".format(acc))
    print("False positive rate:{}".format(fpr))
    print("Miss Rate:{}".format(missrate))
    
    
    return acc,fpr,missrate
    
#Extracting HOG FEatures for training and testing data
print("------------")
print("getting HOG for all training data")
TrainHOGDescriptors,TrainLabels=getHOGDescriptors(TrainPositiveImages,TrainNegativeImages)
print("------------")

print("getting HOG for all test data")
TestHOGDescriptors,TestLabels=getHOGDescriptors(TestPositiveImages,TestNegativeImages)
print("------------")
print("training the SVM classifier with different kernels")

#trainig SVM with linear kernel
acc,fpr,missrate=SVM_Classification(TrainHOGDescriptors,TrainLabels,TestHOGDescriptors,TestLabels)
print("------------")

#training the SVM with kernel=rbf
acc,fpr,missrate=SVM_Classification(TrainHOGDescriptors,TrainLabels,TestHOGDescriptors,TestLabels,k='rbf')
print("------------")

#training the SVM with kernel=polynomial
acc,fpr,missrate=SVM_Classification(TrainHOGDescriptors,TrainLabels,TestHOGDescriptors,TestLabels,k='poly')