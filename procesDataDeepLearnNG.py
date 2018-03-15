import numpy as np
import os
from numpy import genfromtxt
from scipy.io import savemat,loadmat
import matplotlib.pyplot as plt
from scipy import misc
from scipy.io import wavfile
import wavfile
import scipy.signal
from keras.utils import  np_utils
from python_speech_features import mfcc

from sklearn.cross_validation import train_test_split
from PIL import Image

import scipy

    
def load_data_goodNegSet():
    
    positiveEgsDir='./Data/'

    posImageList=os.listdir(positiveEgsDir)
    
    listAudios=[]

    #load all the .wav sudio files from the data folder
    listAudios = [str(x) for x in posImageList if '.wav' in x]
   
    #collect the class labels 
    #extract the labels from the filenames in the same order
    labels=[x[-5:-4] for x in posImageList if '.wav' in x]
           
    ## tranform the single label digits into label vectors 
    Y = np_utils.to_categorical(np.array(labels))  

    filenames=[]
    ALL_SOUNDS=[]
    SOUNDS = [] 
    
    
    
    for j,imname in enumerate(listAudios):    
        
        try :
            
            #Difficulties were encountered with reading all the files correctly, I did not find the best way to handle this but moved onto 
            # other issues
            rate,audiodata_1,bits=wavfile.read('./Data/' + str(listAudios[j]))
                
                
            ## Initial attempt :            
            #resmaple the raw audio data to a common size across all samples and to reduce dimensionality
            #This should enable the network to not overfit on the training dtaa , while at the same time 
            #still capturing the features and variation wihtin the data in order to learn from it.
                                    
            
            #audioDat_1_resampl=scipy.signal.resample(audiodata_1[1], 4096, t=None, axis=0, window=None)
            
            #sizeResamplData=audioDat_1_resampl.flatten(0)
            
            
            #extract the  Mel Frequency Cepstral Coefficients as features per audio example.
            # these are good feature to use in Speech recognition , thus I believe that they will be important here too (assumption with bird calls).
            
            # Also I have raised the default FFT size to avoid truncation of the raw data input
            mfcc_feat = mfcc(audiodata_1,rate,nfft=4096)            
            
            #As features vary in size i am choosing the most common yet larger output size (799*13) and padding all the smaller results
            # So that as little informationas possible is preseverved :
            # normally this shouold be more generalized, but i have fixed it here.
            #what I expect to result here is - num_examples matrix contaning the features per audio file. They will all be the same size 
            #and shall be passed in to the nueral Net. as one large array appended toge–––ther.
            result = np.zeros((800,14))
            
            if mfcc_feat.shape[0]*mfcc_feat.shape[1] < 10387:
                templteArr=np.array(result, copy=True)
                templteArr[:mfcc_feat.shape[0],:mfcc_feat.shape[1]] = mfcc_feat
            else:
                templteArr=mfcc_feat
                
            audioFeatures=misc.imresize(templteArr,size=(32,32))
        
            SOUNDS.append(np.expand_dims(audioFeatures,axis=0))               
            
        except  Exception as inst:
            print("Error occured at doing marker stuff of image file -  " + str(imname) + " ...")
            print(inst)   
            continue
        
        
    ALL_SOUNDS = np.concatenate(SOUNDS,axis=0) 
     
    
    #split data and labels simultaneously at random points  - into test and and training sets 
    idx = np.arange(len(ALL_SOUNDS))#ALL_SOUNDS.shape[0])
    idx_train,idx_test = train_test_split(idx,test_size=0.25)
    
    
    Xtrain = ALL_SOUNDS[idx_train,]
    ytrain = Y[idx_train,:]
    Xval = ALL_SOUNDS[idx_test,] 
    yval = Y[idx_test,:]
    return Xtrain,ytrain,Xval,yval



if __name__ == "__main__":
    #Preprocess()
    load_data_goodNegSet()