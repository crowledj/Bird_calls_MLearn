from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adam
import numpy as np
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import scipy.stats
#import librosa


# For use of audio - MFCC feature extraction
#from scikits.talkbox.features import mfcc
    
# Import data
#from florPlanPreProcGoodNegs import load_data_goodNegSet
from procesDataDeepLearnNG import load_data_goodNegSet

Xtrain,ytrain,Xval,yval  = load_data_goodNegSet()


#Normalize the data so as to treat all samples fairly within the Net's training
Xtrain=scipy.stats.mstats.zscore(Xtrain.astype('float'), axis=0)
Xval = scipy.stats.mstats.zscore(Xval.astype('float'), axis=0)


#I fixed the dimensionality to be the size of the initial convoutional layer - 
# which is also nicely a power 2.
img_dimens_l=32
img_dimens_r=32


# The audio data of bird calls has a sampling frequency of 44,100 Hz.


# I am choosing a ConvNet with two hidden layes as my ML model
# the window sizes in each layer are chosen to shrink the feature data down as it reaches the final classification layer



#In practise and with more time I would split the complete dataset into three parts  - and to validate on a 'external' dataset
#I would also probably try to examine the Signal to noise ratios in order to remove any backround noise which could interfere with classification 
# and remove this possible with a filtering or wavelet noise removal technique. 
#in addition , I would like to play around with different resampling sizes to increase efficiency of the algorithm due to redundancy.

#I initially , decided to use the raw audio data (downsized by resampling ) as input 'featues' for my network, but with unimpressive results 
#I looked for more descriptive , qualitative derivatives of the audio data , such as a spectrogram.


#MFCC features of the Audio are most descriptive 


def VGG_16_local(weights_path=None):
    img_input = Input(shape=(img_dimens_l,img_dimens_r,1))
    # Block 1
    num_classes=10
    
    x = Conv2D(32, (9, 9), activation='relu', padding='same',name='block1_conv1')(img_input)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    x = Conv2D(64, (9, 9), activation='relu', padding='same',name='block2_conv1')(x)
    x= MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
   
    #try to tackle overfitting somewhat using a reduced dropout procedure
    x = Dropout(0.25)(x)
    

    x=Flatten()(x)
    x=Dense(128,activation='relu')(x)    
    x=Dense(num_classes,activation='softmax')(x)
        
    model = Model(img_input, x, name='NgTest_CNN')    
        
    print(model.summary())
    return model


if __name__ == "__main__":
    
    model = VGG_16_local()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    mcp = ModelCheckpoint(filepath='./best_model_DeepLearnTest_2.hdf5', verbose=1,monitor='val_loss',save_best_only=True)
    
    batch_size = 12
    epochs = 32
    Xtrain=Xtrain.reshape(Xtrain.shape[0],img_dimens_l,img_dimens_r,1)
    Xval=Xval.reshape(Xval.shape[0],img_dimens_l,img_dimens_r,1)
    
    history = model.fit(Xtrain, ytrain,
                        shuffle=True,
                        batch_size=batch_size, 
                        epochs=epochs ,
                        verbose=1, callbacks=[mcp],
                        validation_data=(Xval,yval))
    
    
    print(history)