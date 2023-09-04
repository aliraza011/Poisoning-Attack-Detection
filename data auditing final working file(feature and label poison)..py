#!/usr/bin/env python
# coding: utf-8

# In[13]:


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:45:06 2021

@author: ali.raza
"""



import socket
import pickle
import threading
import time

import pygad
import pygad.nn
import pygad.gann
import numpy

import pygad
import pygad.nn
import pygad.gann
import numpy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight
import warnings
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import tensorflow as tf
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential, load_model
import cv2
import h5py
#-----------------------------------------Datan Preparation--------------------------------------------------------------------------
warnings.filterwarnings('ignore')
train_df=pd.read_csv('mitbih_train.csv',header=None)
test_df=pd.read_csv('mitbih_test.csv',header=None)

df_1=train_df[train_df[188]==1]
df_2=train_df[train_df[188]==2]
df_3=train_df[train_df[188]==3]
df_4=train_df[train_df[188]==4]
df_0=(train_df[train_df[188]==0]).sample(n=20000,random_state=42)

df_1_upsample=resample(df_1,replace=True,n_samples=20000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=20000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])

dft_1=test_df[test_df[188]==1]
dft_2=test_df[test_df[188]==2]
dft_3=test_df[test_df[188]==3]
dft_4=test_df[test_df[188]==4]
dft_0=(test_df[test_df[188]==0]).sample(n=10000,random_state=42)

dft_1_upsample=resample(dft_1,replace=True,n_samples=10000,random_state=123)
dft_2_upsample=resample(dft_2,replace=True,n_samples=10000,random_state=124)
dft_3_upsample=resample(dft_3,replace=True,n_samples=10000,random_state=125)
dft_4_upsample=resample(dft_4,replace=True,n_samples=10000,random_state=126)

dft=pd.concat([dft_0,dft_1_upsample,dft_2_upsample,dft_3_upsample,dft_4_upsample])




def add_gaussian_noise(signal):
    noise=np.random.normal(0,10,187)
    out=signal+noise
    return out
target_train=df.iloc[:,-1]
target_test=dft.iloc[:,-1]

y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=df.iloc[:,:-1].values
X_test=dft.iloc[:,:-1].values
X_train_noise=np.array(X_train)
X_test_noise=np.array(X_test)
plt.plot(X_train[10])
plt.show()
for i in range(len(X_train_noise)):
    X_train_noise[i,:187]= add_gaussian_noise(X_train_noise[i,:187])
for i in range(len(X_test_noise)):
    X_test_noise[i,:187]= add_gaussian_noise(X_test_noise[i,:187])
    
X_train_noise= X_train_noise.reshape(len(X_train_noise), X_train_noise.shape[1],1)
X_train= X_train.reshape(len(X_train), X_train.shape[1],1)
X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
X_test_noise= X_test_noise.reshape(len(X_test_noise), X_test_noise.shape[1],1)


X_train=np.float32(X_train)
X_train_noise=np.float32(X_train_noise)
X_test=np.float32(X_test)
X_test_noise=np.float32(X_test_noise)

#normalize mean ok
X_train_noise=(X_train_noise-X_train_noise.mean())/X_train_noise.std()
X_test_noise=(X_test_noise-X_test_noise.mean())/X_test_noise.std()


def classifier_CNN(input_data):
    #encoder
    
    x = layers.Conv1D(55, kernel_size=3,activation='relu',padding='same', name='input')(input_data)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(43, kernel_size=3, activation='relu', padding='same')(x)
    x=layers.Flatten()(x)
    x=layers.Dense(68, activation='relu')(x)
    
    decoded=layers.Dense(5,  activation='softmax')(x)
    
    return decoded
#----------------------------------------------Model Compile---------------------------------------------------------------------
input_data = keras.Input(shape=(X_train_noise.shape[1], 1))
common_model = keras.Model(input_data, classifier_CNN(input_data))
opt = keras.optimizers.RMSprop(learning_rate=0.00001)
def scheduler(epoch, lr):
       if epoch < 40:
        return lr
       else:
        return lr * tf.math.exp(-0.1)
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
common_model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



    


# In[14]:


common_model.summary()

classify_train = common_model.fit(X_train,y_train,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[15]:


edge_model = keras.Model(input_data, classifier_CNN(input_data))
edge_model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
edge_model.summary()

classify_train = edge_model.fit(X_test,y_test,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[22]:


from random import seed
from random import randint
# seed random number generator
seed(1)
# generate some integers
for x in range(target_test.shape[0]):
	value = randint(0, 4)
	target_test[x]=value


# In[23]:


label_poisoning=to_categorical(target_test)


# In[24]:


dedge_model_poision = keras.Model(input_data, classifier_CNN(input_data))
dedge_model_poision.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
dedge_model_poision.summary()

classify_train = dedge_model_poision.fit(X_test_noise,label_poisoning,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[25]:


def Grad_cam(model, input_test,sample_number):
    
    array = np.array(input_test[sample_number])
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #print("array:",array.shape)   
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    #print("array:",array.shape) 
    predict = model.predict(array)
    target_class = np.argmax(predict[0])
   
    #print("Target Class = ", target_class, "corresponding to:", predict, "Obese is [0., 1.]")
    last_conv = model.get_layer('conv1d_3') #last_conv= model.layers[8]
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array) #get activations maps + predictions from last conv layer
        loss = predictions[:, target_class] # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)

    output = conv_outputs[0] #activations maps from last conv layer
    grads = tape.gradient(loss, conv_outputs) #function to obtain gradients from last conv layer

    #print("grads shape:", grads.shape)
    #print("Model output (loss for the target class):", loss.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    pooled_grad= tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs=conv_outputs.numpy()
    pooled_grad = pooled_grad.numpy()
    
    
    return conv_outputs,pooled_grad,loss,target_class,array
def Grad_cam2(model, input_test,sample_number):
    
    array = np.array(input_test[sample_number])
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #print("array:",array.shape)   
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    #print("array:",array.shape) 
    predict = model.predict(array)
    target_class = np.argmax(predict[0])
   
    #print("Target Class = ", target_class, "corresponding to:", predict, "Obese is [0., 1.]")
    last_conv = model.get_layer('conv1d_4') #last_conv= model.layers[8]
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array) #get activations maps + predictions from last conv layer
        loss = predictions[:, target_class] # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)

    output = conv_outputs[0] #activations maps from last conv layer
    grads = tape.gradient(loss, conv_outputs) #function to obtain gradients from last conv layer

    #print("grads shape:", grads.shape)
    #print("Model output (loss for the target class):", loss.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    pooled_grad= tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs=conv_outputs.numpy()
    pooled_grad = pooled_grad.numpy()
    
    
    return conv_outputs,pooled_grad,loss,target_class,array
def Grad_cam3(model, input_test,sample_number):
    
    array = np.array(input_test[sample_number])
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #print("array:",array.shape)   
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    #print("array:",array.shape) 
    predict = model.predict(array)
    target_class = np.argmax(predict[0])
   
    #print("Target Class = ", target_class, "corresponding to:", predict, "Obese is [0., 1.]")
    last_conv = model.get_layer('conv1d_7') #last_conv= model.layers[8]
    grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(array) #get activations maps + predictions from last conv layer
        loss = predictions[:, target_class] # the variable loss gets the probability of belonging to the defined class (the predicted class on the model output)

    output = conv_outputs[0] #activations maps from last conv layer
    grads = tape.gradient(loss, conv_outputs) #function to obtain gradients from last conv layer

    #print("grads shape:", grads.shape)
    #print("Model output (loss for the target class):", loss.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    #print("Output from lat conv layer", conv_outputs.shape)
    pooled_grad= tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs=conv_outputs.numpy()
    pooled_grad = pooled_grad.numpy()
    
    
    return conv_outputs,pooled_grad,loss,target_class,array


# In[26]:


def attack_data_generation(data,flag,classifier,grad):
    size=data.shape[0]
    for x in range(9000):
        output,grads,loss,target_class,array_out=grad(classifier,data,x)
        gradarray=grads
        outputs=output
        loss_array=loss.numpy()
        data_array=data[x]
        flatten_gradients=gradarray.reshape(1,-1)
        flatten_outputs=outputs.reshape(1,-1)
        flatten_loss=loss_array.reshape(1,-1)
        flatten_data=data_array.reshape(1,-1)
        newar=np.concatenate([flatten_data,flatten_outputs,flatten_loss],axis=1)
        
        if flag:
            dataset=newar
            #dataset=flatten_outputs
            flag=0
        else:
            dataset=np.concatenate([dataset,newar])
        
    return dataset


# In[27]:



data_common_model=attack_data_generation(X_train,1, edge_model,Grad_cam2)
data_common=attack_data_generation(X_train,1, common_model,Grad_cam)
data_common_poision=attack_data_generation(X_train,1, dedge_model_poision,Grad_cam3)


# In[46]:


data_common_poision=attack_data_generation(X_train,1, dedge_model_poision,Grad_cam3)


# In[28]:


# Compute the mean and the variance of the training data for normalization.
#data_augmentation.layers[0].adapt(x_train)
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim=6,name=None,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
    def build(self,input_shape):
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim )
    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({"num_patches": self.num_patches,
                })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def sample(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# In[44]:


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:02:55 2021

@author: ali.raza
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time


#disable_eager_execution()

from sklearn.model_selection import train_test_split


from cvxopt import matrix, solvers
import sklearn.metrics.pairwise as smp




from sklearn.metrics import auc



df=pd.read_csv('ecg_final.txt',sep='  ',header=None)
df=df.add_prefix('c')
df['c0']=df['c0'].apply(lambda x: 1 if(x>1) else 0)


input_data = keras.Input(shape=(data_common.shape[1]))

num_classes = 6
input_shape =input_data
learning_rate = 0.01
weight_decay = 0.001
batch_size = 56
num_epochs = 500
image_size = 90  # We'll resize input images to this size
patch_size = 188  # Size of the patches to be extract from the input images
num_patches = 1
projection_dim = 5
num_heads =2
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 1
mlp_head_units = [5]  # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.Normalization(),
        #layers.experimental.preprocessing.Resizing(x_train.shape[1],1),
        #layers.experimental.preprocessing.RandomFlip("horizontal"),
        #layers.experimental.preprocessing.RandomRotation(factor=0.02),
        #layers.experimental.preprocessing.RandomZoom(
         #   height_factor=0.2, width_factor=0.2
        #),
    ],
    name="data_augmentation",
)


    
latent_dim = 10



   

encoder_inputs= layers.Input(shape=(data_common.shape[1],1))
# Augment data.
augmented = data_augmentation(encoder_inputs)
# Create patches.
#patches = Patches(patch_size)(augmented)
# Encode patches.
encoded_patches = PatchEncoder(num_patches, projection_dim)(augmented)


x = layers.Conv1D(64, kernel_size=3,activation='relu', name='input')(encoded_patches)
    
x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)

# Create a [batch_size, projection_dim] tensor.
trans_shape = K.int_shape(encoded_patches) #Shape of conv to be provided to decoder
#Flatten
x = Flatten()(encoded_patches)
x = Dense(100, activation='relu')(x)

#z_mean = Dense(latent_dim, name='z_mean')(x)
#z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use the reparameterization trick and get the output from the sample() function
#z = Lambda(sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(encoder_inputs, x, name='encoder')
encoder.summary()
# decoder model
latent_inputs = Input(shape=(x.shape[1],), name='z_sampling')
x = Dense(trans_shape[1]*trans_shape[2], activation='relu')(latent_inputs)
x = Reshape((trans_shape[1], trans_shape[2]))(x)

# upscale (conv2D transpose) back to original shape
# use Conv2DTranspose to reverse the conv layers defined in the encoder
x = Dense(100,  activation='relu')(x)
#Can add more conv2DTranspose layers, if desired. 
#Using sigmoid activation
outputs = Dense(1,  activation='sigmoid', name='decoder_output')(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
outputs = decoder(encoder(encoder_inputs))
vae_model = Model(encoder_inputs, outputs, name='vae_mlp')


opt = keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
#opt = optimizers.RMSprop(learning_rate=0.0001)




vae_model.compile(optimizer=opt, loss='mse')
vae_model.summary()
# Finally, we train the model:
start = time.time()
history =vae_model.fit(data_common, data_common,
                        shuffle=True,
                        epochs=1,
                        
                        validation_data=(data_common_model,data_common_model),
                        batch_size=7)


# In[45]:


d_common=data_common.reshape(data_common.shape[0],data_common.shape[1],1)
d1=data_common_model.reshape(data_common_model.shape[0],data_common_model.shape[1],1)
d2_poision=data_common_poision.reshape(data_common_poision.shape[0],data_common_poision.shape[1],1)
re_common = vae_model.predict(data_common)
re_d1 = vae_model.predict(data_common_model)
re_d2 = vae_model.predict(d2_poision)


def get_error_term(v1, v2, _rmse=True):
    if _rmse:
        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))
    #return MAE
    return np.mean(abs(v1 - v2), axis=1)
mae_dc = get_error_term(re_common,d_common, _rmse=False)
mae_vector_d1 = get_error_term(re_d1,d1, _rmse=False)
mae_vector_d2 = get_error_term(re_d2,d2_poision, _rmse=False)

threshold=np.mean(mae_dc)+2*np.std(mae_dc)



# In[46]:


with plt.style.context('fivethirtyeight'):
    plt.hist(mae_dc , bins=50)


# In[47]:


with plt.style.context('fivethirtyeight'):
    plt.hist(mae_vector_d1 , bins=50)


# In[48]:


with plt.style.context('fivethirtyeight'):
    plt.hist(mae_vector_d2 , bins=50)


# In[54]:


threshold=np.mean(mae_vector_d1)+3.5*np.std(mae_dc)
dc_check=mae_dc<threshold
d1_check=mae_vector_d1<threshold
d2_check=mae_vector_d2<threshold


# In[55]:


def poision_ammount(data):
    count=0
    for x in range(len(data)):
        if data[x]==0:
            count=count+1
    percentage=count*100/len(data)
    print('percentage of poisoned samples=',percentage)


# In[56]:


poision_ammount(dc_check)


# In[57]:


poision_ammount(d1_check)


# In[58]:


poision_ammount(d2_check)


# In[ ]:





# In[ ]:




