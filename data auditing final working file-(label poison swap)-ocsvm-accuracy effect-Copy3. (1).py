#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

dft_1_upsample=resample(dft_1,replace=True,n_samples=20000,random_state=123)
dft_2_upsample=resample(dft_2,replace=True,n_samples=20000,random_state=124)
dft_3_upsample=resample(dft_3,replace=True,n_samples=20000,random_state=125)
dft_4_upsample=resample(dft_4,replace=True,n_samples=20000,random_state=126)

dft=pd.concat([dft_0,dft_1_upsample,dft_2_upsample,dft_3_upsample,dft_4_upsample])




def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.00,187)
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



    


# In[2]:


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


# In[3]:


common_model.summary()

classify_train = common_model.fit(X_train,y_train,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[14]:


y_pred =common_model.predict(X_test)
#y_pred.reshape(250000,1)
y_pred=np.argmax(y_pred, axis=1)
print(classification_report(target_test, y_pred))
target_names = list("NSVFQ")
clf_report = classification_report(target_test, y_pred, target_names=target_names,output_dict=True)


# In[4]:


edge_model = keras.Model(input_data, classifier_CNN(input_data))
edge_model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
edge_model.summary()

classify_train = edge_model.fit(X_test,y_test,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.2,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[13]:


y_pred =edge_model.predict(X_test)
#y_pred.reshape(250000,1)
y_pred=np.argmax(y_pred, axis=1)
print(classification_report(target_test, y_pred))
target_names = list("NSVFQ")
clf_report = classification_report(target_test, y_pred, target_names=target_names,output_dict=True)


# In[9]:


ave_weights=common_model.get_weights()
ave_weights=[i * 0 for i in ave_weights]

ave_weights_auto=common_model.get_weights()
ave_weights_auto=[i * 0 for i in ave_weights_auto]

weight_scalling_factor=1


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

model_recv=edge_model
scaled_local_weight_list = list()

scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
scaled_local_weight_list.append(scaled_weights)
ave_weights_auto = sum_scaled_weights(scaled_local_weight_list)


# #model aggregation

# In[10]:


model_recv=common_model
scaled_local_weight_list = list()

scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
scaled_local_weight_list.append(scaled_weights)
ave_weights_auto = sum_scaled_weights(scaled_local_weight_list)


# In[7]:


global_model = keras.Model(input_data, classifier_CNN(input_data))
global_model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
global_model.summary()
classify_train = global_model.fit(X_test,y_test,shuffle=True, batch_size=10,epochs=1,verbose=1,
                                validation_split=0.2,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[12]:


global_model.set_weights(ave_weights_auto)
y_pred =global_model.predict(X_train)
#y_pred.reshape(250000,1)
y_pred=np.argmax(y_pred, axis=1)
print(classification_report(target_train, y_pred))
target_names = list("NSVFQ")
clf_report = classification_report(target_test, y_pred, target_names=target_names,output_dict=True)


# In[15]:




randnums= np.random.randint(0,5,target_test.shape[0])

label_poisoning=to_categorical(target_test)


# In[16]:


dedge_model_poision = keras.Model(input_data, classifier_CNN(input_data))
dedge_model_poision.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
dedge_model_poision.summary()

classify_train = dedge_model_poision.fit(X_test_noise,label_poisoning,shuffle=True, batch_size=10,epochs=5,verbose=1,
                                validation_split=0.1,callbacks=[TensorBoard(log_dir='/tmp/autoencoder'),callback])


# In[17]:


model_recv=dedge_model_poision
scaled_local_weight_list = list()

scaled_weights = scale_model_weights(model_recv.get_weights(), weight_scalling_factor)
scaled_local_weight_list.append(scaled_weights)
ave_weights_auto = sum_scaled_weights(scaled_local_weight_list)
global_model.set_weights(ave_weights_auto)

y_pred =global_model.predict(X_test)
#y_pred.reshape(250000,1)
y_pred=np.argmax(y_pred, axis=1)
print(classification_report(target_test, y_pred))
target_names = list("NSVFQ")
clf_report = classification_report(target_test, y_pred, target_names=target_names,output_dict=True)


# In[28]:


acuracy=[]
import matplotlib.pyplot as plt
with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(25, 6))
    ax = fig.add_axes([0,0,1,1])
    langs = ['performance of model trained with Audit data',' FL Without data poisoning', 'label Swapping','random label+feature noise','Random label','feature poisoning(10-20%)']
    accuracy = [83,87,41,24,19,65]
    ax.bar(langs,accuracy)
    plt.xlabel('Global Model')
    plt.ylabel('Accuracy in %')
    plt.title('Comparision of Overall Accuracy (ECG)')
    plt.show()


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt
langs = ['Model trained with Audit data',' Gobal Model Without Data Poisoning', 'Label Swapping','Random label+Feature Noise','Random Label','Feature Poisoning(10-20%)']
accuracy = [86,92,41,24,19,65]
spirit_top = langs
colors = ['red' if (s < max(accuracy)) else 'blue' for s in accuracy]
fig, ax = plt.subplots(figsize=(20,5))
sns.set_style('white')
ax=sns.barplot(x=langs, y=accuracy, palette=colors)
plt.title('Accuracy of Global Model', fontsize=25)
plt.xlabel(None)
plt.xticks(fontsize=10)
plt.ylabel('Accuracy in %', fontsize=20)
plt.yticks(fontsize=15)

sns.despine(bottom=True)
ax.grid(False)
ax.tick_params(bottom=False, left=True)

plt.show()


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
opacity = 0.4
bar_width = 0.35
langs = ['Model trained with Audit data',' Gobal Model Without Data Poisoning', 'Label Swapping','Random label+Feature Noise','Random Label','Feature Poisoning(10-20%)']
accuracy = [87,90,45,24,19,40]
spirit_top = langs
colors = ['black' if (s < max(accuracy)) else 'blue' for s in accuracy]

fig, ax = plt.subplots(figsize=(20,5))
sns.set_style('white')

ax=sns.barplot(x=langs, y=accuracy, palette=colors)

plt.title('Accuracy of Global Model', fontsize=25)
plt.xlabel(None)
plt.xticks(fontsize=10)
plt.ylabel('Accuracy in %', fontsize=20)
plt.yticks(fontsize=15)


plt.show()


# In[22]:


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
    last_conv = model.get_layer('conv1d') #last_conv= model.layers[8]
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
    last_conv = model.get_layer('conv1d_1') #last_conv= model.layers[8]
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


# In[24]:


def attack_data_generation(data,flag,classifier,grad):
    size=data.shape[0]
    for x in range(15000):
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


# In[25]:



data_common_model=attack_data_generation(X_train,1, edge_model,Grad_cam2)
data_common=attack_data_generation(X_train,1, common_model,Grad_cam)
data_common_poision=attack_data_generation(X_train,1, dedge_model_poision,Grad_cam3)


# In[29]:


from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random
import matplotlib.pyplot as plt


svm = OneClassSVM(kernel='rbf', gamma=0.0001, nu=0.03)
print(svm)

svm.fit(data_common)


pred=svm.predict(data_common)

pred1=svm.predict(data_common_model)

pred3=svm.predict(data_common_poision)

def poision_ammount(data):
    count=0
    for x in range(len(data)):
        if data[x]==-1:
            count=count+1
    percentage=count*100/len(data)
    print('percentage of poisoned samples=',percentage)

poision_ammount(pred)

poision_ammount(pred1)

poision_ammount(pred3)


# In[ ]:




