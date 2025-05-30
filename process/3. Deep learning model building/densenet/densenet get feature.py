from tensorflow import keras
from keras import *
from keras.models import Model
from keras.layers import Input, Conv3D, Dense, BatchNormalization, Concatenate, Dropout, AveragePooling3D, GlobalAveragePooling3D, Activation
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
from data_acq import X, text, C, cancer,all_Patient_number
from scipy import ndimage
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
from keras.layers import Dense, Input, Conv2D, MaxPooling3D, LayerNormalization, MultiHeadAttention
import random
from sklearn.model_selection import train_test_split


# 超参数配置
DENSE_NET_GROWTH_RATE = 32        
DENSE_NET_BLOCKS = 3                
DENSE_NET_BLOCK_LAYERS = 4          
DENSE_NET_TRANSITION_COMPRESSION = 0.5  
DENSE_NET_ENABLE_BOTTLENECK = True  
DENSE_NET_INITIAL_CONV_DIM = 64    
DENSE_NET_ENABLE_DROPOUT = True     
DENSE_NET_DROPOUT = 0.5             
# GPU配置
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def bn_relu_conv(x, filters, kernel_size=(3, 3, 3)):
    """Bottleneck结构单元"""
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=filters, 
               kernel_size=kernel_size, 
               padding='same')(x)
    return x

def dense_block(x):
    """Dense Block实现"""
    for _ in range(DENSE_NET_BLOCK_LAYERS):
        y = x
        
        # Bottleneck层（1x1x1卷积）
        if DENSE_NET_ENABLE_BOTTLENECK:
            y = bn_relu_conv(y, 
                            filters=4*DENSE_NET_GROWTH_RATE,  # 修正为4k
                            kernel_size=(1, 1, 1))
        
        # 3x3x3卷积
        y = bn_relu_conv(y, 
                        filters=DENSE_NET_GROWTH_RATE,
                        kernel_size=(3, 3, 3))
        
        # 特征拼接
        x = Concatenate(axis=-1)([x, y])  # 在通道维度拼接
    return x

def transition_block(x):
    """Transition Block实现"""
    filters = int(x.shape[-1] * DENSE_NET_TRANSITION_COMPRESSION)
    
    x = bn_relu_conv(x, 
                    filters=filters,
                    kernel_size=(1, 1, 1))
    x = AveragePooling3D(pool_size=(2, 2, 2), 
                        padding='same')(x)
    return x

def build_3d_densenet(input_shape=(32, 32, 32, 1)):
    inputs = Input(input_shape)
    
    # 初始卷积层
    x = Conv3D(2*DENSE_NET_GROWTH_RATE, 
              kernel_size=(7, 7, 7),
              strides=(2, 2, 2),
              padding='same')(inputs)
    x = MaxPooling3D(pool_size=(3, 3, 3), 
                    strides=(2, 2, 2), 
                    padding='same')(x)
    
    # Dense Blocks
    for i in range(DENSE_NET_BLOCKS):
        x = dense_block(x)
        if i != DENSE_NET_BLOCKS - 1:
            x = transition_block(x)
    
    x = GlobalAveragePooling3D()(x)
    x = Dense(1024, activation='relu')(x)  
    x = Dropout(0.5)(x)
    features = Dense(512, activation='relu', name='feature_layer')(x)  
    # 分类头
    if DENSE_NET_ENABLE_DROPOUT:
        features = Dropout(DENSE_NET_DROPOUT)(features)
    outputs = Dense(2, activation='softmax')(features)
    
    return Model(inputs, outputs)

# 创建模型
model = build_3d_densenet(input_shape=(32, 32, 32, 1))  

model.load_weights("3.深度学习模型搭建\\densenet\\3d_densenet.h5")

sub_model = Model(inputs=model.input,
                   outputs=model.get_layer('feature_layer').output)


feature_tensor = []
for i in range(C):
    cancer1=cancer[i]
    print(cancer1.shape)
    cancer1 =cancer1[np.newaxis,:,:]
    prediction = sub_model.predict(cancer1)
    print(prediction.shape)
    feature_tensor.append(prediction)
    print('********************************************************************************')

dictionary = dict(zip(all_Patient_number,feature_tensor))


k = []
for key in dictionary:
    num = [key]
    arr = np.array(dictionary[key])
    new_arr = np.squeeze(arr)
    arrtolist = new_arr.tolist()
    for i in range(len(arrtolist)):
        num.append(arrtolist[i])
    print(num)
    k.append(num)
print(k)

data = pd.DataFrame(k)
writer = pd.ExcelWriter('C:\\Users\\LL\\Desktop\\1234.xlsx')
data.to_excel(writer)
writer.save()
writer.close()



