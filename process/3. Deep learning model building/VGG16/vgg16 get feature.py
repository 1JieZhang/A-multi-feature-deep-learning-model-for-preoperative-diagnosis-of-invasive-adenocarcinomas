 # 导入基本库
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras import *
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from keras.layers import Dense,Input,Conv2D,MaxPooling2D
import keras 
from data_acq import X ,text,all_Patient_number,C,cancer
from scipy import ndimage
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



x_train,x_test,y_train, y_test=train_test_split(X,text,test_size=0.2)#按照比例划分数据集为训练集与测试集

def get_model(width=32, height=32, depth=3):
    """构建 3D 卷积神经网络模型"""

    inputs = keras.Input((width, height, depth))

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = keras.layers.Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    outputs = x

    # 定义模型
    model = tf.keras.Model(inputs, outputs, name="2d_vgg_cnn")

    return model

# 构建模型
model = get_model(width=32, height=32, depth=3)
model.load_weights("2d_image.h5")

sub_model = Model(inputs=model.input,
                  outputs=model.get_layer(index=18).output)
sub_model.summary()



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



