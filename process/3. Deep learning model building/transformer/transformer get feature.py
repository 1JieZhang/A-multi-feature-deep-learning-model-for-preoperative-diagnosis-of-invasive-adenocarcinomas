from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, LayerNormalization, MultiHeadAttention, Reshape, Embedding, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from data_acq import X, text, all_Patient_number, C, cancer
import numpy as np
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras import layers
import random

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

x_train, x_test, y_train, y_test = train_test_split(X, text, test_size=0.2)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class ClassToken(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.cls_token = None
        
    def build(self, input_shape):

        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True
        )
        super(ClassToken, self).build(input_shape)
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # 扩展分类标记以匹配批次大小
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, inputs], axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])
    
    def get_config(self):
        config = super().get_config()
        config.update({'embed_dim': self.embed_dim})
        return config

def get_pure_transformer_model(width=32, height=32, depth=3):

    patch_size = 8  
    num_patches = (width // patch_size) * (height // patch_size)
    embed_dim = 128  
    
    inputs = Input(shape=(width, height, depth))
    

    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        name='patch_embedding'
    )(inputs)  # 输出形状: (num_patches_w, num_patches_h, embed_dim)
    

    x = layers.Reshape((num_patches, embed_dim))(x)
    

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(
        input_dim=num_patches, 
        output_dim=embed_dim
    )(positions)
    x = x + position_embedding
    

    x = ClassToken(embed_dim=embed_dim)(x)  # 序列长度变为 num_patches+1
    

    for i in range(4):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=4,
            ff_dim=512,
            rate=0.1
        )(x)
    
    x = LayerNormalization(epsilon=1e-6)(x)
    cls_token_output = x[:, 0]  # 提取分类标记
    

    x = Dense(128, activation='relu')(cls_token_output)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="Pure_Transformer")
    return model

model = get_pure_transformer_model(width=32, height=32, depth=3)

# 加载预训练权重（确保路径正确）
model.load_weights("深度学习模型搭建\\transformer\\transformer.h5")  # 

# 创建子模型用于特征提取
# 获取分类标记的特征输出（在分类头之前）
sub_model = Model(inputs=model.input,
                 outputs=model.layers[-3].output)  # 获取分类标记的特征
sub_model.summary()

# 特征提取过程
feature_tensor = []
for i in range(C):
    cancer1 = cancer[i]
    print(cancer1.shape)
    cancer1 = cancer1[np.newaxis, :, :]
    prediction = sub_model.predict(cancer1)
    print(prediction.shape)
    feature_tensor.append(prediction)
    print('********************************************************************************')

# 创建特征字典
dictionary = dict(zip(all_Patient_number, feature_tensor))

# 准备数据保存
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

# 保存到Excel
data = pd.DataFrame(k)
writer = pd.ExcelWriter('C:\\Users\\LL\\Desktop\\1234.xlsx')
data.to_excel(writer)
writer.save()
writer.close()