from tensorflow import keras
from keras import *
from keras.models import Model
from keras.layers import Input, Conv3D, Dense, BatchNormalization, Concatenate, Dropout, AveragePooling3D, GlobalAveragePooling3D, Activation
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from data_acq import X, text
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

X = np.expand_dims(X, axis=-1)  
# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, text, test_size=0.2)


# 动态学习率
initial_learning_rate = 1e-4
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=30, decay_rate=0.96, staircase=True
)

# 编译模型
model.compile(
    loss="binary_crossentropy",  # 假设使用整数标签
    optimizer=Adam(learning_rate=lr_schedule),
    metrics=["acc"]
)

# 训练配置
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_densenet.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15
)

# 数据预处理适配3D输入
@tf.function
def rotate_3d(volume):
    """3D数据增强"""
    def scipy_rotate(vol):
        # 随机选择旋转轴和角度
        axis = random.choice([0, 1, 2])
        angle = random.choice([-20, -10, -5, 5, 10, 20])
        vol = ndimage.rotate(vol, angle, axes=(axis, (axis+1)%3), reshape=False)
        return np.clip(vol, 0, 1)
    return tf.numpy_function(scipy_rotate, [volume], tf.float32)

def train_preprocessing(volume, label):
    return rotate_3d(volume), label


# 创建数据集
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train))
                .map(train_preprocessing)
                .batch(2)  
                .prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(2)
               .prefetch(tf.data.AUTOTUNE))
# 开始训练
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200,
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=1,
)