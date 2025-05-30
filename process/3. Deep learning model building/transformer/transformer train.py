from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, LayerNormalization, MultiHeadAttention, Reshape, Embedding
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from data_acq import X, text
import numpy as np
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras import layers
import random
from scipy import ndimage

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

class ClassToken(layers.Layer):
    """可序列化的分类标记层"""
    def __init__(self, embed_dim, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        # 创建可训练的分类标记权重
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

# 验证输入形状
print("输入形状:", X.shape)


model = get_pure_transformer_model(32, 32, 3)
model.summary()

# 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, text, test_size=0.2)

# 数据增强
@tf.function
def rotate(volume):
    def scipy_rotate(vol):
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        vol = ndimage.rotate(vol, angle, reshape=False)
        return np.clip(vol, 0, 1)
    return tf.numpy_function(scipy_rotate, [volume], tf.float32)

def train_preprocessing(volume, label):
    return rotate(volume), label

# 创建数据集
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train))
                .map(train_preprocessing)
                .batch(2)  
                .prefetch(tf.data.AUTOTUNE))

test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(2)
               .prefetch(tf.data.AUTOTUNE))

# 优化器配置
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=30,
    decay_rate=0.96,
    staircase=True
)

# 编译模型
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=['accuracy']
)

# 回调函数
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "transformer.h5",
        save_best_only=True,
        monitor='val_accuracy',
        save_weights_only=False
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6
    )
]

# 训练模型
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=200,
    verbose=1,
    callbacks=callbacks
)


'''
# 可视化训练过程
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
'''