import os
import numpy as np
from keras import *
import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow import *
import numpy as np
from scipy import ndimage
def read_nifti_file(filepath):
    # 读取文件
    scan = nib.load(filepath)#加载并读取以nii格式的文件名
    # 获取数据
    scan = scan.get_fdata()#将原始数据转为float类型的矩阵数据集
    return scan

def normalize(volume):
    """归一化"""
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def process_scan(path):#变量是一个地址
    # 读取文件
    volume = read_nifti_file(path)#最后的volume是float类型的矩阵
    # 归一化
    volume = normalize(volume)
    return volume
    


file_paths = []
title = []

pathA = r'D:\A'#数据A文件夹的地址（一开始的总地址）
pathB = r'D:\B'#数据B文件夹的地址（一开始的总地址）


def load(path):
    for file in os.listdir(path):#打印出来path下的所有文件夹的名字
        file_path = os.path.join(path, file)  #将路径拼接起来，得到第一级目录下的路径名
        if os.path.isdir(file_path):  #判断file_path(需提供绝对路径)是否为目录
            load(file_path)  # 调用递归函数，再次进行load函数，使其到第二层的路径下
        else:
            if os.path.splitext(file)[0] == '3D data':#分离了文件名与扩展名，并进行判断是否为nii格式的文件
                file_paths.append(os.path.join(file_path))#通过向列append向原来空白的列表末尾添加第二层的文件名，并进行了循环存放。
    return file_paths


data_partA = load(pathA)#执行load函数
data_partB = load(pathB)#执行load函数

normal_scan_paths = []
abnormal_scan_paths = []

for i in data_partA:
    if i[3:4] is 'A':#判断是否为A类
        normal_scan_paths.append(i)#将A类添加到末尾
    if i[3:4] is 'B':#判断是否为B类，需要自己修改成
        abnormal_scan_paths.append(i)#将B类添加到末尾
#normal_scan_paths是A类数据的总地址,字符类型
#abnormal_scan_paths是B类数据的总地址，字符类型

normal_scans = np.array([process_scan(path) for path in normal_scan_paths])#相同

abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])# 循环遍历每一个地址并进行process_scan函数，然后将总的数组变成一个大的数组

#normal_scans=np.squeeze(normal_scans, axis = None)
#abnormal_scans=np.squeeze(abnormal_scans, axis = None)

print(normal_scans.shape)
print(abnormal_scans .shape)
A=normal_scans.shape[0]#判断有几个A类
B=abnormal_scans.shape[0]#判断有几个B类
C=A+B
cancer= np.concatenate((normal_scans, abnormal_scans))#总数据（个数，512）

X = np.array(list(normal_scans) + list(abnormal_scans))
print(X.shape)

text=np.zeros((C, 2))
for i in range(A):
   text[i][0]=1

for i in range(B):

  text[i+A][1]=1


#提特征时候加上
Processing_sequence = abnormal_scan_paths + normal_scan_paths
print("***************************")
print(Processing_sequence)

Processing_sequence_Patient_number = []
for k in Processing_sequence:
    a = k[5:12]

    Processing_sequence_Patient_number.append(a)
# print(Processing_sequence_Patient_number)
# partA is 0 ,partA is normal
# partB is 1 ,partB is abnormal
# first B then A

Processing_sequence_Patient_number_partA = []
Processing_sequence_Patient_number_partB = []

for i in normal_scan_paths:
    Processing_sequence_Patient_number_partA.append(i[5:][:8])
for i in abnormal_scan_paths:
    Processing_sequence_Patient_number_partB.append(i[5:][:8])
print(Processing_sequence_Patient_number_partA)
print(Processing_sequence_Patient_number_partB)
all_Patient_number = Processing_sequence_Patient_number_partB + Processing_sequence_Patient_number_partA
print(all_Patient_number)

