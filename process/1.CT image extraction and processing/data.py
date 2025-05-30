import os
import numpy as np
import nibabel as nib 
import imageio 
import os
import numpy as np
from collections import Counter


file_paths = []
pathA="D:\\A"#数据的总地址
def load(path):
    for file in os.listdir(path):#打印出来path下的所有文件夹的名字
        file_path = os.path.join(path, file)#将路径拼接起来，得到第一级目录下的路径名
        if os.path.isdir(file_path):  #判断file_path(需提供绝对路径)是否为目录
            load(file_path)  # 调用递归函数，再次进行load函数，使其到第二层的路径下
        else:
            if os.path.splitext(file)[0] == 'data':#分离了文件名与扩展名，并进行判断是否为nii格式的文件
               file_paths.append(os.path.join(file_path))#通过向列append向原来空白的列表末尾添加第二层的文件名，并进行了循环存放。
    return file_paths

i=0
data_partA = load(pathA)#执行load函数
print(len(data_partA))

