from http.client import FORBIDDEN
import os
from xml.etree.ElementTree import tostringlist
import numpy as np
import nibabel as nib 
from data import data_partA
from mask import data_partB
def read_nifti_file(filepath):
    # 读取文件
    scan = nib.load(filepath)#加载并读取以nii格式的文件名
    # 获取数据
    scan = scan.get_fdata()#将原始数据转为float类型的矩阵数据集
    return scan

def MAX1(tensor1,min1):
    i=0
    U=0
    ma=0
    c=0
    for u in range(len(tensor1)):
      c=0
      for i in range(len(tensor1)):
         if min1==tensor1[i]:
            c=1
            break
      if c==1:
        min1=min1+1
        continue
      else:
        ma=min1-1
        break
    return ma#返回第一个圆
   

#for i in range(len(data_partA)):根据需求可以设置循环
pathA=data_partA[2]#data
pathB=data_partB[2]#mask
print(pathB)    
print(pathA)   
nii_img = nib.load(pathA)
affine = nii_img.affine.copy()
hdr = nii_img.header.copy()

nii_img1 = nib.load(pathB)
affine1 = nii_img1.affine.copy()
hdr1 = nii_img1.header.copy()



mask_tensor = np.array([read_nifti_file(pathB) ])
yuan_tensor = np.array([read_nifti_file(pathA) ])

  #确定T,长，宽，高
t,h,w,g=yuan_tensor.shape
  #确定T,长，宽，高

  #确定中心点
tempL = np.nonzero(mask_tensor)


minx= np.min(tempL[1])
miny= np.min(tempL[2])
minz = np.min(tempL[3])

maxx= np.max(tempL[1])
maxy= np.max(tempL[2])
maxz= np.max(tempL[3])


maxx=MAX1(tempL[1],minx)
maxy=MAX1(tempL[2],miny)
maxz=MAX1(tempL[3],minz)

pyx=int((minx+maxx)/2)
pyy=int((miny+maxy)/2)
pyz=int((minz+maxz)/2)
  # 确定中心点
  

  # 切割成二维的nii——tensor（原本是4维的nii图像）
yuan_qieX= yuan_tensor[ 0,pyx,:,:] #选择X方向的切片都可以
yuan_qieY= yuan_tensor[ 0,:,pyy,:] #选择Y方向的切片都可以
yuan_qieZ= yuan_tensor[ 0,:,:,pyz] #选择Z方向的切片都可以



mask_qieX= mask_tensor[ 0,pyx,:,:] #选择X方向的切片都可以
mask_qieY= mask_tensor[ 0,:,pyy,:] #选择Y方向的切片都可以
mask_qieZ= mask_tensor[ 0,:,:,pyz] #选择Z方向的切片都可以



 # 32*32
img_qieX =yuan_qieX [max((pyy - 16), 1):min((pyy + 16),w ),max((pyz - 16), 1):min((pyz + 16), g)]
img_qieY =yuan_qieY [max((pyx - 16), 1):min((pyx + 16),h ),max((pyz - 16), 1):min((pyz + 16), g)]
img_qieZ =yuan_qieZ [max((pyx - 16), 1):min((pyx + 16),h ),max((pyy - 16), 1):min((pyy + 16), w)]


mask_qieX =mask_qieX [max((pyy - 16), 1):min((pyy + 16),w ),max((pyz - 16), 1):min((pyz + 16), g)]
mask_qieY =mask_qieY [max((pyx - 16), 1):min((pyx + 16),h ),max((pyz - 16), 1):min((pyz + 16), g)]
mask_qieZ =mask_qieZ [max((pyx - 16), 1):min((pyx + 16),h ),max((pyy - 16), 1):min((pyy + 16), w)]

img_qie= np.array((img_qieX,img_qieY,img_qieZ))

img_qie1= np.array((mask_qieX,mask_qieY,mask_qieZ))


new_niiX = nib.Nifti1Image(img_qie, affine, hdr)

new_niiX1 = nib.Nifti1Image(img_qie1, affine1, hdr1)
 #保存nii文件，后面的参数是保存的文件名


nib.save(new_niiX, "2D data" )#data

nib.save(new_niiX1, "2D mask" )#,mask