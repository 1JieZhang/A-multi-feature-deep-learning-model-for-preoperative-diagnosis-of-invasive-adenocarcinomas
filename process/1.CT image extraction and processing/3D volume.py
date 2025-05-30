from http.client import FORBIDDEN
import os
from xml.etree.ElementTree import tostringlist
import numpy as np
import nibabel as nib 
from data import data_partA
from mask import data_partB

def read_nifti_file(filepath):
    # Read the file
    scan = nib.load(filepath)  # Load and read NIfTI format file
    # Get the data
    scan = scan.get_fdata()  # Convert raw data to float type matrix dataset
    return scan

def MAX1(tensor1, min1):
    i = 0
    U = 0
    ma = 0
    c = 0
    for u in range(len(tensor1)):
        c = 0
        for i in range(len(tensor1)):
            if min1 == tensor1[i]:
                c = 1
                break
        if c == 1:
            min1 = min1 + 1
            continue
        else:
            ma = min1 - 1
            break
    return ma  # Return the first circle

#for i in range(len(data_partA)): 同理，可以设置成训练，将下面的data_partA[0]变成data_partA[i]就行了
pathA = data_partA[2]  # data
pathB = data_partB[2]  # mask
print(pathB)    
print(pathA)   
nii_img = nib.load(pathA)
affine = nii_img.affine.copy()
hdr = nii_img.header.copy()

nii_img1 = nib.load(pathB)
affine1 = nii_img1.affine.copy()
hdr1 = nii_img1.header.copy()

mask_tensor = np.array([read_nifti_file(pathB)])
yuan_tensor = np.array([read_nifti_file(pathA)])

# Determine T, length, width, height
t, h, w, g = yuan_tensor.shape

# Determine center point
tempL = np.nonzero(mask_tensor)

minx = np.min(tempL[1])
miny = np.min(tempL[2])
minz = np.min(tempL[3])

maxx = np.max(tempL[1])
maxy = np.max(tempL[2])
maxz = np.max(tempL[3])

maxx = MAX1(tempL[1], minx)
maxy = MAX1(tempL[2], miny)
maxz = MAX1(tempL[3], minz)

pyx = int((minx + maxx) / 2)
pyy = int((miny + maxy) / 2)
pyz = int((minz + maxz) / 2)

# Extract 32×32×32 block around the center point
start_x = max(pyx - 16, 0)
end_x = min(pyx + 16, h)
start_y = max(pyy - 16, 0)
end_y = min(pyy + 16, w)
start_z = max(pyz - 16, 0)
end_z = min(pyz + 16, g)

# Extract 3D block from original data and mask
yuan_block = yuan_tensor[0, start_x:end_x, start_y:end_y, start_z:end_z]
mask_block = mask_tensor[0, start_x:end_x, start_y:end_y, start_z:end_z]

# If the block is smaller than 32×32×32 due to edge cases, pad it with zeros
if yuan_block.shape != (32, 32, 32):
    pad_width = [(0, 32 - s) for s in yuan_block.shape]
    yuan_block = np.pad(yuan_block, pad_width, mode='constant')
    mask_block = np.pad(mask_block, pad_width, mode='constant')

# Create new NIfTI images
new_nii = nib.Nifti1Image(yuan_block, affine, hdr)
new_nii1 = nib.Nifti1Image(mask_block, affine1, hdr1)

# Save NIfTI files
nib.save(new_nii, "3D data" )#data

nib.save(new_nii1, "3D mask" )#,mask