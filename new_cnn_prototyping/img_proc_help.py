import numpy as np
import cv2

def resize_crop(image,size=224):
    min_dim = np.argmin(image.shape[:2])
    scale_percent = size / image.shape[min_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent) 
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if min_dim == 1:
        image = image[:size]
    else:
        image = image[:,:size]
    return image

def normalize(imgs):
    return swapaxis(imgs).astype(np.float32)/255

def swapaxis(imgs):
    return np.moveaxis(imgs,-1,1)