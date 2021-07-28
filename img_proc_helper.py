import numpy as np
import cv2

def resize_crop(image, size=80):
    max_dim = np.argmax(image.shape[:2])
    scale_percent = size / image.shape[max_dim]
    width = round(image.shape[1] * scale_percent)
    height = round(image.shape[0] * scale_percent) 
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if max_dim == 0:
        image = cv2.copyMakeBorder(image, 0,0,0,size-image.shape[1],cv2.BORDER_CONSTANT,0)
    else:
        image = cv2.copyMakeBorder(image, 0,size-image.shape[0],0,0,cv2.BORDER_CONSTANT,0)
    return image

def normalize(imgs):
    return swapaxis(imgs).astype(np.float32)/255

def swapaxis(imgs):
    return np.moveaxis(imgs,-1,1)