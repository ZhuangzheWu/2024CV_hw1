import numpy as np
from PIL import Image
import math

def extend(image,kernel):#根据kernel的大小，为图片补0. image已转为二维array
    # image=np.array(image)
    # print(image.shape)
    k=np.array(kernel)
    k_h=k.shape[0]
    k_w=k.shape[1]
    img_h=image.shape[0]
    img_w=image.shape[1]
    h=k_h//2
    w=k_w//2
    image=np.concatenate([image,np.zeros((h,img_w))],axis=0)
    image=np.concatenate([np.zeros((h,img_w)),image],axis=0)
    image=np.concatenate([image,np.zeros((img_h+2*h,w))],axis=1)
    image=np.concatenate([np.zeros((img_h+2*h,w)),image],axis=1)
    return image

def gaussian_kernel(height,width,sigma):#输入kernel大小，
    arr=np.zeros((height,width))
    xs=1.0/(2*math.pi*math.pow(sigma,2))
    midh=height//2
    midw=width//2
    for i in range(height):
        for j in range(width):
            arr[i][j]=xs*math.exp(-1*((i-midh)**2+(j-midh)**2)/(2*sigma**2))
    arr=arr/arr.sum()
    # print(arr.sum())
    return arr

def cross_correlation_2d(image,kernel):#转为二维array的图片，未拓展
    im_w,im_h=image.shape
    ker_w,ker_h=kernel.shape
    return_image=np.zeros_like(image)
    image=extend(image,kernel)
    for i in range(im_w):
        for j in range(im_h):
            return_image[i][j]=np.sum(np.multiply(kernel,image[i:i+ker_h,j:j+ker_w]))
    return return_image

def convolve_2d(image,kernel):#image为原始格式图片,返回原始格式图片
    image=np.array(image)
    image=np.flip(image,0)
    image=np.flip(image,1)

    img_r=image[:,:,0]
    img_g=image[:,:,1]
    img_b=image[:,:,2]

    r_new=cross_correlation_2d(img_r,kernel)
    g_new=cross_correlation_2d(img_g,kernel)
    b_new=cross_correlation_2d(img_b,kernel)

    image[:,:,0]=r_new
    image[:,:,1]=g_new
    image[:,:,2]=b_new

    image=np.flip(image,0)
    image=np.flip(image,1)

    return Image.fromarray(image)

def gaussian_blur_kernel_2d(image,sigma,height,width):
    kernel=gaussian_kernel(height,width,sigma)
    return convolve_2d(image,kernel)

def low_pass(image,sigma,height,width):
    kernel=gaussian_kernel(height,width,sigma)
    return convolve_2d(image,kernel)

def image_subsampling(image,s):
    image=np.array(image)
    img_w,img_h=image[:,:,0].shape
    # assert(img_w%s==0 and img_h%s==0),"s should be devided by m and n"
    new_image=np.zeros((img_w//s,img_h//s,image.shape[2]),dtype='uint8')
    for i in range(img_w//s):
        for j in range(img_h//s):
            for k in range(image.shape[2]):
                new_image[i][j][k]=image[i*s][j*s][k]
                # print(image[i*s][j*s][k])
    # print(new_image.shape)       
    return Image.fromarray(new_image)

def gaussian_pyramid(image,sigma,n):
    image=np.array(image)
    image_2=low_pass(image,sigma,n,n)
    image_2=image_subsampling(image_2,2)
    image_4=low_pass(image_2,sigma,n,n)
    image_4=image_subsampling(image_4,2)
    image_8=low_pass(image_4,sigma,n,n)
    image_8=image_subsampling(image_8,2)
    
    image_2.save('image_2.png')
    image_4.save('image_4.png')
    image_8.save('image_8.png')

image=Image.open('Vangogh.png')#You can choose other picture in the folder
# image.show()
# print(np.array(image))
gaussian_pyramid(image,1.0,7)
# image.show()
