import numpy as np
from PIL import Image
import gaussian_pyramid as f

image=Image.open('Vangogh.png')#You can choose other picture in the folder
# image.show()
# print(np.array(image))
f.gaussian_pyramid(image,1.0,7)
# image.show()