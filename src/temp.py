import cv2
import os
import imutils
from tensorflow.keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
import numpy as np

beard_path = os.path.join(os.path.join("..",'data_aug'),'beard1.png')

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                        horizontal_flip=True, fill_mode="nearest")

image = load_img(beard_path)
image = img_to_array(image)
image = np.expand_dims(image,axis = 0)
label = "beard"
save_to_dir = "temp"
imageGen = aug.flow(image, batch_size=1, save_to_dir=save_to_dir,save_prefix=label, save_format="jpg")
total = 0 
for image in imageGen:
    total +=1

print("Produced {} images in total".format(str(total)))