import sys
sys.path.append('/its/home/drs25/Tactile_Language_Model/Library/')
import cv2
from TLM import *
import numpy as np
import pickle 
import re
def round_numbers_in_text(text, decimals=2):
    def repl(match):
        num = float(match.group())
        return f"{num:.{decimals}f}"
    return re.sub(r"\d+\.?\d*", repl, text)

import json
import random
import string
import numpy as np
import os

def random_string(length=5):
    """Generate a random lowercase string."""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def random_flat_json(num_keys=3, key_len=5, val_len=5):
    """Generate a random flat JSON object (dict with string values)."""
    obj = {random_string(key_len): random_string(val_len) 
           for _ in range(num_keys)}
    return json.dumps(obj)

def random_noise_image(size=(64, 64, 3)):
    """Generate a random noise image as a NumPy array."""
    return np.random.randint(0, 256, size, dtype=np.uint8)

def generate_dataset(num_samples=10, 
                     image_size=(320, 240), 
                     num_keys_range=(2, 5), 
                     save_dir="synthetic_dataset"):
    X = []
    y=[]
    for i in range(num_samples):
        num_keys = random.randint(*num_keys_range)
        json_obj = random_flat_json(num_keys=num_keys)
        img_array = random_noise_image(image_size)
        X.append(img_array)
        y.append("json "+str(json_obj))
    return np.array(X),y

X,y = generate_dataset(num_samples=8000)

print(X.shape)
print(len(y))
def apply_sobel_filter(images,blur_ksize=(3,3), blur_sigma=0):
    sobel_images = []
    
    for img in images:
        blurred = cv2.GaussianBlur(img, blur_ksize, blur_sigma)
        if img.ndim == 3 and img.shape[-1] == 3:  # RGB image
            channels = []
            for c in range(3):
                # Compute gradients along x and y
                sobelx = cv2.Sobel(blurred[..., c], cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(blurred[..., c], cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                channels.append(sobel)
            sobel_img = np.stack(channels, axis=-1)
        else:  # Grayscale image
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            sobel_img = cv2.magnitude(sobelx, sobely)
        
        # Normalize to 0–255 and convert to uint8
        sobel_img = cv2.normalize(sobel_img, None, 0, 255, cv2.NORM_MINMAX)
        sobel_img = sobel_img.astype(np.uint8)
        sobel_images.append(sobel_img)
    
    return np.array(sobel_images)
indices = np.random.permutation(len(X))
X = X[indices]
y = [y[i] for i in indices]  # if y is a list; if it's np.array, just do y = y[indices]
sample_amount=8000
X=X[:sample_amount]
y=y[:sample_amount]
X_norm=(X-np.mean(X))/(np.std(X))
X=apply_sobel_filter(X_norm)


# Now split
train_X = X[:len(X)//4]
train_y = y[:len(y)//4]
test_X  = X[len(X)//4:]
test_y  = y[len(y)//4:]


del y
tlm = TLM()
tlm.train(train_X,train_y,epochs=5000,save="/its/home/drs25/Tactile_Language_Model/data/models/pretrained_json2")
#tlm.save("/its/home/drs25/Tactile_Language_Model/data/gelsight_model_10")
image=X[0].reshape((1,1,*X[0].shape))
print(image.shape)
print("generated:",tlm.generate_caption(image))