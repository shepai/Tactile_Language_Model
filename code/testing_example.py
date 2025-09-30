import sys
sys.path.append('/its/home/drs25/Tactile_Language_Model/Library/')

from TLM import *
import numpy as np
import pickle 
import re
import cv2 
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
def round_numbers_in_text(text, decimals=2):
    def repl(match):
        num = float(match.group())
        return f"{num:.{decimals}f}"
    return re.sub(r"\d+\.?\d*", repl, text)

X=np.load("/mnt/data0/drs25/data/gelsight_language/X_data.npy")
with open("/mnt/data0/drs25/data/gelsight_language/y_json.pkl", "rb") as file:
    y = pickle.load(file)
y = [round_numbers_in_text(s, 2).replace('\n', ' ') for s in y]
indices = np.random.permutation(len(X))
X = X[indices]
X=X[0:100]

X_norm=(X-np.mean(X))/(np.std(X))
X=apply_sobel_filter(X_norm)


print(X.shape)
print(len(y))

tlm = TLM()
bos = tlm.tokenizer.bos_token
eos = tlm.tokenizer.eos_token
y = [bos + y[i] + eos for i in indices]
y=y[0:100]
tlm.load("/its/home/drs25/Tactile_Language_Model/data/models/newembedding") #transferlearning_noresnet pretrained_json 
for i in range(10):
    print("*****************")
    image=torch.tensor(X[0:1].reshape((1,*X[0].shape,1)))
    print(image.shape)
    print("\n\tGenerated greedy:",tlm.generate_caption(image))
    print("\n\tGenerated non greedy:",tlm.generate_caption_nongreedy(image))
    print("\n\tExpected:",y[0])