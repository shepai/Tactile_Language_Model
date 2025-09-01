import sys
sys.path.append('/its/home/drs25/Tactile_Language_Model/Library/')

from TLM import *
import numpy as np
import pickle 

X=np.load("/mnt/data0/drs25/data/gelsight_language/X_data.npy")
with open("/mnt/data0/drs25/data/gelsight_language/y_data.pkl", "rb") as file:
    y = pickle.load(file)

print(X.shape)
print(len(y))

tlm = TLM()
tlm.load("/its/home/drs25/Tactile_Language_Model/data/gelsight_model")
image=X[0].reshape((1,1,*X[0].shape))
print(image.shape)
print("generated:",tlm.generate_caption(image))