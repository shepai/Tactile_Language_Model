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
tlm.train(X,y)
tlm.save("/its/home/drs25/Tactile_Language_Model/data/gelsight_model")
print("generated:",tlm.generate_caption(X[0][0]))