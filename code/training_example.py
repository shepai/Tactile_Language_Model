import sys
sys.path.append('/its/home/drs25/Tactile_Language_Model/Library/')

from TLM import *
import numpy as np
import pickle 
import re
def round_numbers_in_text(text, decimals=2):
    def repl(match):
        num = float(match.group())
        return f"{num:.{decimals}f}"
    return re.sub(r"\d+\.?\d*", repl, text)

X=np.load("/mnt/data0/drs25/data/gelsight_language/X_data.npy")
with open("/mnt/data0/drs25/data/gelsight_language/y_data.pkl", "rb") as file:
    y = pickle.load(file)
y = [round_numbers_in_text(s, 2) for s in y]

print(X.shape)
print(len(y))
train_X=X[0:len(X)//2]
train_y=y[0:len(y)//2]
del X
del y
tlm = TLM()
tlm.train(train_X,train_y,epochs=50,save="/its/home/drs25/Tactile_Language_Model/data/gelsight_model_50",lr=0.01)
#tlm.save("/its/home/drs25/Tactile_Language_Model/data/gelsight_model_10")
image=X[0].reshape((1,1,*X[0].shape))
print(image.shape)
print("generated:",tlm.generate_caption(image))