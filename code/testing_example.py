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

tlm = TLM()
tlm.load("/its/home/drs25/Tactile_Language_Model/data/gelsight_model_100")
for i in range(10):
    print("*****************")
    image=X[0].reshape((1,1,*X[0].shape))
    print(image.shape)
    print("\tGenerated:",tlm.generate_caption(image))
    print("\tExpected:",y[0])