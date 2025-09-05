import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TKAgg')

import numpy as np 

photos=np.load("/its/home/drs25/Tactile_Language_Model/code/__pycache__/photos.npy")

plt.imshow(photos[0])
plt.show()