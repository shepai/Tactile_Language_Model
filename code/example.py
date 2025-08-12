import sys
sys.path.append('/its/home/drs25/Tactile_Language_Model/Library/')
from TLM import *
from label_example import *
import numpy as np
print(X.shape)
tlm = TLM()
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tlm.load("/its/home/drs25/Tactile_Language_Model/data/trainedModel",vocab_size=tlm.tokenizer.vocab_size)

print("generated:",tlm.generate_caption(X[0][0]))