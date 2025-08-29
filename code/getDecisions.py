import pandas as pd
import os
import pickle
import random
clear = lambda: os.system('clear')
with open("/mnt/data0/drs25/data/gelsight_language/y_data.pkl", "rb") as file:
    y = pickle.load(file)

import json
with open("/its/home/drs25/Tactile_Language_Model/data/texture_desciptions.json", "r") as f:
    texture_descriptions = json.load(f)
print(len(texture_descriptions.keys()))
indices = list(range(len(y)))
random.shuffle(indices)
y = [y[i] for i in indices]
sample=1000
for i in range(sample):
    y[i]=y[i].replace(",",";").replace("\n"," ")
df=pd.DataFrame({
    "ID":[indices[i] for i in range(len(y[:sample]))],
    "text":y[:sample],
    "gemma":[None for i in range(len(y[:sample]))],
    "llamas":[None for i in range(len(y[:sample]))],
    "gpt":[None for i in range(len(y[:sample]))],
    "mistral":[None for i in range(len(y[:sample]))],
    "deepseek":[None for i in range(len(y[:sample]))],
    "gemmaspeed":[0 for i in range(sample)],
    "llamasspeed":[0 for i in range(sample)],
    "gptspeed":[0 for i in range(sample)],
    "gemmaspeed":[0 for i in range(sample)],
    "deepseekspeed":[0 for i in range(sample)],
})


from desicionClass import * 
#mistral
#gemma3
#deepseek-r1
#gpt-oss
#llama3.1
models=["mistral","gemma3","gpt-oss","llama3.1","deepseek-r1"]
names=['mistral','gemma','gpt','llamas',"deepseek"]
import time
for j in range(len(names)):
    modelA=Decisions(model=models[j])
    ar=[]
    times=[]
    for i in range(len(y[:sample])):
        t1=time.time()
        text=y[i]
        clear()
        print(models[j],":\n",(j*len(y[:sample]) + i)/(len(y[:sample])*len(names)) *100,"%")
        ar.append(modelA.chat(text).replace(",",";").replace("\n"," "))
        t2=time.time() 
        times.append(t2-t1)
    df[names[j]+"speed"]=times
    df[names[j]]=ar
    df.to_csv("/its/home/drs25/Tactile_Language_Model/data/Functional.csv")
