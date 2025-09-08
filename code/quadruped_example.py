if __name__=="__main__":
    import sys
    #sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
    sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
    sys.path.insert(1, '/its/home/drs25/Tactile_Language_Model/Library')
    #sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Quadruped/"

from environment import *
from CPG import CTRNNQuadruped
import pickle
from TLM import Decisions
with open(datapath+'/models/6_neurons/genotypes_dt0.1__6_neurons_0_F10.5.pkl', 'rb') as f:
    population = pickle.load(f)
fitnesses=np.load(datapath+"/models/6_neurons/fitnesses_dt0.1__6_neurons_0_F10.5.npy")
def fitness_(robot,history={}): 
    fitness=0
    #look at behaviour over time
    if len(history.get('motors',[]))>0:
        distances=np.array(history['positions'])-np.array([robot.start])#euclidean_distance(np.array(history['positions']),np.array([robot.start]))
        distancesX=distances[-1][0]
        distancesY=distances[-1][1]
        distancesZ=distances[-1][2]
        fitness+=distancesX - (distancesY+distancesZ)/10 #np.sum(distances)

    if robot.hasFallen(): fitness=0
    if type(fitness)!=type(0): 
        try:
            if type(fitness)==type([]): fitness=float(fitness[0])
            else:fitness=float(fitness)
        except:
            print("shit",fitness,np.array(history['motors']).shape,np.array(history['positions']).shape,np.array(history['orientations']).shape)
            fitness=0
    if fitness<0: fitness=0
    return fitness
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2,axis=1))

def increaseSpeed(geno):
    geno.dt=0.5
def decreaseSpeed(geno):
    geno.dt=0.01
def increaseBody(env):
    env.INCREASE=10
    env.BALANCE=-10
def lowerBody(env):
    env.INCREASE=-30
    env.BALANCE=30
def maintainSpeed(geno):
    geno.dt=0.1
def maintainBody(env):
    env.INCREASE=0
    env.BALANCE=0
def widenLegStride(env):
    env.stide=2
def maintainLegStride(env):
    env.stide=1
def decreaseLegStride(env):
    env.stide=0.5

import json 
with open("/its/home/drs25/Tactile_Language_Model/data/floor_descriptions.json","r", encoding="utf-8-sig") as f:
    floor_des=json.load(f)
def runExperiment(textureNum,generations=100,delay=0,fitness=fitness_,control=False,fric=0.5,model="gemma3"):
    index=np.argmax(fitnesses)
    if textureNum>=0:
        env=environment(record=0,floorpath="/its/home/drs25/Terrain_generator_3D/assets/tactile"+str(textureNum)+".urdf",friction=fric) #
    else:
        env=environment(record=0,friction=fric) #

    photos=-1
    agent=population[index]
    env.reset()
    d=Decisions(model)#load in LLM prediction
    prompt = d.chat(floor_des[str(textureNum)]+" and the lateral coefficient friction is "+ "high" if fric>0.4 else "low")#activate those LLM functions before run
    if control: #predetermined LLM outputs
        maintainSpeed(agent) 
        maintainLegStride(env)
        maintainBody(env)
    else:
        if "maintainspeed" in prompt.lower():
            maintainSpeed(agent) 
            print("\tMaintain speed")
        elif "increasespeed" in prompt.lower():
            increaseSpeed(agent)
            print("\tIncrease speed")
        elif "decreasespeed" in prompt.lower():
            decreaseSpeed(agent)
            print("\tDecrease speed")
        if "widenlegstride" in prompt.lower() or "increaselegstride" in prompt.lower():
            widenLegStride(env)
            print("\tWiden stride")
        elif "decreaselegstride" in prompt.lower(): 
            decreaseLegStride(env)
            print("\tDecrease stride")
        elif "maintainlegstride" in prompt.lower():
            maintainLegStride(env)
            print("\tMaintain stride")
        if "lowerbody" in prompt.lower():
            lowerBody(env)
            print("\tLower body")
        elif "increasebody" in prompt.lower():
            increaseBody(env)
            print("\tIncrease body")
        elif "maintainbody" in prompt.lower():
            maintainBody(env)
            print("\tMaintain body")
    
    history=run(env,generations,delay,agent,photos,fitness)
    #np.save("/its/home/drs25/Documents/GitHub/Quadruped/Code/data_collect_proj/trials_all/"+str(filename),history)
    env.stop()
    #np.savez(datapath+"/Code/GAs/test",history,allow_pickle=True)

    #np.save("/its/home/drs25/Tactile_Language_Model/code/__pycache__/photos",np.array(photos_l))
    return history

def run(env,generations,delay,agent,photos,fitness):
    a=[]
    photos_l=[]
    history={}
    history['positions']=[]
    history['orientations']=[]
    history['motors']=[]
    history['accumalitive_reward']=[]
    history['self_collisions']=[]
    history['feet']=[]
    for i in range(generations*10):
        pos=env.step(agent,0,delay=delay)
        if photos>-1 and i%photos==0:
            print("snap")
            photos_l.append(env.take_agent_snapshot(p,env.robot_id))
        #pos[[2,5,8,11]]=180-pos[[1,4,7,10]]
        basePos, baseOrn = p.getBasePositionAndOrientation(env.robot_id) # Get model position
        history['positions'].append(basePos)
        history['orientations'].append(baseOrn[0:3])
        history['motors'].append(pos)
        history['accumalitive_reward'].append(fitness(env.quad,history=history))
        history['self_collisions'].append(env.quad.get_self_collision_count())
        history['feet'].append(env.quad.getFeet())
        p.resetDebugVisualizerCamera( cameraDistance=0.3, cameraYaw=75, cameraPitch=-20, cameraTargetPosition=basePos) # fix camera onto model
        if env.quad.hasFallen():
            break
        if env.quad.hasFallen():
            break
        a.append(pos)
    history['positions']=np.array(history['positions'])
    history['orientations']=np.array(history['orientations'])
    history['motors']=np.array(history['motors'])
    history['accumalitive_reward']=np.array(history['accumalitive_reward'])
    history['self_collisions']=np.array(history['self_collisions'])
    history['feet']=np.array(history['feet'])
    filename = str(uuid.uuid4())
    return history
def testAll(generations=100,delay=0,fitness=fitness_,control=False,fric=0.5):
    speed=[maintainSpeed,increaseSpeed,decreaseSpeed]
    body=[maintainBody,increaseBody,increaseBody]
    stride=[maintainLegStride,widenLegStride,decreaseLegStride]
    index=np.argmax(fitnesses)
    
    env=environment(record=0,friction=fric) #
    photos=-1
    agent=population[index]
    env.reset()
    
    env.stop()
    for i,sp in enumerate(speed):
        for j,bo in enumerate(body):
            for k,st in enumerate(stride):
                print(i,j,k)
                sp(agent)
                bo(env)
                st(env)
                history=run(env,generations,delay,agent,photos,fitness)
                np.savez("/its/home/drs25/Tactile_Language_Model/data/quadruped/"+str(i)+"_"+str(j)+"_"+str(k),history,allow_pickle=True)

    return history
    

if __name__=="__main__":
    testAll()
    """models=["mistral","gemma3","gpt-oss","llama3.1"] #,"deepseek-r1"
    for model in models:
        for i in range(7):
            for j in range(10):
                print("experiment",i)
                hist=runExperiment(i,delay=0,fric=0.05,model=model)
                np.savez("/its/home/drs25/Tactile_Language_Model/data/quadruped/"+model+"_hist_0.5_"+str(i)+"_"+str(j)+"_low",hist,allow_pickle=True)
                hist=runExperiment(i,delay=0,control=True,fric=0.05,model=model)
                np.savez("/its/home/drs25/Tactile_Language_Model/data/quadruped/"+model+"_control_0.5_"+str(i)+"_"+str(j)+"_low",hist,allow_pickle=True)"""