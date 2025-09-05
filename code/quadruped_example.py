if __name__=="__main__":
    import sys
    #sys.path.insert(1, '/its/home/drs25/Documents/GitHub/Quadruped/Code')
    sys.path.insert(1, '/its/home/drs25/Quadruped/Code')
    #sys.path.insert(1, 'C:/Users/dexte/Documents/GitHub/Quadruped/Code')
datapath="/its/home/drs25/Quadruped/"

from environment import *
from CPG import CTRNNQuadruped
import pickle

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
def slowSpeed(geno):
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

def runExperiment(textureNum,generations=100,delay=0,fitness=fitness_,control=False):
    #load in LLM prediction
    
    #activate those LLM functions before run
    index=np.argmax(fitnesses)
    env=environment(record=0,floorpath="/its/home/drs25/Terrain_generator_3D/assets/tactile"+str(textureNum)+".urdf") #
    photos=-1
    agent=population[index]

    history={}
    history['positions']=[]
    history['orientations']=[]
    history['motors']=[]
    history['accumalitive_reward']=[]
    history['self_collisions']=[]
    history['feet']=[]
    env.reset()
    a=[]
    photos_l=[]
    if control: #predetermined LLM outputs
        pass
    elif textureNum==0:
        maintainSpeed(agent)
        widenLegStride(env)
        lowerBody(env)
    elif textureNum==1:
        maintainSpeed(agent)
        widenLegStride(env)
        maintainBody(env)
    elif textureNum==2:
        maintainSpeed(agent)
        widenLegStride(env)
        lowerBody(env)
    elif textureNum==3:
        increaseSpeed(agent)
        maintainLegStride(env)
        lowerBody(env)
    elif textureNum==4:
        increaseSpeed(agent)
        maintainLegStride(env)
        maintainBody(env)
    elif textureNum==5:
        slowSpeed(agent)
        widenLegStride(env)
        lowerBody(env)
    elif textureNum==6:
        maintainSpeed(agent)
        widenLegStride(env)
        lowerBody(env)
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
    #np.save("/its/home/drs25/Documents/GitHub/Quadruped/Code/data_collect_proj/trials_all/"+str(filename),history)
    env.stop()
    #np.savez(datapath+"/Code/GAs/test",history,allow_pickle=True)

    #np.save("/its/home/drs25/Tactile_Language_Model/code/__pycache__/photos",np.array(photos_l))
    return history

if __name__=="__main__":
    for i in range(7):
        print("experiment",i)
        hist=runExperiment(i,delay=0)
        np.savez("/its/home/drs25/Tactile_Language_Model/data/hist_"+str(i),hist,allow_pickle=True)
        hist=runExperiment(i,delay=0,control=True)
        np.savez("/its/home/drs25/Tactile_Language_Model/data/hist_control_"+str(i),hist,allow_pickle=True)