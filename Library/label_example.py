import cv2
import numpy as np

def load(filename="/mnt/data0/drs25/data/optical-tactile-dataset-for-textures/texture-tactip/X_data_15.npz"):
    if "npz" in filename:
        X = np.load(filename)['arr_0'].astype(np.uint8)
        y = np.load(filename.replace("X","y"))['arr_0'].astype(np.uint8)
    X=X[:,0:7,:,:]
    X=X.reshape((len(X),X.shape[1]*X.shape[2],X.shape[3]))
    print(X.shape)
    def reshape(X,percent):
        w=int(X.shape[1]*percent)
        h=int(X.shape[2]*percent)
        new_array=np.zeros((X.shape[0],w,h),dtype=np.uint8)
        for i in range(X.shape[0]):
            new_array[i] = cv2.resize(X[i],(h,w),interpolation=cv2.INTER_AREA)
        return new_array
    X=reshape(X,0.5)
    keys=['Carpet', 'LacedMatt', 'wool', 'Cork', 'Felt', 'LongCarpet', 'cotton', 'Plastic', 'Flat', 'Ffoam', 'Gfoam', 'bubble', 'Efoam', 'jeans', 'Leather']
    material_descriptions = {
        "carpet": "Dense, woven fibers, soft yet coarse, typically synthetic or wool blend.",
        "lacedmatt": "Light, airy mesh structure with interwoven hard lace patterns; bumpy and flexible.",
        "wool": "Natural fiber, soft, warm, and slightly scratchy; high friction texture.",
        "cork": "Lightweight, firm but compressible, slightly rough with granular texture.",
        "felt": "Compressed fabric, soft and smooth surface, uniform texture with slight give.",
        "longcarpet": "High-pile carpet with long fibers, soft and plush, deep texture.",
        "cotton": "Smooth and soft woven fabric, breathable with moderate friction.",
        "plastic": "Hard, smooth surface with low friction; can vary from rigid to flexible.",
        "flat": "Smooth and even surface, minimal texture; likely hard or semi-soft material.",
        "ffoam": "Soft with slight springiness, absorbs pressure well.",
        "gfoam": "Grainy foam, slightly rougher texture, spongy and compressible.",
        "bubble": "Bubble wrap or bubbled plastic, soft with raised circular nodes, very bumpy.",
        "efoam": "Soft with slight springiness, absorbs pressure well.",
        "jeans": "Sturdy cotton denim, rough woven texture, moderate friction.",
        "leather": "Smooth and durable natural material, slightly soft with subtle grain."
    }
    new_y=[]
    for i in range(len(y)):
        label=keys[int(y[i])].lower()
        new_y.append(material_descriptions[label])
    return X, new_y