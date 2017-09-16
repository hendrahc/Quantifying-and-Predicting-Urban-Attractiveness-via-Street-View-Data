import pysal
import pandas as pd
import numpy as np




def load_data(input_file = "../../DATA/attr_judgments.csv",city = "ams"):
    df_dat = pd.read_csv(input_file)
    [lats,longs] = coordinate_mapper(df_dat["latitude"].values,df_dat["longitude"].values, city = city)

    lats.shape = (lats.shape[0],1)
    longs.shape = (longs.shape[0], 1)

    coord_data = np.hstack([lats, longs])

    y = np.array(df_dat["attractiveness"].values)

    return [coord_data, y]

def coordinate_mapper(lat,long,city = "ams"):
    if(city=="ams"):
        return [(lat-52.29)/0.13*17, (long-4.73)/0.25*14]
    elif(city=="delft"):
        return [(lat - 52.014) / 0.019 * 2, (long - 4.319) / 0.089 * 6]

def global_autocorrel(coord_data,y):
    for k in [2,3,4,5,6,7,8,9,10]:
        w = pysal.weights.KNN(coord_data, k=k)
        mi = pysal.Moran(y, w)
        print(str(k)+";"+str(mi.I)+";"+str(mi.p_rand)+";"+str(mi.p_norm))

    for th in [5,4,3,2,1,0.9,0.8,0.7,0.6, 0.5,0.4, 0.3, 0.2, 0.1]:
        w = pysal.weights.DistanceBand.from_array(coord_data, th)
        mi = pysal.Moran(y, w)
        print(str(th)+";"+str(mi.I)+";"+str(mi.p_rand)+";"+str(mi.p_norm))
