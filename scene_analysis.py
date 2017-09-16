import pandas as pd
import numpy as np


def read_ref(img_data_f,loc_im_f,loc_f):
    df_img = pd.read_csv(img_data_f)
    df_loc_im = pd.read_csv(loc_im_f)
    views_im = {}
    for index, row in df_loc_im.iterrows():
        views_im[row["loc_id"]] = {"img1": row["img1"], "img2": row["img2"], "img3": row["img3"], "img4": row["img4"]}
    df_loc = pd.read_csv(loc_f)
    return [df_img,views_im,df_loc]

#parameters
input_filename = "CrowdData/pilot_judgements.csv"
img_data_filename = "Data/images.csv"
loc_im_filename = "Data/loc_im.csv"
loc_filename = "Data/locations.csv"
corr_mat_filename = "CrowdData/corr_mat.csv"
aggr_part1_filename = "CrowdData/pilot_aggregates_part1.csv"
aggr_part2_filename = "CrowdData/pilot_aggregates_part2.csv"
input_image_loc = '../Website/crowdsourcing/public/images'
dataset_image_loc = 'InputImages/Training'
summary_filename = "CrowdData/summary_ori.csv"

scene_filename = "Data/SceneFeatures.csv"



df_aggr_part1 = pd.read_csv(aggr_part1_filename)

df_scene = pd.read_csv(scene_filename)

counter = {}
for idx,row in df_aggr_part1.iterrows():
    label=row["median"]
    scenes = df_scene[df_scene["img_id"]==row["img_id"]]
    for i in range(1,6):
        s = scenes["scene"+str(i)].values[0]
        key = str(label)+"|"+s
        if key in counter.keys():
            counter[key] = counter[key] + 1
        else:
            counter[key] = 1

for k in counter.keys():
    print(k+">"+str(counter[k]))