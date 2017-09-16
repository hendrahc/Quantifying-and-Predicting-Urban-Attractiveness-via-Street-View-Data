import urllib
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import os.path
from keras.preprocessing import image

width = 600
height = 400
size = str(width) + "x" + str(height)
pitch = -0.76

def generate_gsv_url(size,lat,long,heading,pitch):
    location = str(lat) + "," + str(long)
    url = "https://maps.googleapis.com/maps/api/streetview?size=" + size + "&location=" + location + "&heading=" + str(
        heading) + "&pitch=" + str(pitch) + "&key=AIzaSyDeww92hY7OZDVGFyE7u5wHKXInVBmujHg"
    #print(url)
    return url


def image_valid(lat, long):
    heading = 0
    url = generate_gsv_url(size,lat,long,heading,pitch)
    urllib.request.urlretrieve(url, "test.jpg")
    statinfo = os.stat("test.jpg")
    if (statinfo.st_size < 7000):
        return 0
    return 1


def download_image(lat, long, heading, filename):
    url = generate_gsv_url(size,lat,long,heading,pitch)
    urllib.request.urlretrieve(url, filename)

#def get_nearby_image(lat,long):

def attr_f(input_dir,img_name,loc_im,df_img, df_vote):
    nameparts = img_name.split("_")
    loc_id =  int(nameparts[1])
    dir = int(nameparts[2].split(".")[0])
    i = 1
    img_id_1 = loc_im[loc_id]["img"+str(i)]
    img_1 = df_img[df_img["id"]==img_id_1]
    dir_1 = img_1["heading"].values[0]

    if((dir_1-dir)%90 == 0):
        pos_dir = ((dir-dir_1)//90)%4+1
        #print(str(dir)+" is the same as "+str(df_img[df_img["id"]==loc_im[loc_id]["img"+str(pos_dir)]]["heading"].values[0]))
        attr_dir = df_vote[df_vote["img_id"] == loc_im[loc_id]["img"+str(pos_dir)]]["median"].values[0]
        #print(str(dir)+" <=> "+str(attr_dir))
        return attr_dir
    else:
        pos_left_dir = ((dir-dir_1)//90)%4+1
        pos_right_dir = pos_left_dir%4+1

        dir_left = df_img[df_img["id"]==loc_im[loc_id]["img"+str(pos_left_dir)]]["heading"].values[0]
        dir_right = (dir_left+90)%360
        #print(str(dir) +" is between "+str(dir_left) + " and "+ str(dir_right))

        attr_left = df_vote[df_vote["img_id"] == loc_im[loc_id]["img"+str(pos_left_dir)]]["median"].values[0]
        attr_right = df_vote[df_vote["img_id"] == loc_im[loc_id]["img" + str(pos_right_dir)]]["median"].values[0]

        closeness_left =  ((dir_right-dir)%90) / 90
        closeness_right = ((dir-dir_left) % 90) / 90

        pred_attr =  closeness_left*attr_left + closeness_right*attr_right
        #print(str(dir_left)+":"+str(dir)+":"+str(dir_right)+" <=> "+str(attr_left)+":"+str(pred_attr)+":"+str(attr_right))
        return round(pred_attr)




def label_views(loc_im,df_img=pd.read_csv("Data/images.csv"),df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_view/",output_log="Expansion/attr_exp_view.csv"):
    df_expview = pd.DataFrame(columns=["loc_id","img_name","attractiveness"])

    files = [x for x in os.listdir(input_dir) if x.endswith('.jpg')]
    for loc_id in loc_im.keys():
        imgs = [x for x in files if x.split("_")[1]==str(loc_id)]
        for img_name in imgs:
            attr = attr_f(input_dir, img_name, loc_im,df_img, df_vote)
            newdat = {}
            newdat["loc_id"] = loc_id
            newdat["img_name"] = img_name
            newdat["attractiveness"] = attr
            df_expview = df_expview.append(newdat, ignore_index=True)
    df_expview["loc_id"] = df_expview["loc_id"].astype(int)
    df_expview["attractiveness"] = df_expview["attractiveness"].astype(int)
    df_expview.to_csv(output_log,sep=",")
    return df_expview


def label_views_linear(df_loc,loc_im,df_img=pd.read_csv("Data/images.csv"),df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_view/",output_log="Expansion/attr_exp_view_linear.csv"):
    df_expview_linear = pd.DataFrame(columns=["loc_id", "img_name", "attractiveness"])
    for loc_id in loc_im.keys():
        for pos in [0, 1, 2, 3]:
            pos_left = pos + 1
            pos_right = (pos + 1) % 4 + 1
            img_id_left = loc_im[loc_id]["img" + str(pos_left)]
            img_id_right = loc_im[loc_id]["img" + str(pos_right)]

            attr_left = df_vote[df_vote["img_id"] == img_id_left]["median"].values[0]
            attr_right = df_vote[df_vote["img_id"] == img_id_right]["median"].values[0]


            dir_left = df_img[df_img["id"] == img_id_left]["heading"].values[0]

            for k in [0, 1, 2]:
                dir_new = (dir_left + k*30 )%360

                attr_pred = attr_left
                if k==1:
                    attr_pred = attr_right

                newdat = {}
                newdat["loc_id"] = loc_id
                newdat["img_name"] = "EXPV_" + str(loc_id) + "_" + str(dir_new) + ".jpg"
                newdat["attractiveness"] = attr_pred
                df_expview_linear = df_expview_linear.append(newdat, ignore_index=True)

    df_expview_linear["loc_id"] = df_expview_linear["loc_id"].astype(int)
    df_expview_linear["attractiveness"] = df_expview_linear["attractiveness"].astype(int)
    df_expview_linear.to_csv(output_log, sep=",")
    return df_expview_linear

def expand_view(id,df_img,df_loc,loc_im,expand_view_dir = "../../DATA/Expansion_view/"):
    headings = df_img[df_img["loc_id"]==id]["heading"].values



    #get coordinate
    [lat,long] = df_loc[df_loc["loc_id"]==id].values[0][[1,2]]
    for i in [0,1,2,3]:
        current_img = "img"+str(i+1)
        right_img = "img" + str((i+1)%4+1)

        c_heading = headings[i]
        #check to the right
        for h_d in [i*10 for i in range(0,9)]:
            head = (c_heading + h_d)%360
            fname = expand_view_dir+"EXPV_"+str(id)+"_"+str(head)+".jpg"
            if not(os.path.isfile(fname)):
                download_image(lat, long, head, fname)
            else:
                print("File "+fname+" is already exists")

def expand_loc(id,df_img,df_loc,loc_im,expand_view_dir = "../../DATA/Expansion_loc/"):
    headings = df_img[df_img["loc_id"]==id]["heading"].values

    #get coordinate
    [lat,long] = df_loc[df_loc["loc_id"]==id].values[0][[1,2]]
    for i in [0,1,2,3]:
        head = headings[i]

        for la in[lat-0.0002, lat, lat+0.0002]:
            for lo in [long - 0.0002, long, long + 0.0002]:
                fname = expand_view_dir + "EXPL_" + str(id) + "_" + str(head) + "_"  + str(round(la,4)) +"_"+ str(round(lo,4)) + ".jpg"
                if not (os.path.isfile(fname)):
                    download_image(la, lo, head, fname)
                else:
                    print("File " + fname + " is already exists")

def label_locs(df_img=pd.read_csv("Data/images.csv"), df_vote=pd.read_csv("CrowdData/pilot_aggregates_part1.csv"),input_dir="../../DATA/Expansion_loc/",output_log="Expansion/attr_exp_loc.csv"):
    df_exploc = pd.DataFrame(columns=["loc_id","img_name","attractiveness"])

    files = [x for x in os.listdir(input_dir) if x.endswith('.jpg')]
    for loc_id in loc_im.keys():
        imgs = [x for x in files if x.split("_")[1]==str(loc_id)]
        for img_name in imgs:
            head = int(img_name.split("_")[2])
            img_id = int(df_img[(df_img["loc_id"]==loc_id) & (df_img["heading"]==head)]["id"])
            attr = int(df_vote[df_vote["img_id"]==img_id]["median"])
            newdat = {}
            newdat["loc_id"] = loc_id
            newdat["img_name"] = img_name
            newdat["attractiveness"] = attr
            df_exploc = df_exploc.append(newdat, ignore_index=True)
    df_exploc["loc_id"] = df_exploc["loc_id"].astype(int)
    df_exploc["attractiveness"] = df_exploc["attractiveness"].astype(int)
    df_exploc.to_csv(output_log,sep=",")
    return df_exploc

def get_patches(img_name, target_loc = "../../DATA/Patches2/", input_loc = "../../DATA/PILOT/"):
    img_path = input_loc+img_name
    img = image.load_img(img_path, target_size=(400, 600))
    for l in range(0, 600, 120):
        for w in range(0,400,80):
            cropped = img.crop((l, w, l + 120, w + 80))
            filename = target_loc+"PATCH_"+img_name.split(".")[0]+"_"+str(w)+"_"+str(l)+".jpg"
            cropped.save(filename)

def extract_patches(df_img):
    for path in df_img["filepath"].values:
        get_patches(path.split("/")[1])

img_data_filename = "Data/images.csv"
loc_im_filename = "Data/loc_im.csv"
loc_data_filename = "Data/locations.csv"