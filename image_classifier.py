import os

import keras
import numpy as np
import warnings
import pandas as pd
from PIL.ImageOps import crop

from keras import metrics
from keras import optimizers

from keras.models import Model, Sequential
from keras.layers import Flatten, Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import model_from_json

from keras.utils import plot_model
import h5py
import datetime
from PIL import Image

from keras.wrappers.scikit_learn import KerasClassifier
import math
import random

def create_basic_model(do1, do2):
    input_shape = (3,224,224)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', trainable=False)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6', trainable=False)(x)
    x = Dropout(do1, name="dropout_1")(x)
    x = Dense(4096, activation='relu', name='fc7', trainable=True)(x)
    x = Dropout(do2, name="dropout_2")(x)
    x = Dense(205, activation='softmax', name='fc8', trainable=True)(x)

    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16')
    return model


def load_PLACES_weight(model,weights_path):
    model.load_weights(weights_path)
    return model

def preprocess_image(x):
    x[0], x[1], x[2] = x[2].transpose() - 105, x[1].transpose() - 114, x[0].transpose() - 116
    return x

def preprocess_dataset(dat):
    n = dat.shape[0]
    for i in range(0,n):
        x = dat[i]
        x[0], x[1], x[2] = x[2].transpose() - 105, x[1].transpose() - 114, x[0].transpose() - 116
        dat[i] = x
    return dat

def save_model(model,out_file,weight_file,layer_name=""):
    model_json = model.to_json()
    with open(out_file, "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    print("model saved")
    if(weight_file):
        model.save_weights(weight_file)
    print("weights saved")

def load_model(in_file,weight_file):
    json_file = open(in_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("model loaded")
    loaded_model = model_from_json(loaded_model_json)

    if (weight_file):
        loaded_model.load_weights(weight_file)
    print("weights loaded")

    return loaded_model

def read_img(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    x = image.img_to_array(img)
    return x

def load_dataset(path,ref_file, val_list, width):
    ref = pd.read_csv(ref_file)
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    vals = pd.read_csv(val_list)["img_id"].values.tolist()
    for index,row in ref.iterrows():
        filename = path + row["img_path"]
        img_id = row["img_id"]
        is_val = (img_id in vals)

        img = ""
        if is_val:
            img = image.load_img(filename, target_size=(224, 224))
        else:
            img = image.load_img(filename, target_size=(width, width))

        x = image.img_to_array(img)
        cls = row["median"]
        y = []

        if cls==1:
            y = [0, 0, 0, 0]
        elif cls == 2:
            y = [1, 0, 0, 0]
        elif cls == 3:
            y = [1, 1, 0, 0]
        elif cls == 4:
            y = [1, 1, 1, 0]
        elif cls == 5:
            y = [1, 1, 1, 1]
        else:
            y = [1, 1, 0, 0]

        if is_val:
            X_val.append(x)
            Y_val.append(y)
        else:
            X_train.append(x)
            Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    return [X_train, Y_train, X_val, Y_val]

def load_expansion(ref_file, loc_list, width = 224, path = "../../DATA/Expansion_view/"):
    ref = pd.read_csv(ref_file)
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    locs = pd.read_csv(loc_list)["loc_id"].values.tolist()
    for index,row in ref.iterrows():
        filename = path + row["img_name"]
        loc_id = int(row["img_name"].split("_")[1])
        is_train = (loc_id in locs)

        img = ""
        if is_train:
            img = image.load_img(filename, target_size=(width, width))

            x = image.img_to_array(img)
            cls = row["attractiveness"]
            y = []

            if cls==1:
                y = [0, 0, 0, 0]
            elif cls == 2:
                y = [1, 0, 0, 0]
            elif cls == 3:
                y = [1, 1, 0, 0]
            elif cls == 4:
                y = [1, 1, 1, 0]
            elif cls == 5:
                y = [1, 1, 1, 1]
            else:
                y = [1, 1, 0, 0]

            X_train.append(x)
            Y_train.append(y)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return [X_train, Y_train]


def load_exp_view(path, ref_file, width):
    ref = pd.read_csv(ref_file)
    X_train = []
    Y_train = []
    for index,row in ref.iterrows():
        filename = path + row["img_name"]
        img = image.load_img(filename, target_size=(width, width))
        x = image.img_to_array(img)
        cls = row["attractiveness"]
        y = []

        if cls==1:
            y = [0, 0, 0, 0]
        elif cls == 2:
            y = [1, 0, 0, 0]
        elif cls == 3:
            y = [1, 1, 0, 0]
        elif cls == 4:
            y = [1, 1, 1, 0]
        elif cls == 5:
            y = [1, 1, 1, 1]
        else:
            y = [1, 1, 0, 0]

        X_train.append(x)
        Y_train.append(y)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return [X_train,Y_train]

def predict_attractiveness(model,img_path):
    x = read_img(img_path)
    x = preprocess_image(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds


def class_accuracy(y_true,y_pred):
    return K.mean(K.equal(K.sum(K.abs(K.round(y_pred) - y_true),axis=-1),0))


def train_model(model,X_train,Y_train,X_val,Y_val,callbacks_list=[], batch_size = 20, lr = 0.01):
    # dimensions of our images.
    img_width, img_height = 224, 224
    nb_train_samples = X_train.shape[0]
    nb_validation_samples = X_val.shape[0]
    epochs = 1
    batch_size = batch_size

    optim = keras.optimizers.SGD(lr=lr, momentum=0.9, decay=0, nesterov=True);

    model.compile(loss='binary_crossentropy',
                  optimizer=optim,
                  #metrics=[class_accuracy]
                  metrics=[]
                  )

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        shear_range=0.1,
        rotation_range=5,
        channel_shift_range = 5,
        zoom_range=[0.8,1.2]
        )

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        X_train, Y_train,
        batch_size=batch_size
    )

    test_generator = test_datagen.flow(
        X_val, Y_val,
        batch_size=batch_size
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks = callbacks_list)
    return model

def start_model(do1, do2, deep = True):
    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model(do1, do2)
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    if(deep):
        last = model.layers[-2].output
    else:
        last = model.layers[-3].output
    x = Dense(4096, activation='relu', name='fc7new', trainable=True)(last)
    x = Dense(4, activation='sigmoid', name='predictor', trainable=True)(last)
    model = Model(model.input, x, name='newModel')

    return model

def decode_class(bins):
    if(bins[0]==0):
        return 1
    elif (bins[1]==0):
        return 2
    elif (bins[2] == 0):
        return 3
    elif (bins[3] == 0):
        return 4
    elif (bins[3] == 1):
        return 5
    return 3

def get_evaluation(Y_pred,Y_true):
    Y_pred = binarize_result(Y_pred)
    n = Y_pred.shape[0]
    correct = 0
    sum_error = 0.0
    for i in range(0,n):
        y_pred = decode_class(Y_pred[i])
        y_true = decode_class(Y_true[i])

        if(y_pred == y_true):
            correct = correct+1

        sum_error = sum_error + (y_true - y_pred)**2

    accuracy = float(correct)/float(n)
    rmse = math.sqrt(sum_error/float(n))
    return [accuracy, rmse]

def convert_weight(h5_file = '../../CNN/PredefinedModels/vgg_places.h5',out_file = '../../CNN/PredefinedModels/vgg_places_keras.h5'):
    res = h5py.File(out_file,'r+')
    f = h5py.File(h5_file,'r')
    ff = f[u'data'].values()
    at = np.array(f.keys()).astype("|S12")
    for dat in ff:
        for dts in dat.values():
            nm = dts.name.split("/")[2]
            idx = dts.name.split("/")[3]
            dtsname = "/" + nm + "/" + nm+"/"
            if(idx=="0"):
                dtsname = dtsname +"kernel:0"
            elif(idx=="1"):
                dtsname = dtsname +"bias:0"
            print (dts.name+" => "+dtsname)
            del res[dtsname]
            res[dtsname] = dts.value.transpose()
    res.close()
    return res

def get_places_ref(ref_path = '../../CNN/PredefinedModels/categoryIndex_places205.csv'):
    df_cat = pd.read_csv(ref_path,delimiter=" ")
    res = {}
    for idx,row in df_cat.iterrows():
        ctg = row["category"].split("/")[2]
        res[str(row["id"])] = ctg
    return res

def decode_scene(preds,reff=get_places_ref()):
    sorted = np.flipud(preds.argsort())
    for i in range(0,5):
        ct = sorted[i]
        print("["+str(preds[ct])+"] "+reff[str(ct)])


def classify_scene(model,img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_image(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    decode_scene(preds)
    return preds

def predict_scenes():
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    dat = pd.read_csv(ref)
    reff = get_places_ref()
    [X,Y] = load_dataset(path, ref)
    X = preprocess_dataset(X)

    WEIGHTS_PATH = '../../CNN/PredefinedModels/vgg_places_keras.h5'
    model = create_basic_model()
    model = load_PLACES_weight(model, WEIGHTS_PATH)

    preds = model.predict(X)

    df_scene = pd.DataFrame(
        columns=["img_id", "scene1", "scene2", "scene3", "scene4", "scene5"])

    for i in range(0,preds.shape[0]):
        scene = {}
        pred_i = preds[i]
        scene["img_id"] = dat["img_id"][i]
        sorted = np.flip(pred_i.argsort(), 0)
        for j in range(1, 6):
            ct = sorted[j-1]
            scene["scene"+str(j)] = "'"+reff[str(ct)]+"'"
        df_scene = df_scene.append(scene, ignore_index=True)

        df_scene["img_id"] = df_scene["img_id"].astype(int)
    df_scene.to_csv("Data/SceneFeatures.csv")


def crop_im224(Xbig,xx,yy):
    img = image.array_to_img(Xbig)
    cropped = img.crop((xx,yy,xx+224,yy+224))
    return image.img_to_array(cropped)

def get_crops(X,Y):
    X_crop = []
    Y_crop = []
    n = Y.shape[0]

    for i in range(0,n):
        #crop_center
        X_crop.append(crop_im224(X[i],88,88))
        Y_crop.append(Y[i])

        # crop1
        X_crop.append(crop_im224(X[i], 0, 0))
        Y_crop.append(Y[i])

        # crop2
        X_crop.append(crop_im224(X[i], 0, 176))
        Y_crop.append(Y[i])

        # crop3
        X_crop.append(crop_im224(X[i], 176, 0))
        Y_crop.append(Y[i])

        # crop4
        X_crop.append(crop_im224(X[i], 176, 176))
        Y_crop.append(Y[i])

    X_crop = np.array(X_crop)
    Y_crop = np.array(Y_crop)
    return [X_crop, Y_crop]

def convert_to_binary(pred):
    res = np.zeros(pred.shape).astype(int)
    res[pred>0.5] = 1
    return res

def binarize_result(preds):
    p = np.zeros(preds.shape).astype(int)
    for i in range(0,preds.shape[0]):
        p[i] = convert_to_binary(preds[i])
    return p

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, batch, logs={}):
        print(self.losses)

def exp_training(name, batch_size = 5, lr_init = 0.01, decay = 0.1, do1=0.2,do2=0.2, epochs=10):
    path="../Website/crowdsourcing/public/images/"
    ref="CrowdData/pilot_aggregates_part1.csv"
    val_list = "CrowdData/val_list.csv"
    [X_train_ori, Y_train_ori, X_val_ori, Y_val_ori] = load_dataset(path, ref, val_list, 224)
    #[X_train_big, Y_train_big, X_val_big, Y_val_big] = load_dataset(path, ref, val_list, 400)

    #activate cropping
    #[X_train_big, Y_train_big] = get_crops(X_train_big, Y_train_big)

    #X_train = np.concatenate((X_train_ori, X_train_big), axis=0)
    #Y_train = np.concatenate((Y_train_ori, Y_train_big), axis=0)

    X_train = X_train_ori
    Y_train = Y_train_ori

    X_train = preprocess_dataset(X_train)
    X_train_ori = preprocess_dataset(X_train_ori)
    X_val_ori = preprocess_dataset(X_val_ori)

    model = start_model(do1, do2)

    logf = open("MODELS/log_"+name+".txt", 'w')
    logf.write("timestamp,name,batch_size,LR,dropout1,dropout2,decay,epoch,acc_train,acc_val,rmse_train,rmse_val\n")
    logf.flush()

    best_rmse = 99
    lr = lr_init
    for ep in range(1, epochs + 1):
        model = train_model(model,X_train,Y_train,X_val_ori,Y_val_ori, batch_size = batch_size, lr = lr)
        lr = lr*(1-decay)

        Y_train_pred = model.predict(X_train_ori)
        [acc_train, rmse_train] = get_evaluation(Y_train_pred, Y_train_ori)

        Y_val_pred = model.predict(X_val_ori)
        [acc_val, rmse_val] = get_evaluation(Y_val_pred, Y_val_ori)

        wfile = "MODELS/" + name + "_epoch_" + str(ep) + "_err_" + str(
            round(rmse_train, 2)) + "_" + str(round(rmse_val, 2)) + ".h5"
        if (rmse_val <= best_rmse):
            save_model(model, "MODELS/complete_model.json", wfile)
            best_rmse = rmse_val

        log = str(datetime.datetime.now()) + "," + name + "," + str(batch_size) + "," + str(lr_init) + "," + str(
            do1) + "," + str(do2) + "," + str(decay) + "," + str(ep) + "," + str(round(acc_train, 2)) + "," + str(
            round(acc_val, 2)) + "," + str(round(rmse_train, 2)) + "," + str(round(rmse_val, 2)) + "\n"
        logf.write(log)
        logf.flush()
        print(log)

def train_expansion(name, batch_size = 5, lr_init = 0.01, decay = 0.1, do1=0.2,do2=0.2, epochs=10, expandset = "Expansion/attr_exp_view_linear.csv", deep=True, init_weight = "", exppath = "../../DATA/Expansion_view/"):
    path = "../Website/crowdsourcing/public/images/"
    ref = "CrowdData/pilot_aggregates_part1.csv"
    val_list = "CrowdData/val_list.csv"
    loc_train_list = "CrowdData/loc_train_list.csv"
    [X_train_ori, Y_train_ori, X_val_ori, Y_val_ori] = load_dataset(path, ref, val_list, 224)

    [X_train, Y_train] = load_expansion(expandset, loc_train_list, path = exppath)

    X_train = preprocess_dataset(X_train)
    X_train_ori = preprocess_dataset(X_train_ori)
    X_val_ori = preprocess_dataset(X_val_ori)

    model = start_model(do1, do2, deep = deep)

    if(init_weight !=""):
        model.load_weights(init_weight)

    logf = open("MODELS/log_" + name + ".txt", 'w')
    logf.write("timestamp,name,batch_size,LR,dropout1,dropout2,decay,epoch,acc_train_ex,acc_train,acc_val,rmse_train_ex,rmse_train,rmse_val\n")
    logf.flush()

    best_rmse = 99
    lr = lr_init
    for ep in range(1, epochs + 1):
        model = train_model(model, X_train, Y_train, X_val_ori, Y_val_ori, batch_size=batch_size, lr=lr)
        lr = lr * (1 - decay)

        Y_train_ex_pred = model.predict(X_train)
        [acc_train_ex, rmse_train_ex] = get_evaluation(Y_train_ex_pred, Y_train)

        Y_train_pred = model.predict(X_train_ori)
        [acc_train, rmse_train] = get_evaluation(Y_train_pred, Y_train_ori)

        Y_val_pred = model.predict(X_val_ori)
        [acc_val, rmse_val] = get_evaluation(Y_val_pred, Y_val_ori)

        wfile = "MODELS/" + name + "_epoch_" + str(ep) + "_err_" + str(
            round(rmse_train, 2)) + "_" + str(round(rmse_val, 2)) + ".h5"
        if (rmse_val <= best_rmse):
            save_model(model, "MODELS/complete_model.json", wfile)
            best_rmse = rmse_val

        log = str(datetime.datetime.now()) + "," + name + "," + str(batch_size) + "," + str(lr_init) + "," + str(
            do1) + "," + str(do2) + "," + str(decay) + "," + str(ep) + "," + str(round(acc_train_ex, 2)) + "," + str(round(acc_train, 2)) + "," + str(
            round(acc_val, 2)) + "," + str(round(rmse_train_ex, 2)) + "," + str(round(rmse_train, 2)) + "," + str(round(rmse_val, 2)) + "\n"
        logf.write(log)
        logf.flush()
        print(log)

def observe_attractiveness(img_dir= "../../DATA/Amsterdam/", logfile="../../DATA/amsterdam_log.csv", name = "ams"):
    imgs = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]

    logf = open("MODELS/attractiveness_" + name + ".txt", 'w')
    logf.write(
        "lat,long,heading,attractiveness\n")
    logf.flush()

    model = start_model(0, 0.2)
    model.load_weights("MODELS/BEST/best_model.h5")

    for im in imgs:
        img_path =img_dir+im

        img_splitted = im.split("_")
        ev = {}
        ev["lat"] = round(float(img_splitted[1]),3)
        ev["long"] = round(float(img_splitted[2]),3)
        ev["heading"] = img_splitted[3].split(".")[0]

        #evaluate attractiveness of image
        x = read_img(img_path)
        x = preprocess_image(x)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        outp = preds[0]
        outbin = binarize_result(outp)
        att = decode_class(outbin)
        ev["attractiveness"] = att

        logf.write(str(ev["lat"])+","+str(ev["long"])+","+str(ev["heading"])+","+str(outp)+","+str(ev["attractiveness"]+"\n"))
        logf.flush()

    logf.close()

def classify_patches(img_dir= "../../DATA/Patches/", logfile="../../DATA/patches_log.csv", name = "patch"):
    imgs = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]

    logf = open("MODELS/attractiveness_" + name + ".txt", 'w')
    logf.write(
        "filename,attractiveness\n")
    logf.flush()

    model = start_model(0, 0.2)
    model.load_weights("MODELS/BEST/best_model.h5")

    for im in imgs:
        img_path =img_dir+im

        #evaluate attractiveness of image
        x = read_img(img_path)
        x = preprocess_image(x)
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        outp = preds[0]
        outbin = binarize_result(outp)
        att = decode_class(outbin)

        logf.write(im+","+str(outp)+","+str(att)+"\n")

    logf.close()

def aggregate_city_attractiveness(input_file="../../DATA/amsterdam_log.csv",output_file="../../DATA/amsterdam_attractiveness.csv"):
    df_input = pd.read_csv(input_file)
    agg_idx = {}
    df_aggregate = pd.DataFrame(columns=["lat", "long", "h1", "att1", "att2", "att3", "att4", "attractiveness"])

    for idx,row in df_input.iterrows():
        ky = str(row["lat"])+"|"+str(row["long"])
        if ky not in agg_idx:
            agg_idx[ky] = 1
            agg = {}
            agg["lat"] = round(row["lat"],3)
            agg["long"] = round(row["long"],3)
            agg["h1"] = 0

            df_filtered = df_input[(df_input["lat"] == row["lat"]) & (df_input["long"] == row["long"])]
            for ii in [1,2,3,4]:
                agg["att"+str(ii)] = df_filtered[df_filtered["heading"]==90*(ii-1)]["attractiveness"].values[0]

            agg["attractiveness"] = (agg["att1"]+agg["att2"]+agg["att3"]+agg["att4"])/4
            df_aggregate = df_aggregate.append(agg,ignore_index=True)

    df_aggregate["h1"] = df_aggregate["h1"].astype("int")
    for ii in [1,2,3,4]:
        df_aggregate["att"+str(ii)] = df_aggregate["att"+str(ii)].astype("int")
    df_aggregate["attractiveness"] = df_aggregate["attractiveness"].astype("int")
    df_aggregate.to_csv(output_file)

path = "../Website/crowdsourcing/public/images/"
ref = "CrowdData/pilot_aggregates_part1.csv"
val_list = "CrowdData/val_list.csv"