import urllib
import random
import requests
import os
import random
import numpy as np

n_image = 1000
start_counter = 1
start_c = 1

width = 600
height = 400
size = str(width) + "x" + str(height)
pitch = -0.76

eps = 0.00005

image_loc = "../../Dataset/Images_PILOT2/"
mass_loc = "../../Dataset/Collection/"
log_loc = "../../Dataset/log_PILOT2.txt"
prefix = "GSV_PILOT_"
# coordinates

min_lat = 52.29
max_lat = 52.42
min_long = 4.73
max_long = 4.98

def generate_random_point():
    rdm_lat = random.uniform(min_lat, max_lat)
    rdm_long = random.uniform(min_long, max_long)
    return [rdm_lat, rdm_long]


def get_heading(lat, long):

    #default
    if(1):
        return random.randint(0,359)

    d_lat = lat + 10 * eps
    d_long = long
    found = 0
    head = 90
    while (not (found) and head < 180):
        if (image_valid(d_lat, d_long)):
            return head
        d_lat = d_lat - eps
        d_long = d_long + eps
        head = head + 9

    return 0


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

def process_location(lat, long, it):
    if (not (image_valid(lat, long))):
        return 0
    location = str(lat) + ";" + str(long)
    heading = get_heading(lat, long)

    log = ""

    for k in range(0, 4):
        head = (heading + 90 * k)%360
        imname = prefix+str(it) + "_" + str(k+1) + ".jpg"
        filename = image_loc + imname
        download_image(lat, long, head, filename)
        log1 = str(it)+";"+imname+";"+str(lat)+";"+str(long)+";"+str(head)
        log = log+"\n"+log1

    #print(location + " heading:" + str(heading))
    return log

def start_crawling():
    logfile = open(log_loc,"a")
    for iter in range(start_counter, start_counter + n_image):
        valid = 0
        while (not (valid)):
            random_point = generate_random_point()
            log = process_location(format(random_point[0],".10f"), format(random_point[1],".10f"), iter)
            if(log != 0):
                valid = 1
                logfile.write(log)
        if(iter%100 == 0):
            print("index "+str(iter)+" has been created")
    logfile.close()

def start_defined_crawling():
    logfile = open(log_loc, "a")
    input_filename = "../../Dataset/find_golden.txt"
    input_file = open(input_filename, "r")
    iter = start_c
    for line in input_file:
        fields = line.split(",")
        if(len(fields) == 2):
            log = process_location(format(float(fields[0]),".10f"), format(float(fields[1]),".10f"), iter)
            if (log != 0):
                valid = 1
                logfile.write(log)
        iter = iter + 1
    logfile.close()
    input_file.close()

def crawl_mass(step_lat, step_long, lat_from, lat_to, long_from,long_to):
    for lat in np.arange(lat_from, lat_to, step_lat):
        for long in np.arange(long_from, long_to, step_long):
            if image_valid(lat, long):
                location = str(lat) + ";" + str(long)
                heading = 0

                for k in range(0, 4):
                    head = (heading + 90 * k) % 360
                    imname = "GSV_"+str(lat) + "_" + str(long)+"_"+str(head)+".jpg"
                    filename = mass_loc + imname
                    download_image(lat, long, head, filename)
                    print(filename+" downloaded..")