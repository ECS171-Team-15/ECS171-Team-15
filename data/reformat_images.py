import os
import numpy as np
import csv
from PIL import Image


def get_image_names(data_type: str) -> list:
    if data_type == 'covid':
        return os.listdir("../Images-processed/COVID/Useable")
    return os.listdir("../Images-processed/CT_NonCOVID/Usable")


def get_mean_dimensions():
    covid_names = get_image_names('covid')
    non_covid_names = get_image_names('non-covid')
    height = []
    width = []
    for name in covid_names:
        im = Image.open(f"../Images-processed/COVID/Useable/{name}")
        w, h = im.size
        height.append(h)
        width.append(w)

    for name in non_covid_names:
        im = Image.open(f"../Images-processed/CT_NonCOVID/Usable/{name}")
        w, h = im.size
        height.append(h)
        width.append(w)

    return sum(height)//len(height), sum(width)//len(width)


def get_resized_image(height: int, width: int):
    covid_names = get_image_names('covid')
    non_covid_names = get_image_names('non-covid')
    covid_image_objs = []
    non_covid_image_objs = []
    for name in covid_names:
        im = Image.open(f"../Images-processed/COVID/Useable/{name}")
        im = im.resize((width, height))
        covid_image_objs.append(im.convert("L"))
    for name in non_covid_names:
        im = Image.open(f"../Images-processed/CT_NonCOVID/Usable/{name}")
        im = im.resize((height, width))
        non_covid_image_objs.append(im.convert("L"))
    return covid_image_objs, non_covid_image_objs

if __name__ == '__main__':
    height, width = get_mean_dimensions()
    covid_obj, non_covid_obj = get_resized_image(height=height, width=width)
    csv_file = []
    header = [f"pixel{i+1}" for i in range(width*height)] + ["class"]
    csv_file.append(header)
    for obj in covid_obj:
        pixel_array = np.array(obj).flatten()
        pixel_array = np.append(pixel_array, 0).tolist()
        csv_file.append(pixel_array)
    for obj in non_covid_obj:
        pixel_array = np.array(obj).flatten()
        pixel_array = np.append(pixel_array, 0).tolist()
        csv_file.append(pixel_array)
    with open("full_data.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(csv_file)