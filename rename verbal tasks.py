import pandas as pd
from os import listdir
import csv

def file_rename():
    path = "Data/Verbal Tasks original/"
    newpath = "Data/Verbal Tasks original copy/"
    allfiles = listdir(path)
    csvfiles = [x for x in allfiles if '.csv' in x]

    for file in csvfiles:
        if 'piscines' in file:
            try:
                filename = path + file
                with open(filename) as csv_file:


            except:
                print('problem with:', file)

    return None