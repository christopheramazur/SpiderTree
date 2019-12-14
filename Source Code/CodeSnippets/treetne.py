import ete3
import os

def get_names(source = "./SnipDat/Total/"):
    directories = os.listdir(source)
    names = []
    for i in range(0, len(directories)):
        names.append(source + directories[i] + "/")
