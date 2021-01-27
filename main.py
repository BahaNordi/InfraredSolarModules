# import glob
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import shutil
# from torchvision import transforms
# from torchvision import models
# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# from torch.optim import lr_scheduler
# from torch import optim
# from torchvision.datasets import ImageFolder
# from torchvision.utils import make_grid
# import warnings
#
# warnings.filterwarnings("ignore")
# import time
#
# import os
# import random
# import shutil
# import json
# import shutil
#
# with open('/home/baha/codes/solar_data/InfraredSolarModules/module_metadata.json') as f:
#     data = json.load(f)
#
#
#
# # files = os.listdir(source)
#
#
# total_class = ["Vegetation", "Cell", "Cell-Multi", "Cracking", "Hot-Spot", "Hot-Spot-Multi",
#                "Shadowing", "Diode", "Diode-Multi", "Soiling", "Offline-Module", "No-Anomaly"]
#
# count = {"Vegetation": 0, "Cell": 0, "Cell-Multi": 0, "Cracking": 0, "Hot-Spot": 0, "Hot-Spot-Multi": 0, "Shadowing": 0,
#          "Diode": 0, "Diode-Multi": 0, "Soiling": 0, "Offline-Module": 0, "No-Anomaly": 0}
# for label in total_class:
#     for item in data:
#         if label == data[item]['anomaly_class']:
#             shutil.move(os.path.join(source, item)+'.jpg', dest)
#             print(item)
#             count[label] += 1
# print(count)


# separate 12 classes in different folders
# label = "No-Anomaly"
#
# for item in data:
#     if label == data[item]['anomaly_class']:
#         shutil.move(os.path.join(source, item)+'.jpg', dest)

# for label in total_class:
#     source = '/home/baha/codes/solar_data/InfraredSolarModules/'+label
#     dest = '/home/baha/codes/solar_data/InfraredSolarModules/train/'+label
#     files = os.listdir(source)
#     no_of_files = int(len(files) // 1.428)
#     for file_name in random.sample(files, no_of_files):
#         shutil.move(os.path.join(source, file_name), dest)
