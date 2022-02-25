import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'radiate_sdk/'))
import json

import numpy as np

import radiate

root_path = './data/'
sequence_names = os.listdir(root_path)
    
for sequence_name in sequence_names:
    seq = radiate.Sequence(f"{root_path}/{sequence_name}", 'radiate_sdk/config/config.yaml')
    with open(f"{root_path}/{sequence_name}/meta.json", "r") as file:
        meta = json.load(file)
    
    c = len(seq.timestamp_radar["time"])
    
    full_list = np.arange(c)
    np.random.shuffle(full_list)

    list_radar = full_list
    list_lidar = [seq.get_id(np.array(seq.timestamp_radar["time"][x]), seq.timestamp_lidar, seq.config['sync']['lidar'])[0] for x in list_radar]
    
    if meta["set"].split("_")[0] == "train":
        if not os.path.exists(f"Complex-YOLOv3/split/{meta['set']}/{sequence_name}"):
            os.makedirs(f"Complex-YOLOv3/split/{meta['set']}/{sequence_name}")

        if meta["set"] == "train_good_weather":
            np.savetxt(f"Complex-YOLOv3/split/{meta['set']}/{sequence_name}/train_split_radar.txt", np.array(list_radar) + 1)
            np.savetxt(f"Complex-YOLOv3/split/{meta['set']}/{sequence_name}/train_split_lidar.txt", np.array(list_lidar) + 1)
        elif meta["set"] == "train_good_and_bad_weather":
            np.savetxt(f"Complex-YOLOv3/split/train_good_and_bad_weather/{sequence_name}/train_split_radar.txt", np.array(list_radar) + 1)
            np.savetxt(f"Complex-YOLOv3/split/train_good_and_bad_weather/{sequence_name}/train_split_lidar.txt", np.array(list_lidar) + 1)
    elif meta["set"] == "test":
        if sequence_name.split("_")[0] in ["city", "junction", "motorway", "rural"]:
            
            if not os.path.exists(f"Complex-YOLOv3/split/test_good_weather/{sequence_name}"):
                os.makedirs(f"Complex-YOLOv3/split/test_good_weather/{sequence_name}")
                
            np.savetxt(f"Complex-YOLOv3/split/test_good_weather/{sequence_name}/test_split_radar.txt", np.array(list_radar) + 1)
            np.savetxt(f"Complex-YOLOv3/split/test_good_weather/{sequence_name}/test_split_lidar.txt", np.array(list_lidar) + 1)
        else:
            if not os.path.exists(f"Complex-YOLOv3/split/test_bad_weather/{sequence_name}"):
                os.makedirs(f"Complex-YOLOv3/split/test_bad_weather/{sequence_name}")
                
            np.savetxt(f"Complex-YOLOv3/split/test_bad_weather/{sequence_name}/test_split_radar.txt", np.array(list_radar) + 1)
            np.savetxt(f"Complex-YOLOv3/split/test_bad_weather/{sequence_name}/test_split_lidar.txt", np.array(list_lidar) + 1)
    else:
        print("AAAAAA")
