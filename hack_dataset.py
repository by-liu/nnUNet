import argparse
import os.path as osp
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_pickle, write_pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-dir", type=str)
# parser.add_argument("--split", type=str, default="0.7,0.1,0.2")

# data_path = "./nnunet_data/nnUNet_preprocessed/Task004_Hippocampus/splits_final.pkl"
# data_dir = "/home/bingyuan/scratch/Data/Task003_Liver"
data_dir = "/home/bingyuan/scratch/Data/Task216_AMOS2022_task1"

# #data_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl.bak"
# #save_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl"

# data_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D.pkl.bak")
# save_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D.pkl")

# data = load_pickle(data_path)

# for i in range(len(data["plans_per_stage"])):
#     data["plans_per_stage"][i]["batch_size"] = 2
#     print(data["plans_per_stage"][i])


# write_pickle(data, save_path)


def load_list_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        return [line.strip() for line in lines]


data_dir = "/home/bliu/work/Data/nnunet_data/nnUNet_preprocessed/Task216_AMOS2022_task1"

data_path = osp.join(data_dir, "splits_final.pkl.bak")
save_path = osp.join(data_dir, "splits_final.pkl")

data = load_pickle(data_path)

print("train samples : ", data[0]["train"].shape)
print("val samples : ", data[0]["val"].shape)

train = load_list_from_file(osp.join(data_dir, "tr.txt"))
val = load_list_from_file(osp.join(data_dir, "ts.txt"))

train = np.array(train)
val = np.array(val)

data[0]["train"] = train
data[0]["val"] = val
data_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D.pkl")
save_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D_b4.pkl")

data = load_pickle(data_path)

for i in range(len(data["plans_per_stage"])):
    data["plans_per_stage"][i]["batch_size"] = 4
    print(data["plans_per_stage"][i])

print("------------------------")
print("train samples : ", data[0]["train"].shape)
print("val samples : ", data[0]["val"].shape)

write_pickle(data, save_path)
