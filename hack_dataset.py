import argparse
import os.path as osp
from batchgenerators.utilities.file_and_folder_operations import load_pickle, write_pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-dir", type=str)
# parser.add_argument("--split", type=str, default="0.7,0.1,0.2")

# data_path = "./nnunet_data/nnUNet_preprocessed/Task004_Hippocampus/splits_final.pkl"
data_dir = "/home/bingyuan/scratch/Data/Task003_Liver"

#data_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl.bak"
#save_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl"

data_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D.pkl.bak")
save_path = osp.join(data_dir, "nnUNetPlansv2.1_plans_3D.pkl")

data = load_pickle(data_path)

for i in range(len(data["plans_per_stage"])):
    data["plans_per_stage"][i]["batch_size"] = 2
    print(data["plans_per_stage"][i])


write_pickle(data, save_path)
