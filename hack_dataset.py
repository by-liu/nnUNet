import argparse
from batchgenerators.utilities.file_and_folder_operations import load_pickle, write_pickle

# parser = argparse.ArgumentParser()
# parser.add_argument("--data-dir", type=str)
# parser.add_argument("--split", type=str, default="0.7,0.1,0.2")

# data_path = "./nnunet_data/nnUNet_preprocessed/Task004_Hippocampus/splits_final.pkl"
data_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl.bak"
save_path = "./nnunet_data/nnUNet_preprocessed/Task005_Prostate/nnUNetPlansv2.1_plans_3D.pkl"

data = load_pickle(data_path)

data["plans_per_stage"][0]["batch_size"] = 2

print(data["plans_per_stage"][0])

write_pickle(data, save_path)