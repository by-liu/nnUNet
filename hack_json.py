import os.path as osp
import json


def load_list_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        return [line.strip() for line in lines]


data_dir = "/home/bliu/work/Data/nnunet_data/nnUNet_raw_data/Task216_AMOS2022_task1"

json_path = osp.join(data_dir, "dataset.json.bak")

data = json.load(open(json_path, "r"))

print(len(data["training"]))

test_list = load_list_from_file(osp.join(data_dir, "ts.txt"))

for name in test_list:
    data["training"].append(
        {
            "image": osp.join("imagesTr", name),
            "label": osp.join("labelsTr", name),
        }
    )

print(len(data["training"]))

# save json
save_path = osp.join(data_dir, "dataset.json")

json.dump(data, open(save_path, "w"))
