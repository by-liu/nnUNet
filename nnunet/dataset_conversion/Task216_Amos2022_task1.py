from batchgenerators.utilities.file_and_folder_operations import *
import shutil

from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data

if __name__ == '__main__':
    amos_base = '/home/bliu/work/Data/nnunet_data/amos22'

    # Arbitrary task id. This is just to ensure each dataset ha a unique number. Set this to whatever ([0-999]) you
    # want
    task_id = 216
    task_name = "AMOS2022_task1"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    # dataset_json_source = load_json(join(amos_base, 'task1_dataset.json'))
    dataset_json_source = load_json(join(amos_base, 'dataset.json'))

    training_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['training']]

    for tr in training_identifiers:
        print(tr)
        shutil.copy(join(amos_base, 'imagesTr', tr + '.nii.gz'), join(imagestr, f'{tr}_0000.nii.gz'))
        shutil.copy(join(amos_base, 'labelsTr', tr + '.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    # import ipdb; ipdb.set_trace()

    # test_identifiers = [i.split('/')[-1][:-7] for i in dataset_json_source['validation']]
    # test_identifiers = [i.split('/')[-1][:-7] for i in dataset_json_source['test']]
    test_identifiers = [i['image'].split('/')[-1][:-7] for i in dataset_json_source['validation']]

    for ts in test_identifiers:
        print(ts)
        shutil.copy(join(amos_base, 'imagesVa', ts + '.nii.gz'), join(imagests, f'{ts}_0000.nii.gz'))
        shutil.copy(join(amos_base, 'labelsVa', ts + '.nii.gz'), join(labelsts, f'{ts}.nii.gz'))
        # shutil.copy(join(amos_base, 'imagesTs', ts + '.nii.gz'), join(imagests, f'{ts}_0000.nii.gz'))

    generate_dataset_json(join(out_base, 'dataset.json'), imagestr, imagests, ("CT", ), dataset_json_source['labels'],
                          foldername)
