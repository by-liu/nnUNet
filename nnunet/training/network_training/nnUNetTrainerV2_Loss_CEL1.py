from nnunet.training.loss_functions.ce_bias import CrossEntropyWithL1
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNet_variants.architectural_variants import nnUNetTrainerV2_noDeepSupervision


class nnUNetTrainerV2_Loss_CEL1(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                              unpack_data, deterministic, fp16)

        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        loss_params = {
            "mode": "multiclass",
            "alpha": 1.0,
            "temp": 20
        }
        self.loss = CrossEntropyWithL1(**loss_params)
