import os
import sys

import torch

# Add the parent directory of /exp to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from torch_uncertainty import TUTrainer
from torch_uncertainty.baselines.classification.deep_ensembles import DeepEnsemblesBaseline
from torch_uncertainty.datamodules import CIFAR10DataModule

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    dm = CIFAR10DataModule(
        root="./data",
        batch_size=128,
        eval_ood=True
    )

    dm.prepare_data()
    dm.setup()

    # model
    # args.task = "classification"
    model = DeepEnsemblesBaseline(
        num_classes=dm.num_classes,
        log_path="logs/cifar10/resnet18/standard",
        checkpoint_ids=[0, 1, 2, 3, 4],
        backbone="resnet",
        eval_ood=True
    )

    # Create the trainer that will handle the training
    trainer = TUTrainer(devices=1, max_epochs=10)

    # trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    ens_perf = trainer.test(model, dataloaders=dm.test_dataloader())
