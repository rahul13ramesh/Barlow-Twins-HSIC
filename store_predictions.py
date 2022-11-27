import glob
import numpy as np
import os

fname = "barlow1000"

def ge
    ckpt_list = []
    for fl in glob.glob(fname + "/*.ckpt"):
        print(fl)
        orig_name = fl
        fl = fl.split("/")[-1]
        fl = fl.split(".")[0]

        ep, step = fl.split("-")
        ep = ep.split("=")[-1]
        step = step.split("=")[-1]
        print(ep, step)

        ckpt_list.append((int(ep), int(step), orig_name))

    ckpt_list = sorted(ckpt_list)

    # Get dataloader

    dataset_test = lightly.data.LightlyDataset.from_torch_dataset(torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transforms,
        download=True))

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Load network
    # Compute prototypes
    # Store predictions

