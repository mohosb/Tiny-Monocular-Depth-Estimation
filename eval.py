import torch
import numpy as np
import torchvision.transforms.v2 as tvv2
from torch.utils.data import random_split, DataLoader
from utils import evaluate, ScaleInvariantMSELoss, ScaleInvariantRMSELoss, AbsoluteRelativeLoss, DepthAnything
from data import RawNYUDepthV2

if __name__ == '__main__':
    DEVICE = 'cuda:0'
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    DATA_PATH = '/home/mohosb/seagate_exp/nyu_depth_v2'
    MODEL_PATH = None  # Path to finetuned model weights or None for baseline

    DEVICE = torch.device(DEVICE)

    teacher_model = DepthAnything('large').eval().to(DEVICE)

    student_model = DepthAnything('small').eval().to(DEVICE)
    if MODEL_PATH is not None:
        student_model.load_state_dict(torch.load(MODEL_PATH, DEVICE))

    transform = tvv2.Compose([
        tvv2.Resize(IMAGE_SIZE),
        tvv2.CenterCrop(IMAGE_SIZE),
        np.array,
    ])
    datasets = {}
    datasets['all'] = random_split(
        RawNYUDepthV2(DATA_PATH, '*room*', transform),
        (0.9, 0.1),
        torch.Generator().manual_seed(42)
    )[1]
    datasets['living_room'] = RawNYUDepthV2('', '', transform)
    datasets['living_room'].paths = list(filter(lambda p: 'living_room' in p, datasets['all'].dataset.paths))
    datasets['bedroom'] = RawNYUDepthV2('', '', transform)
    datasets['bedroom'].paths = list(filter(lambda p: 'bedroom' in p, datasets['all'].dataset.paths))

    criterions = [
        ScaleInvariantRMSELoss(use_log=True, reduction='none'),
        AbsoluteRelativeLoss(reduction='none')
    ]

    for criterion in criterions:
        for ds_name, ds in datasets.items():
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, multiprocessing_context='spawn', persistent_workers=True, num_workers=4)
            result = evaluate(teacher_model, student_model, criterion, loader)
            print(f'[{criterion.__class__.__name__}, {ds_name}]: {result:.6f}')

