import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import numpy as np
import torchvision.transforms.v2 as tvv2
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data import RawNYUDepthV2
from utils.training import *
from utils.model import DepthAnything
from collections import deque
from itertools import chain


if __name__ == '__main__':
    DEVICE = 'cuda:0'
    NUM_EPOCHS = 3
    IMAGE_SIZE = 224
    BATCH_SIZE = 64
    LERANING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    DATA_PATH = '/home/mohosb/seagate_exp/nyu_depth_v2'

    DEVICE = torch.device(DEVICE)

    teacher_model = DepthAnything('large').eval().to(DEVICE)
    student_model = DepthAnything('small').train().to(DEVICE)
    student_model.core.backbone.requires_grad_(False)  # Freeze DINOv2 encoder parameters

    train_criterion = WeightedComposedLoss((
        (0.5, ScaleInvariantMSELoss(use_log=False, reduction='none')),
        (0.5, GradientMatchingLoss(scales=(1., 0.5, 0.25), reduction='none')),
    )).to(DEVICE)
    test_criterion = ScaleInvariantMSELoss(use_log=False, reduction='none').to(DEVICE)
    opt = torch.optim.Adam(
        chain(student_model.core.neck.parameters(), student_model.core.head.parameters()), 
        lr=LERANING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    transform = tvv2.Compose([
        tvv2.Resize(IMAGE_SIZE),
        tvv2.CenterCrop(IMAGE_SIZE),
        np.array,
    ])
    train_dataset, test_dataset = random_split(
        RawNYUDepthV2(DATA_PATH, '*room*', transform),
        (0.9, 0.1),
        torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, multiprocessing_context='spawn', persistent_workers=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, multiprocessing_context='spawn', persistent_workers=True, num_workers=4)

    with torch.no_grad():
        print('Mean Testing Loss:', evaluate(teacher_model, student_model, test_criterion, test_loader))

        for epoch_idx in range(NUM_EPOCHS):
            print(f'Epoch {epoch_idx + 1}/{NUM_EPOCHS}')
            progress = tqdm(train_loader, unit='batch')
            losses = deque(maxlen=40)
            student_model.train()
            for x in progress:
                x = list(x.numpy())

                y = teacher_model(x)

                with torch.enable_grad():
                    z = student_model(x)

                    y = y.to(z.device, z.dtype)
                    loss = train_criterion(z, y)
                    opt.zero_grad()
                    loss.mean().backward()
                    opt.step()

                losses.append(loss.cpu())
                mean_train_loss = torch.cat(tuple(losses)).mean().item()
                progress.set_postfix({'mean_loss': mean_train_loss})

            mean_test_loss = evaluate(teacher_model, student_model, test_criterion, test_loader)
            print(f'Mean Training Loss: {mean_train_loss}, Mean Testing Loss: {mean_test_loss}')

            torch.save(student_model.state_dict(), f'training_{epoch_idx + 1}.pt')

