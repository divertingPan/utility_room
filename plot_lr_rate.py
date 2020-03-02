# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torchvision.models as models


model = models.resnext101_32x8d(pretrained=True)
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.06)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])

plt.plot(x, y)
