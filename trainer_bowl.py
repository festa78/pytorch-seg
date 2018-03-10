from __future__ import division

import json
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
import tqdm

# from resnet import FCN
from upsample_softmax import FCN
# from duc import FCN
# from gcn import FCN
from datasets_bowl import BowlDataSet
from loss import CrossEntropy2d, CrossEntropyLoss2d
from visualize import LinePlotter
from transform import ReLabel, ToLabel, ToSP, Scale

# NUM_CLASSES.
NUM_CLASSES = 2

normalize_info = json.load(open('../../input/drn/v1/info.json', 'r'))
mean = [m / 255. for m in normalize_info['mean']]
std = [m / 255. for m in normalize_info['std']]

input_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor(),
    Normalize(mean, std),
])

target_transform = Compose([
    Scale((256, 256), Image.NEAREST),
    ToSP(256),
    ToLabel(),
    # ReLabel(255, 21),
])

trainloader = data.DataLoader(BowlDataSet("../../input/drn/v1/", img_transform=input_transform,
                                         label_transform=target_transform),
                              batch_size=16, shuffle=True, pin_memory=True)

if torch.cuda.is_available():
    model = torch.nn.DataParallel(FCN(NUM_CLASSES))
    model.cuda()

epoches = 80
lr = 1e-4
weight_decay = 2e-5
momentum = 0.9
weight = torch.ones(NUM_CLASSES)
# weight[21] = 0
max_iters = 92*epoches

criterion = CrossEntropyLoss2d(weight.cuda())
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
ploter = LinePlotter()

model.train()
for epoch in range(epoches):
    running_loss = 0.0
    for i, (images, labels_group) in tqdm.tqdm(enumerate(trainloader), total=int(len(trainloader.dataset) / trainloader.batch_size)):
        if torch.cuda.is_available():
            images = [Variable(image.cuda()) for image in images]
            labels_group = [labels for labels in labels_group]
        else:
            images = [Variable(image) for image in images]
            labels_group = [labels for labels in labels_group]

        optimizer.zero_grad()
        losses = []
        for img, labels in zip(images, labels_group):
            outputs = model(img)
            labels = [Variable(label.cuda()) for label in labels]
            for pair in zip(outputs, labels):
                losses.append(criterion(pair[0], pair[1]))

        if epoch < 40:
            loss_weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        else:
            loss_weight = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

        loss = 0
        for w, l in zip(loss_weight, losses):
            loss += w*l

        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]

        # lr = lr * (1-(92*epoch+i)/max_iters)**0.9
        # for parameters in optimizer.param_groups:
        #     parameters['lr'] = lr

    print("Epoch [%d] Loss: %.4f" % (epoch+1, running_loss/i))
    ploter.plot("loss", "train", epoch+1, running_loss/i)
    running_loss = 0

    if (epoch+1) % 20 == 0:
        lr /= 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        torch.save(model.state_dict(), "./pth/fcn-deconv-%d.pth" % (epoch+1))


torch.save(model.state_dict(), "./pth/fcn-deconv.pth")
