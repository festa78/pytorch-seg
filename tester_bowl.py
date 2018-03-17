import json
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import pandas as pd
from skimage.io import imsave
from skimage.morphology import label
from skimage.transform import resize
import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop, Normalize
from transform import Colorize
from transform import Scale
# from resnet import FCN
from upsample_softmax import FCN
# from duc import FCN
# from gcn import FCN
from datasets_bowl import BowlTestSet

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# Constants
NUM_CLASSES = 2
TEST_PATH = '../../input/stage1_test'
BATCH_SIZE = 1

normalize_info = json.load(open('../../input/drn/v1/info.json', 'r'))
mean = [m / 255. for m in normalize_info['mean']]
std = [m / 255. for m in normalize_info['std']]

input_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor(),
    Normalize(mean, std)
])

dst = BowlTestSet(TEST_PATH, transform=input_transform)

testloader = data.DataLoader(dst, batch_size=BATCH_SIZE,
                             num_workers=8)

model = torch.nn.DataParallel(FCN(NUM_CLASSES))
model.cuda()
model.load_state_dict(torch.load("./pth/fcn-deconv.pth"))
model.eval()

new_test_ids = []
rles = []
for index, (imgs, name, size) in tqdm(enumerate(testloader)):
    imgs = Variable(imgs.cuda())
    outputs = model(imgs)

    output = outputs[0][0].data.max(0)[1]
    output = output.cpu().numpy().astype(np.uint8)
    imsave("./results/%s" % name, output)
    pred_test_upsampled = resize(output,
                                 (size[0].numpy()[0], size[1].numpy()[0]),
                                 mode='constant', preserve_range=True)
    rle = list(prob_to_rles(pred_test_upsampled))
    rles.extend(rle)
    idx = os.path.splitext(name[0])[0]
    new_test_ids.extend([idx] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('20180225-sub-dsbowl2018-pytorchseg.csv', index=False)
