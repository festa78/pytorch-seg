import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop, Normalize
from transform import Colorize
from transform import Scale
# from resnet import FCN
from upsample import FCN
# from duc import FCN
# from gcn import FCN
from datasets_bowl import BowlTestSet

# Constants
NUM_CLASSES = 2

normalize_info = json.load(open('../../input/drn/v1/info.json', 'r'))
mean = [m / 255. for m in normalize_info['mean']]
std = [m / 255. for m in normalize_info['std']]

input_transform = Compose([
    Scale((256, 256), Image.BILINEAR),
    ToTensor(),
    Normalize(mean, std)
])

batch_size = 1
dst = BowlTestSet("../../input/drn/v1/", transform=input_transform)

testloader = data.DataLoader(dst, batch_size=batch_size,
                             num_workers=8)

model = torch.nn.DataParallel(FCN(NUM_CLASSES))
model.cuda()
model.load_state_dict(torch.load("./pth/fcn-deconv-60.pth"))
model.eval()

for index, (imgs, name, size) in tqdm(enumerate(testloader)):
    imgs = Variable(imgs.cuda())
    outputs = model(imgs)

    output = outputs[0][0].data.max(0)[1]
    # output = Colorize()(output)
    # print(output)
    output = np.transpose(output.cpu().numpy())
    img = Image.fromarray(output.astype(np.uint8))
    # img = Image.fromarray(output, "RGB")
    # img = Image.fromarray(output[0].cpu().numpy(), "P")
    img = img.resize((size[0].numpy(), size[1].numpy()), Image.NEAREST)
    img.save("./results/%s" % name)
