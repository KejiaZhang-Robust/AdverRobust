import torch.backends.cudnn as cudnn
from models import *
from SSA_attack import *
from easydict import EasyDict
import yaml
import logging
import os

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# net = ResNet18()

with open('configs_test.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = WRN34_10_F(Num_class=config.DATA.num_class)

file_name = config.Operation.Prefix
data_set = config.DATA.Data
check_path = os.path.join('./checkpoint', data_set, file_name)
if not os.path.isdir(check_path):
    os.mkdir(check_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(check_path, file_name + '_test.log')),
        logging.StreamHandler()
    ])

net.Num_class = config.DATA.num_class
norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
if config.Operation.Method == 'AT':
    net.Norm = True
    net.norm_mean = norm_mean
    net.norm_std = norm_std
    Data_norm = False
    logger.info("Adversarial Training Model Robustness")
else: 
    net.Norm = False
    Data_norm = True
    logger.info("Natural Training Model Robustness")

_, test_loader = create_dataloader(data_set, Norm=Data_norm)

net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
cudnn.benchmark = True
net.eval()

#TODO: PGD Attack test
print("==> Loading best model:"+file_name+"\n")
assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))

net.load_state_dict(checkpoint_best['state_dict'])
pgd_acc = evaluate_SSA(net, test_loader, eps=16.0)
logger.info(f"SSA_attack:[nb_iter:{16.0}]->pgd_acc: {pgd_acc: .2f}")