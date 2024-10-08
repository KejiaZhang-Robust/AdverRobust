import torch.backends.cudnn as cudnn
import datetime
from easydict import EasyDict
import yaml
import logging
from models import *
from utils_train import *
from utils import *
from utils_AWP import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('configs_train.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = WRN34_10()
proxy = WRN34_10()

net.num_classes = config.DATA.num_class
net.norm = True
net.mean = torch.tensor(config.DATA.mean).to(device)
net.std = torch.tensor(config.DATA.std).to(device)

proxy.num_classes = config.DATA.num_class
proxy.norm = True
proxy.mean = torch.tensor(config.DATA.mean).to(device)
proxy.std = torch.tensor(config.DATA.std).to(device)

file_name = config.Operation.Prefix
data_set = config.Train.Data
learning_rate = config.Train.Lr
check_path = os.path.join('./checkpoint', data_set, file_name)

if not os.path.isdir(os.path.join('./checkpoint', data_set)):
    os.mkdir(os.path.join('./checkpoint', data_set))
if not os.path.isdir(check_path):
    os.mkdir(check_path)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(check_path, file_name + '_record.log')),
        logging.StreamHandler()
    ])

train_loader, test_loader = create_dataloader(data_set, Norm=False)

net = torch.nn.DataParallel(net).to(device)
proxy = torch.nn.DataParallel(proxy).to(device)# parallel GPU
cudnn.benchmark = True

if config.Operation.Resume == True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
else:
    start_epoch = 0
    best_prec1 = 0
    logger.info(config.Operation.record_words)
    logger.info('%-5s\t%-10s\t%-9s\t%-9s\t%-8s\t%-15s \t %-7s \t %-14s',
                'Epoch','Train Loss','Train Acc','Test Loss','Test Acc','Test Robust Acc','Val Acc','Val Robust Acc')

proxy_optim = optim.SGD(proxy.parameters(), lr=0.01)
awp_adversary = AdvWeightPerturb(model=net, proxy=proxy, proxy_optim=proxy_optim, gamma=0.01)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

for epoch in range(start_epoch+1, config.Train.Epoch+1):
    learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch, config.Train.lr_change_iter[0], config.Train.lr_change_iter[1])
    #TODO Train
    acc_train, train_loss = train_AWP(net, train_loader, optimizer, epoch, awp_adversary, config)
    #TODO Test
    acc_test, pgd_acc, loss_test, best_prec1 = test_net_robust(net, test_loader, epoch, optimizer, best_prec1, config, save_path=check_path)
    logger.info('%-5d\t%-10.2f\t%-9.2f\t%-9.2f\t%-8.2f\t%.2f', epoch, train_loss, acc_train, loss_test, acc_test, pgd_acc) 