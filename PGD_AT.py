import torch.backends.cudnn as cudnn
import datetime
from easydict import EasyDict
import yaml
import logging
from models import *
from utils_train import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('configs_train.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = WRN34_10(Num_class=config.DATA.num_class)

net.Norm = True
net.norm_mean = torch.tensor(config.DATA.mean).to(device)
net.norm_std = torch.tensor(config.DATA.std).to(device)
file_name = config.Operation.Prefix
data_set = config.Train.Data
check_path = os.path.join('./checkpoint', data_set, file_name)
if not os.path.isdir(check_path):
    os.mkdir(check_path)
learning_rate = config.Train.Lr

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(check_path, file_name + '_record.log')),
        logging.StreamHandler()
    ])

train_loader, test_loader, val_loader = create_loader_with_val_CIFAR_10()
net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
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

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
best_val_robust_acc = 0
for epoch in range(start_epoch+1, config.Train.Epoch+1):
    learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch)
    #TODO Train
    if config.Train.Train_Method == 'AT':
        acc_train, train_loss = train_adversarial(net, epoch, train_loader, optimizer, config)
    elif config.Train.Train_Method == 'TRADES':
        acc_train, train_loss = train_adversarial_TRADES(net, epoch, train_loader, optimizer, config)
    else:
        acc_train, train_loss = train(net, epoch, train_loader, optimizer, config)
    #TODO Test
    test_acc, adv_acc, loss_test = test_net(net, test_loader, config)
    #TODO Val
    val_test_acc, val_adv_acc, best_val_robust_acc = val_net(net, epoch, val_loader, optimizer, best_val_robust_acc, config, check_path)
    logger.info('%-5d\t%-10.2f\t%-9.2f\t%-9.2f\t%-8.2f\t%-15.2f \t %-7.2f \t %-14.2f',
                epoch, train_loss, acc_train, loss_test,
                test_acc, adv_acc, val_test_acc, val_adv_acc)