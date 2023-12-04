import torch.backends.cudnn as cudnn
from easydict import EasyDict
import yaml
import logging
from models import *
from utils_train import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('configs_train.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

# net = WRN34_10(Num_class=config.DATA.num_class)
net = ResNet18_F(Num_class=config.DATA.num_class)

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

net.Num_class = config.DATA.num_class
norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
if config.Train.Train_Method == 'AT' or config.Train.Train_Method == 'Constrain_AT':
    net.Norm = True
    net.norm_mean = norm_mean
    net.norm_std = norm_std
    Data_norm = False
    logger.info('Adversarial Training || net: '+config.Operation.Prefix + ' || '+config.Train.Train_Method)
else:
    net.Norm = False
    Data_norm = True
    logger.info('Natural Training || net: '+config.Operation.Prefix)

train_loader, test_loader = create_dataloader(data_set, Norm=Data_norm)

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
    logger.info('%-5s\t%-10s\t%-9s\t%-9s\t%-8s\t%-15s', 'Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Test Robust Acc')


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
for epoch in range(start_epoch + 1, config.Train.Epoch + 1):
    learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch)
    if config.Train.Train_Method == 'AT':
        acc_train, train_loss = train_adversarial(net, epoch, train_loader, optimizer, config)
    elif config.Train.Train_Method == 'TRADES':
        acc_train, train_loss = train_adversarial_TRADES(net, epoch, train_loader, optimizer, config)
    else:
        acc_train, train_loss = train(net, epoch, train_loader, optimizer, config)
    # acc_test, pgd_acc, loss_test, best_prec1 = test_net_normal(net, test_loader, epoch, optimizer, best_prec1, config, save_path=check_path)
    acc_test, pgd_acc, loss_test, best_prec1 = test_net_robust(net, test_loader, epoch, optimizer, best_prec1, config, save_path=check_path)
    logger.info('%-5d\t%-10.2f\t%-9.2f\t%-9.2f\t%-8.2f\t%.2f', epoch, train_loss, acc_train, loss_test,
                acc_test, pgd_acc) 