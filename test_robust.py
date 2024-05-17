import torch.backends.cudnn as cudnn
from models import *
from utils_test import evaluate_normal, evaluate_pgd, evaluate_autoattack, evaluate_cw
from easydict import EasyDict
import yaml
import logging
import os

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('configs_test.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = ResNet18(Num_class=config.DATA.num_class)

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
        logging.FileHandler(os.path.join(check_path, file_name + config.Operation.Addtional_string +'_test.log')),
        logging.StreamHandler()
    ])
logger.info(config.Operation.Record_string)

norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
net.num_classes = config.DATA.num_class
net.norm = True
net.mean = norm_mean
net.std = norm_std
Data_norm = False
logger.info("Training Model:"+config.Operation.Method+" Robustness")

_, test_loader = create_dataloader(data_set, Norm=Data_norm)

net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
cudnn.benchmark = True
net.eval()

#TODO: PGD Attack test
print("==> Loading best model:"+file_name+"\n")
assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))
checkpoint_last = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))

# ['apgd-ce', 'apgd-t', 'fab-t', 'square']
auto_attacks_methods = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
if config.Operation.Validate_Best == True:
    logger.info("=======Best_trained_model Performance=======")
    net.load_state_dict(checkpoint_best['state_dict'])
    if config.Operation.Validate_Natural:
        ##----->Clean
        clean_acc = evaluate_normal(net, test_loader)
        logger.info(f"Normal Acc: {clean_acc:.2f}")
    if config.Operation.Validate_PGD:
        ##----->FGSM
        fgsm_acc = evaluate_pgd(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 1)
        logger.info(f"PGD_attack:[nb_iter:1,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->pgd_acc: {fgsm_acc: .2f}")
        ##----->PDG
        for pgd_param in config.ADV.pgd_test:
            pgd_acc = evaluate_pgd(net, test_loader, pgd_param[1], pgd_param[2], pgd_param[0])
            logger.info(f"PGD_attack:[nb_iter:{pgd_param[0]},eps:{pgd_param[1]},step_size:{pgd_param[2]}]->pgd_acc: {pgd_acc: .2f}")
    if config.Operation.Validate_CW:
        cw_acc = evaluate_cw(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 20)
        logger.info(f"CW_attack:[nb_iter:20,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->CW_acc: {cw_acc: .2f}")
    if config.Operation.Validate_Autoattack:
        ##----->Autoattack
        auto_acc = evaluate_autoattack(net, test_loader, config.ADV.clip_eps, auto_attacks_methods)
        logger.info(f"Auto_attack:[eps:{config.ADV.clip_eps}]->AA_acc: {auto_acc: .2f}")


if config.Operation.Validate_Last == True:
    print("==> Loading last model:"+file_name+"\n")
    logger.info("=======Last_trained_model Performance=======")
    net.load_state_dict(checkpoint_last['state_dict'])
    if config.Operation.Validate_Natural:
        ##----->Clean
        clean_acc = evaluate_normal(net, test_loader)
        logger.info(f"Normal Acc: {clean_acc:.2f}")
    if config.Operation.Validate_PGD:
        ##----->FGSM
        fgsm_acc = evaluate_pgd(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 1)
        logger.info(f"PGD_attack:[nb_iter:1,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->pgd_acc: {fgsm_acc: .2f}")
        ##----->PDG
        for pgd_param in config.ADV.pgd_test:
            pgd_acc = evaluate_pgd(net, test_loader, pgd_param[1], pgd_param[2], pgd_param[0])
            logger.info(f"PGD_attack:[nb_iter:{pgd_param[0]},eps:{pgd_param[1]},step_size:{pgd_param[2]}]->pgd_acc: {pgd_acc: .2f}")
    if config.Operation.Validate_CW:
        cw_acc = evaluate_cw(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 20)
        logger.info(f"CW_attack:[nb_iter:20,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->CW_acc: {cw_acc: .2f}")
    if config.Operation.Validate_Autoattack:
        ##----->Autoattack
        auto_acc = evaluate_autoattack(net, test_loader, config.ADV.clip_eps, auto_attacks_methods)
        logger.info(f"Auto_attack:[eps:{config.ADV.clip_eps}]->AA_acc: {auto_acc: .2f}")