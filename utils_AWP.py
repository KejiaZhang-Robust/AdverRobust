import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from tqdm import tqdm
from torch.autograd import Variable
from utils_train import pgd_attack,LabelSmoothLoss,_label_smoothing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])

class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


def perturb_input_KL_loss(model,
                  x_natural,
                  step_size=2./255,
                  epsilon=8./255,
                  perturb_steps=10):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                F.softmax(model(x_natural), dim=1),
                                reduction='sum')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def train_AWP(model, train_loader, optimizer, epoch, awp_adversary, config):
    train_loss = 0
    correct = 0
    total = 0
    print('\n[ Epoch: %d ]' % epoch)
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)

        # craft adversarial examples
        x_adv = pgd_attack(model, x_natural, target, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= 10:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                             targets=target)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(target, config.DATA.num_class, config.Train.Factor)).to(device))
            loss = LabelSmoothLoss(logits_adv, label_smoothing.float())
        else:
            loss = criterion(logits_adv, target)

        train_loss += loss.item()
        _, predicted = logits_adv.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 10:
            awp_adversary.restore(awp)
        
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2))
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss