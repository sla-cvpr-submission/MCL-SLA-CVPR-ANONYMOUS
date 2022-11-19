import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm

from loaders.domainnet import build_dataset
from loaders.office_home import build_dataset_officehome
from loaders.visda import build_dataset_visda
from utils.utils import set_seed, weights_init, print_options, AllMeters
from utils.lr_schedule import InvLr
from utils.ema import ModelEMA
from model.basenet import Predictor_deep
from model.resnet import resnet34
from model.ppc import ProtoClassifier
from utils.losses import Prototype, loss_unl

from mdh import ModelHandler, GlobalHandler


def get_args():
    parser = argparse.ArgumentParser(description='Multi-level consistency learning')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--bs', default=24, type=int)
    parser.add_argument('--bs_unl_multi', default=2, type=int)
    parser.add_argument('--n_workers', default=4, type=int)
    parser.add_argument('--source', type=str, default='painting')
    parser.add_argument('--target', type=str, default='real')
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--arch', type=str, default='resnet',
                        help='which network to use')
    parser.add_argument('--data_root', type=str, default='/home/xxx/SLC/office_home_data/')
    parser.add_argument('--unl_transform', type=str, default='fixmatch')
    parser.add_argument('--labeled_transform', type=str, default='labeled')
    parser.add_argument('--num', type=int, default=3,
                        help='number of labeled examples in the target')
    parser.add_argument('--num_classes', type=int, default=65)
    parser.add_argument('--dataset', type=str, default='office_home',
                        choices=['multi', 'office', 'office_home', 'visda'],
                        help='the name of dataset')
    parser.add_argument('--base_path', type=str, default='./data/txt/office_home/')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--T', type=float, default=0.05, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--T2', type=float, default=1.25, metavar='T',
                        help='temperature (default: 0.05)')
    parser.add_argument('--lambda_u', type=float, default=1)
    parser.add_argument('--lambda_cls', type=float, default=0.2)
    parser.add_argument('--lambda_ot', type=float, default=1)
    parser.add_argument('--log_dir', type=str, default='./logs/domainnet/contras_cls/debug')
    parser.add_argument('--uda', action='store_true', default=False)
    parser.add_argument('--test_interval', type=float, default=500)
    parser.add_argument('--print_interval', type=float, default=100)
    parser.add_argument('--num_steps', type=int, default=13000)
    parser.add_argument('--warm_steps', type=int, default=250)
    parser.add_argument('--method', type=str, default='base')
    parser.add_argument('--order', type=int, default=0)
    parser.add_argument('--note', type=str, default='')

    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--update_interval', type=int, default=500)
    parser.add_argument('--ppc_T', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=0.3)

    args = parser.parse_args()
    return args


# Training
def train(args, source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, net_F, optimizer, scheduler):
    net_G.train()
    net_F.train()
    if 'LC' in args.method:
        ppc = ProtoClassifier(65)

    loss_name_list = ['Lx', 'L_fix', 'mask_prop', 'L_con_cls']

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    target_unl_iter = iter(target_loader_unl)

    criterion = nn.CrossEntropyLoss().cuda()
    best_acc_val, cur_acc_test = 0, 0

    net_G_ema = ModelEMA(net_G, decay=0.99)
    net_F_ema = ModelEMA(net_F, decay=0.99)

    proto_s = Prototype(C=args.num_classes, dim=args.inc)

    writer = SummaryWriter(args.mdh.getLogPath())
    writer.add_text('Hash', args.mdh.getHashStr())
    lc_criterion = nn.CrossEntropyLoss(reduction='none')
    for batch_idx in range(args.num_steps):
        lambda_warm = 1 if batch_idx > args.warm_steps else 0
        try:
            data_batch_source = source_iter.next()
        except:
            source_iter = iter(source_loader)
            data_batch_source = source_iter.next()

        try:
            data_batch_target = target_iter.next()
        except:
            target_loader.dataset.shuffle_repeat()
            target_iter = iter(target_loader)
            data_batch_target = target_iter.next()

        try:
            data_batch_unl = target_unl_iter.next()
        except:
            target_unl_iter = iter(target_loader_unl)
            data_batch_unl = target_unl_iter.next()

        imgs_s_w = data_batch_source[0].cuda()
        gt_s = data_batch_source[1].cuda()

        imgs_t_w = data_batch_target[0].cuda()
        gt_t = data_batch_target[1].cuda()

        imgs_tu_w, imgs_tu_s = data_batch_unl[0][0].cuda(), data_batch_unl[0][1].cuda()
        gt_tu = data_batch_unl[1].cuda()

        data = torch.cat((imgs_s_w, imgs_t_w), 0)
        target = torch.cat((gt_s, gt_t), 0)
        output = net_G(data)

        feat_t_con, out1 = net_F(output)
        feat_s, feat_t = feat_t_con.chunk(2)
        out_s, out_t = out1.chunk(2)
        if batch_idx > args.warmup and 'LC' in args.method:
            sy2 = ppc(feat_s.detach(), args.ppc_T)

            log_softmax_out = F.log_softmax(out_s, dim=1)
            l_loss = lc_criterion(out_s, gt_s)
            soft_loss = -(sy2 * log_softmax_out).sum(axis=1)
            sloss = ((1 - args.alpha) * l_loss + args.alpha * soft_loss).mean()
            tloss = criterion(out_t, gt_t)
            Lx = 0.5 * sloss + 0.5 * tloss
            # Lx = criterion(out1, target)
        else:
            Lx = criterion(out1, target)

        L_ot, L_con_cls, L_fix, consis_mask = loss_unl(net_G, net_F, imgs_tu_w, imgs_tu_s, proto_s, args)

        # backward
        Loss = Lx + L_fix * args.lambda_u + L_con_cls * args.lambda_cls + lambda_warm * L_ot * args.lambda_ot
        Loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(batch_idx if batch_idx < args.warmup else batch_idx - args.warmup)
        net_G_ema.update(net_G)
        net_F_ema.update(net_F)

        if 'LC' in args.method and args.warmup > 0 and batch_idx == args.warmup-1:
            ppc.init(net_G_ema.ema, net_F_ema.ema, target_loader_unl)

        if batch_idx >= args.warmup and args.update_interval > 0 and batch_idx % args.update_interval == 0 and 'LC' in args.method:
            ppc.init(net_G_ema.ema, net_F_ema.ema, target_loader_unl)

        if batch_idx % args.print_interval == 0:
            writer.add_scalar('Loss/Lx', Lx.item(), batch_idx)
            writer.add_scalar('Loss/L_fix', L_fix.item(), batch_idx)
            writer.add_scalar('Loss/L_con_cls', L_con_cls.item(), batch_idx)
            writer.add_scalar('Loss/L_ot', L_ot.item(), batch_idx)
            writer.add_scalar('Loss/total', Loss.item(), batch_idx)

        proto_s.update(feat_s, gt_s, batch_idx, norm=True)

        loss_name_list = ['Lx', 'L_fix', 'mask_prop', 'L_con_cls']
        loss_value_list = [Lx.item(), L_fix.item(), consis_mask.sum().item() / consis_mask.shape[0],
                           L_con_cls.item()]
        my_best_val, my_best_acc, my_best_ema_acc, counter = 0, 0, 0, 0
        if (batch_idx + 1) % args.test_interval == 0:
            writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], batch_idx)

            acc_val = test(target_loader_val, net_G_ema.ema, net_F_ema.ema, batch_idx)
            acc1, acc2,  per_acc1, per_acc2 = test_multi(target_loader_test,
                                                         net_G, net_F,
                                                         net_G_ema.ema, net_F_ema.ema)
            writer.add_scalar('Val/acc', acc_val, batch_idx)                                       
            writer.add_scalar('Test/acc', acc1, batch_idx)
            writer.add_scalar('Test/acc_ema', acc2, batch_idx)
            writer.add_scalar('Test/mAcc', per_acc1, batch_idx)
            writer.add_scalar('Test/mAcc_ema', per_acc2, batch_idx)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                cur_acc_test = acc1
                cur_acc_test_ema = acc2
            writer.add_scalar('Test/BestAcc', cur_acc_test, batch_idx)
            writer.add_scalar('Test/BestAcc_ema', cur_acc_test_ema, batch_idx)


            if batch_idx >= 4999:
                if acc_val > my_best_val:
                    my_best_val = acc_val
                    my_best_acc = acc1
                    my_best_ema_acc = acc2
                    counter = 0
                else:
                    counter += 1
                if counter > 5 or batch_idx == args.num_steps-1:
                    writer.add_scalar('Final/acc', my_best_acc, 0)
                    writer.add_scalar('Final/ema_acc', my_best_ema_acc, 0)


@torch.no_grad()
def test(test_loader, net_G, net_F, iter_idx):
    net_G.eval()
    net_F.eval()

    correct = 0
    total = 0
    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        _, outputs = net_F(net_G(inputs))

        outputs = torch.softmax(outputs, dim=1)
        max_prob, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    net_G.train()
    net_F.train()
    acc = correct / total * 100.
    return acc


def test_multi(test_loader, net_G1, net_F1, net_G2, net_F2):
    net_G1.eval()
    net_F1.eval()
    net_G2.eval()
    net_F2.eval()

    correct1 = 0
    correct2 = 0
    total = 0

    predicted_list1 = []
    predicted_list2 = []
    predicted_list = []
    target_list = []

    for batch_idx, data_batch in enumerate(tqdm(test_loader)):
        try:
            inputs, targets = data_batch[0].cuda(), data_batch[1].cuda()
        except:
            inputs, targets = data_batch[0][0].cuda(), data_batch[1].cuda()
        _, outputs1 = net_F1(net_G1(inputs))
        _, outputs2 = net_F2(net_G2(inputs))

        l = 0.5
        outputs1 = torch.softmax(outputs1, dim=1)
        outputs2 = torch.softmax(outputs2, dim=1)
        max_prob, predicted1 = outputs1.max(1)
        max_prob, predicted2 = outputs2.max(1)
        total += targets.size(0)

        correct1 += predicted1.eq(targets).sum().item()
        correct2 += predicted2.eq(targets).sum().item()

        predicted_list1 += predicted1.cpu().numpy().tolist()
        predicted_list2 += predicted2.cpu().numpy().tolist()
        target_list += targets.cpu().numpy().tolist()

    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(target_list, predicted_list1, normalize='true')
    cm2 = confusion_matrix(target_list, predicted_list2, normalize='true')

    per_acc1 = cm1.diagonal().mean() * 100.
    per_acc2 = cm2.diagonal().mean() * 100.

    net_G1.train()
    net_F1.train()
    net_G2.train()
    net_F2.train()
    acc1 = correct1 / total * 100.
    acc2 = correct2 / total * 100.
    print('acc1 %.2f acc2 %.2f ' % (acc1, acc2))
    print('mAcc1 %.2f mAcc2 %.2f ' % (per_acc1, per_acc2))

    return acc1, acc2,  per_acc1, per_acc2


def get_optim_params(model, lr):
    lr_list, lr_multi_list = [], []
    for name, param in model.named_parameters():
        if 'fc' in name:
            lr_multi_list.append(param)
        else:
            lr_list.append(param)
    return [{'params': lr_list, 'lr': lr},
            {'params': lr_multi_list, 'lr': 10 * lr}]


def main(args):

    print_options(args)

    if args.dataset == 'multi':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset(args)
    elif args.dataset == 'office_home':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_officehome(args)
    elif args.dataset == 'visda':
        source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test, class_list = build_dataset_visda(args)

    set_seed(args.seed)

    net_G = resnet34().cuda()
    net_F = Predictor_deep(num_class=args.num_classes, inc=512, temp=args.T)
    weights_init(net_F)
    net_F.cuda()
    args.inc = 512

    optimizer = optim.SGD(get_optim_params(net_G, args.lr) + net_F.get_optim_params(args.lr),
                          momentum=0.9, weight_decay=5e-4, nesterov=True)

    scheduler = InvLr(optimizer)

    train(args, source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test,
          net_G, net_F, optimizer, scheduler)


if __name__ == '__main__':
    gh = GlobalHandler('MCL')
    args = get_args()
    args.mdh = ModelHandler(args, keys=['source', 'target', 'seed', 'num', 'T2', 'lambda_u', 'lambda_cls', 'num_steps', 'warm_steps', 'method', 'order', 'note', 'ppc_T', 'alpha', 'update_interval', 'warmup'], gh=gh)
    main(args)
