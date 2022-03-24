import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from model.model import Unet
from dataset import dataset
from torch.utils.data import DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from cal_eval import *
import logging
from DICE_LOSS import Dice

logging.basicConfig(
    filename=r'logging_txt.txt',
    level=logging.INFO
)

parse = argparse.ArgumentParser(description=f'para of net')
parse.add_argument('--train_path', type=str, default=r'/media/vge/DataA/yhj/data/WHU/train/image')
parse.add_argument('--val_path', type=str, default=r'/media/vge/DataA/yhj/data/WHU/val/image')
parse.add_argument('--in_channels', type=int, default=3)
parse.add_argument('--n_classes', type=int, default=1)
parse.add_argument('--epochs', type=int, default=100)
parse.add_argument('--batch_size', type=int, default=8)
parse.add_argument('--lr', type=int, default=0.01)
parse.add_argument('--tensorboard_path', type=str, default=r'tensorboard')
parse.add_argument('--eval_step', type=int, default=5)
args = parse.parse_args()


def train(net: nn.Module, device: torch.device):
    train_dataset = dataset(data_path=args.train_path)
    train_datalolader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True)  # 批处理大小 shuffle指是否打乱顺序
    val_dataset = dataset(data_path=args.val_path)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    # criterion = nn.BCEWithLogitsLoss()  # loss函数
    criterion = Dice()  # loss函数
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=1e-04,
                          momentum=0.9)  # lr:学习率 超参数weight+decay:正则化 超参数momentum:动量(加速收敛)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.35, patience=2, verbose=True,
                                                     threshold_mode='rel', threshold=1e-08, min_lr=1e-05,
                                                     eps=1e-08)  # 学习率调整器
    train_writer = SummaryWriter(log_dir=args.tensorboard_path, flush_secs=1)
    check_point = 0
    eval_iou_check = 0
    for epoch in range(args.epochs):
        with tqdm(total=len(train_dataset)//args.batch_size, desc=f'进度:{epoch + 1}/{args.epochs}') as pbar:
            net.train()  # 训练模式
            for image, label in train_datalolader:
                optimizer.zero_grad()  # 梯度清零，不清零会导致梯度累加
                image = image.to(device=device, dtype=torch.float)
                label = label.to(device=device, dtype=torch.float)
                pred = net(image)  # 网络计算，预测结果
                loss = criterion(pred, label)  # 计算loss
                loss.backward()  # 梯度反算
                optimizer.step()  # 优化器进行网络参数反向传播
                train_iou, train_f1, train_para = Cal_iou_para(pred, label)  # 计算该批精度指标
                check_point += 1
                pbar.set_postfix(**{'loss': loss.item(), 'iou': train_iou.item()})
                pbar.update(1)
                train_writer.add_scalar('loss', loss.item(), check_point)
                train_writer.add_scalar('iou', train_iou.item(), check_point)
                train_writer.add_scalar('f1', train_f1.item(), check_point)
                if check_point % 200 == 0:
                    logging.info(
                        f'check({check_point}): loss:{loss.item()} iou:{train_iou.item()} f1:{train_f1.item()}')
                    logging.info(f'- - - - - - - - - - - - - - - - - - - - - - - - - \n')
        torch.save(net.state_dict(), 'model2.pth')  # 保存网络
        pbar.close()
        optimizer.zero_grad()
        if epoch % args.eval_step:
            net.eval()  # 测试模式
            tp, tn, fp, fn = 0, 0, 0, 0
            with tqdm(total=len(val_dataset), desc=f'验证中:') as pbar:
                with torch.no_grad():  # 该区域内不跟踪计算梯度
                    for image, label in val_dataloader:
                        image = image.to(device=device, dtype=torch.float)
                        label = label.to(device=device, dtype=torch.float)
                        pred = net(image)
                        loss = criterion(pred, label)
                        eval_iou, eval_f1, eval_para = Cal_iou_para(pred, label)
                        pbar.set_postfix(**{'loss': loss.item(), 'iou': eval_iou.item()})
                        tp += eval_para[0]
                        tn += eval_para[1]
                        fp += eval_para[2]
                        fn += eval_para[3]
                        pbar.update(1)
            pbar.close()
            eval_iou = tp / (tp + fp + fn + 1e-08)
            eval_precise = tp / (tp + fp + 1e-08)
            eval_recall = tp / (tp + fn + 1e-08)
            eval_f1 = 2 * eval_precise * eval_recall / (eval_precise + eval_recall + 1e-08)
            scheduler.step(eval_iou)
            if eval_iou_check < eval_iou:
                eval_iou_check = eval_iou
                torch.save(net.state_dict(), 'model1.pth')
            logging.info(f'eval({epoch // args.eval_step + 1}) iou:{eval_iou} f1:{eval_f1}')
            logging.info(f'- - - - - - - - - - - - - - - - - - - - - - - - - \n')


if __name__ == '__main__':
    net = Unet(in_channels=args.in_channels, n_classes=args.n_classes,phi=1)
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    net = net.to(device=device)
    train(net, device)
