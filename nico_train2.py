import argparse

import torch
import os
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.cutmix import cutmix
from data.data_helper import available_datasets
from models import model_factory

from utils.Logger import Logger
from utils.util import *

from models.augnet import AugNet
from models.caffenet import caffenet
from models.resnet import resnet18
from utils.contrastive_loss import SupConLoss

from torchvision import transforms
"""
#!/usr/bin/env bash
max=1
for i in $(seq 1 $max); do
  python train.py \
    --task='PACS' \
    --seed=1 \
    --alpha1=1\
    --alpha2=1\
    --beta=0.1\
    --lr_sc=0.005
done
Footer
"""
def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--visualization", default=False, type=bool)
    parser.add_argument("--epochs_min", type=int, default=1,
                        help="")
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--ckpt1", default="logs/model", type=str)
    #
    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr_sc", default=10, type=float)
    parser.add_argument("--task", default='PACS', type=str)

    return parser.parse_args()


class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature=4, alpha=1, beta=1, reduction="batchmean", **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = super().forward(
            torch.log_softmax(student_output / self.temperature, dim=1),
            torch.softmax(teacher_output / self.temperature, dim=1),
        )
        hard_loss = super().forward(torch.log_softmax(student_output, 1), targets)
        return self.alpha * hard_loss + self.beta * (self.temperature ** 2) * soft_loss

class EMA(object):
    def __init__(self, teacher_model_list, student_model_list, momentum=0.999):
        super(EMA, self).__init__()
        self.teacher_model_list = teacher_model_list
        self.student_model_list = student_model_list
        self.momentum = momentum

    @torch.no_grad()
    def step(self):
        for teacher_model, student_model in zip(self.teacher_model_list, self.student_model_list):
            for teacher_parameter, student_parameter in zip(
                teacher_model.parameters(), student_model.parameters()
            ):
                teacher_parameter.mul_(self.momentum).add_(
                    (1.0 - self.momentum) * student_parameter
                )

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.counterk=0
        self.warmup_epoch=10
        # Caffe Alexnet for singleDG task, Leave-one-out PACS DG task.
        # self.extractor = caffenet(args.n_classes).to(device)
        self.extractor = resnet18(classes=args.n_classes).to(device)
        self.teacher = resnet18(classes=args.n_classes).to(device)
        self.teacher.load_state_dict(torch.load("./ckpt1/best_art_painting_stage1"))
        self.teacher.eval()
        self.kd_loss=KDLoss().cuda()
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=False)
        if len(self.args.target) > 1:
            self.target_loader = data_helper.get_multiple_val_dataloader(args, patches=False)
        else:
            self.target_loader = data_helper.get_val_dataloader(args, patches=False)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        self.ema = EMA([self.teacher],[self.extractor])
        # Get optimizers and Schedulers, self.discriminator
        self.optimizer = torch.optim.SGD(self.extractor.parameters(), lr=self.args.learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
        self.lr = self.args.learning_rate
        self.n_classes = args.n_classes
        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None


    def _do_epoch(self, epoch=None):
        criterion = nn.CrossEntropyLoss()
        self.extractor.train()
        count_nums=0
        count_acc=0
        total_loss=0
        for it, ((data, _, class_l), _, idx) in enumerate(self.source_loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            data,class_l = cutmix(data,class_l)
            self.optimizer.zero_grad()
            logits, tuple = self.extractor(data)
            # Total loss & backward
            # TODO: 分类损失
            with torch.no_grad():
                t_logits, t_tuple = self.teacher(data)
            class_loss = self.kd_loss(logits,t_logits,class_l)
            # class_loss = criterion(logits, class_l)
            # TODO: phase1
            class_loss.backward()
            self.optimizer.step()
            _, cls_pred = logits.max(dim=1)
            count_acc += (cls_pred ==  class_l.argmax(1)).sum().item()
            count_nums += class_l.shape[0]
            total_loss += class_loss.item()
            if epoch%50==0:
                self.ema.step()
        total_loss = total_loss/len(self.source_loader)
        total_acc = round(count_acc*100 / count_nums, 2)
        print(f"Epoch: {epoch}, Acc: {total_acc}%")
        self.extractor.eval()
        with torch.no_grad():
            if len(self.args.target) > 1:
                avg_acc = 0
                for i, loader in enumerate(self.target_loader):
                    total = len(loader.dataset)
                    class_correct = self.do_test(loader)
                    class_acc = float(class_correct) / total
                    self.logger.log_test('test', {"class": class_acc})
                    avg_acc += class_acc
                avg_acc = avg_acc / len(self.args.target)
                print(f"Epoch: {epoch}, Acc: {avg_acc}%")
                return total_loss, epoch, avg_acc
            else:
                for phase, loader in self.test_loaders.items():
                    if self.args.task == 'HOME' and phase == 'val':
                        continue
                    total = len(loader.dataset)
                    class_correct = self.do_test(loader)
                    class_acc = float(class_correct) / total
                    print(f"Phase: {phase}, Epoch: {epoch}, Acc: {class_acc}%")
                return total_loss, epoch, class_acc
    def do_test(self, loader):
        class_correct = 0
        for it, ((data, nouse, class_l), _, _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)
            z = self.extractor(data, train=False)[0]
            _, cls_pred = z.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def do_training(self):
        prev_loss = 999
        train_loss = 99
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        current_high = 0
        for self.current_epoch in range(self.args.epochs):
            self.warm_up(self.current_epoch,now_loss=train_loss,prev_loss=prev_loss,decay_rate=0.9)
            prev_loss = train_loss
            train_loss, current_epoch, val_acc = self._do_epoch(self.current_epoch)
            if val_acc > current_high:
                print('Saving Best model ...')
                torch.save(self.extractor.state_dict(), os.path.join('ckpt1/', 'best_' + self.args.target[0] + '_stage2'))
                current_high = val_acc
        print(f"Best Acc is {current_high}%")

    def do_eval(self):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        self.extractor.eval()
        total_acc=0
        total_nums=0
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                total_acc += float(class_correct)
                total_nums+= total
        acc = round(total_acc/total_nums,2)
        print(f"Best Acc is {acc}%")

    def warm_up(self, epoch, now_loss=None, prev_loss=None, decay_rate=0.9):
        if epoch <= self.warmup_epoch:
            self.optimizer.param_groups[0]["lr"] = self.lr * epoch / self.warmup_epoch
        elif now_loss is not None and prev_loss is not None:
            delta = prev_loss - now_loss
            if delta / now_loss < 0.02 and delta < 0.02 and delta > -0.02:
                self.optimizer.param_groups[0]["lr"] *= decay_rate
        p_lr = self.optimizer.param_groups[0]["lr"]
        print(f"lr = {p_lr}")

def main():
    args = get_args()

    if args.task == 'PACS':
        args.n_classes = 7
        # args.source = ['art_painting', 'cartoon', 'sketch']
        # args.target = ['photo']
        # args.source = ['art_painting', 'photo', 'cartoon']
        # args.target = ['sketch']
        # args.source = ['art_painting', 'photo', 'sketch']
        # args.target = ['cartoon']
        args.source = ['photo', 'cartoon', 'sketch']
        args.target = ['art_painting']
        # --------------------- Single DG
        # args.source = ['photo']
        # args.target = ['art_painting', 'cartoon', 'sketch']

    elif args.task == 'VLCS':
        args.n_classes = 5
        # args.source = ['CALTECH', 'LABELME', 'SUN']
        # args.target = ['PASCAL']
        args.source = ['LABELME', 'SUN', 'PASCAL']
        args.target = ['CALTECH']
        # args.source = ['CALTECH', 'PASCAL', 'LABELME' ]
        # args.target = ['SUN']
        # args.source = ['CALTECH', 'PASCAL', 'SUN']
        # args.target = ['LABELME']

    elif args.task == 'HOME':
        args.n_classes = 65
        # args.source = ['real', 'clip', 'product']
        # args.target = ['art']
        # args.source = ['art', 'real', 'product']
        # args.target = ['clip']
        # args.source = ['art', 'clip', 'real']
        # args.target = ['product']
        args.source = ['art', 'clip', 'product']
        args.target = ['real']
    # --------------------------------------------
    print("Target domain: {}".format(args.target))
    fix_all_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trainer = Trainer(args, device)
    if not args.eval:
        trainer.do_training()
    else:
        trainer.extractor.load_state_dict(torch.load(''))
        trainer.do_eval()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
