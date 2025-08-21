import os
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from torchvision.utils import make_grid  # save_image
import albumentations as A
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score, confusion_matrix
from torchvision import transforms as tf
import cv2
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def save_image(tensor, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((256, 256))
    im.save(filename)


def get_val_dataset(args, dataroot, tra_test):
    if args.dataset == "DDR":
        from dataloaders.datasets.DDR_24 import DDRDataset
        val_dataset = DDRDataset(root=dataroot, mode='test', transform1=tra_test, resize=args.resize)
        return val_dataset
    else:
        from dataloaders.datasets.IDRiD_24 import IDRiDDataset
        val_dataset = IDRiDDataset(root=dataroot, mode='test', transform=tra_test, resize=args.resize)
        return val_dataset


def cal_acc(grad, all_target, all_output):
    """acc = round(accuracy_score(all_target, all_output), 4)
    precision_ma = round(precision_score(all_target, all_output, average="macro"), 4)
    precision_w = round(precision_score(all_target, all_output, average="weighted"), 4)
    f1_ma = round(f1_score(all_target, all_output, average="macro"), 4)
    f1_mi = round(f1_score(all_target, all_output, average="micro"), 4)
    kappa = round(cohen_kappa_score(all_target, all_output), 4)  #
    print("[{}]: ACC:{}, Precision_ma:{}, Precision_w:{}, F1_ma:{}, F1_mi:{}， kappa:{}".format(grad, acc, precision_ma, precision_w, f1_ma, f1_mi, kappa))"""

    accs = ACC(all_target, all_output, 2 if grad == 1 else 4)
    """ses = sen(all_target, all_output, 2 if grad == 1 else 4)
    sps = spe(all_target, all_output, 2 if grad == 1 else 4)"""
    precisions = precision_score(all_target, all_output, average=None)
    f1s = F1(all_target, all_output, 2 if grad == 1 else 4)
    """for a in accs:
        print("accs:", round(a, 4))"""
    """for se in ses:
        print("ses:", round(se, 4))"""
    """for sp in sps:
        print("sps:", round(sp, 4))"""
    """for p in precisions:
        print("precisions:", round(p, 4))"""
    return accs, precisions, f1s  # , ses, sps

def calculate(all_target1, all_output1, all_target2, all_output2):
    accs1, precisions1, f1s1 = cal_acc(1, all_target1, all_output1)
    accs2, precisions2, f1s2 = cal_acc(2, all_target2, all_output2)
    """for c in range(4):
        print("{}: acc:{:.4f}  precision:{:.4f}  F1:{:.4f}".format(c, accs2[c], precisions2[c], f1s2[c]))
    print("4: acc:{:.4f}  precision:{:.4f}  F1:{:.4f}".format(accs1[1], precisions1[1], f1s1[1]))"""
    return (accs1[1] + sum(accs2)) / 5, (precisions1[1] + sum(precisions2)) / 5, (f1s1[1] + sum(f1s2)) / 5


def draw_confusion_matrix(root, grad, all_target, all_output):
    cm = confusion_matrix(all_target, all_output)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(np.arange(2 if grad == 1 else 4))
    plt.yticks(np.arange(2 if grad == 1 else 4))
    plt.title(f"Confusion Matrix{grad}")
    # plt.colorbar()
    plt.savefig(os.path.join(root, f'Confusion Matrix{grad}.png'), bbox_inches='tight')


def predict(img_name, st):
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out-path', type=str, default='./run/CHN6/',
                        help='mask image to save')
    parser.add_argument('--resize', action='store_true', default=True,
                        help='resize input or not')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt1', type=str, default='/Users/yanggq/Downloads/stream-seg/model/ygq_grading/run/DDR/DinkNet34/img_vessel_xception2_se_pre/best_acc.pth.tar',
                        help='saved model')
    parser.add_argument('--ckpt2', type=str, default='/Users/yanggq/Downloads/stream-seg/model/ygq_grading/run/DDR/DinkNet34/label_4_hff_cross/best_acc.pth.tar',
                        help='saved model')
    """parser.add_argument('--ckpt1', type=str,
                        default='/home/yanggq/xcfq/ygq_grading/run/idrid/DinkNet34/image_vessel_xception_se',
                        help='saved model')
    parser.add_argument('--ckpt2', type=str,
                        default='/home/yanggq/xcfq/ygq_grading/run/idrid/DinkNet34/label_4_hff_small_cross',
                        help='saved model')"""
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--dataroot', type=str, default='/Users/yanggq/Downloads/xcfq/ygq_grading/data',
                        help='root path of datasets')
    parser.add_argument('--dataset', type=str, default='DDR',
                        choices=['DDR', 'idrid', 'eophtha', 'IDRiD_M', 'IDRiD_SAMESIZE'],
                        help='dataset name (default: DeepGlobe)')
    parser.add_argument('--classes1', type=int, default=2,
                        choices=[2, 3, 4],
                        help='number of classes')
    parser.add_argument('--classes2', type=int, default=4,
                        choices=[2, 3, 4],
                        help='number of classes')
    parser.add_argument('--image-size', type=int, default=512,
                        help='base image size. DeepGlobe:1024.')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()

    kwargs = {'num_workers': args.workers, 'pin_memory': False}

    def worker_init_fn(worker_id):
        random.seed(1 + worker_id)

    from pages.grading.modeling.first.Xception import xception
    model1 = xception(args.classes1)
    from pages.grading.modeling.HFF_self import HiFuse_Small
    model2 = HiFuse_Small(num_classes=args.classes2)

    ckpt_root1 = os.path.join(args.ckpt1)
    ckpt1 = torch.load(ckpt_root1, map_location=torch.device('cpu'))

    model1.load_state_dict(ckpt1['state_dict'])
    ckpt_root2 = os.path.join(args.ckpt2)
    ckpt2 = torch.load(ckpt_root2, map_location=torch.device('cpu'))
    model2.load_state_dict(ckpt2['state_dict'])

    out_path = os.path.join(args.out_path, 'Output', 'DinkNet34/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    """channel_mean = torch.tensor([0.485, 0.456, 0.406])
    channel_std = torch.tensor([0.229, 0.224, 0.225])
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]

    # 归一化和反归一化生成器
    denormalizer = tf.Normalize(mean=MEAN, std=STD)"""

    model1.eval()
    model2.eval()
    # tbar = tqdm(test_loader, desc='\r')
    print('Test:')
    """print('best epoch:', ckpt['epoch'])"""


    dataroot = os.path.join(args.dataroot, args.dataset)

    if args.resize is True:
        transform1 = None
    else:
        transform1 = A.Compose([
            A.Resize(height=args.image_size, width=args.image_size, interpolation=cv2.INTER_CUBIC)])


    mode = 'test'
    transform2 = A.Compose([
        A.Resize(height=224, width=224, interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, p=0.5)]) if mode == 'train' else A.Compose([
        A.Resize(height=224, width=224, interpolation=cv2.INTER_CUBIC)])
    
    name_list = []
    grad_label_list = []
    norm = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = True

    def find_string_index(string_list, target_string):
        try:
            index = string_list.index(target_string)
            return index

        except ValueError:
            return -1  # 如果字符串不在列表中，返回 -1 或其他你喜欢的默认值

    grad_label_list = []
    with open(dataroot + f'/{mode}.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            name, label = line.strip().split(' ')
            if int(label) != 5:
                name_list.append(name)
                grad_label_list.append(int(label))


    name = img_name
    print(name)
    index = find_string_index(name_list,name)
    print(index)
    grad2 = grad_label_list[find_string_index(name_list,name)]
    if grad2 != 4:
        grad1 = 0
    else:
        grad1 = 1

    grad1 = torch.tensor(grad1, dtype=torch.long)
    grad2 = torch.tensor(grad2, dtype=torch.long)



    from PIL import Image
    if args.resize is True:
        img1 = Image.open(dataroot + f'/{mode}/img_vessel/' + name).convert('RGB')  # img_vessel
    else:
        img1 = Image.open(dataroot + f'/{mode}/image/' + name).convert('RGB')  # img_vessel
    img1 = np.asarray(img1)  # h,w,3

    img2 = Image.open(dataroot + f'/{mode}/img_labels/' + name).convert('RGB')
    img2 = np.asarray(img2)
    if transform1 is not None:
        al = transform1(image=img1)  #
        img1 = al['image']
    if transform2 is not None:
        al = transform2(image=img2)  #
        img2 = al['image']
    img1, img2 = tf.ToTensor()(img1), tf.ToTensor()(img2)
    image1, image2 = norm(img1), norm(img2)  # 3,512,512
    image1 = image1.unsqueeze(0)  # 形状变为 (1, c, h, w)
    image2 = image2.unsqueeze(0)  # 形状变为 (1, c, h, w)

    
    # img_name = sample[1][0].split('.')[0]
    # _, _, h, w = image1.shape
    output = model1(image1)  #
    indices1 = torch.max(output, dim=1)[1]


    if grad1==0:
        output2 = model2(image2)
        indices2 = torch.max(output2, dim=1)[1]
    else:
        output2 = 0
    print(grad1)

    print(output2)
    #x_value = x_normalized = (output2 - output2.min(dim=1, keepdim=True).values) / (output2.max(dim=1, keepdim=True).values - output2.min(dim=1, keepdim=True).values)
    
    """
    st.subheader('Output Result')
    cols = st.columns(4)
    cols[0].markdown(f"**病变等级: {indices1.cpu().data.numpy()[0]}**")  # 自动识别 Markdown/HTML
    """
    
    return output2

    




