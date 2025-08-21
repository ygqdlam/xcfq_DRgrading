import argparse
import os

import numpy as np
import torch
from torchvision.utils import make_grid  # save_image
# from dataloaders import make_data_loader
from pages.funds.utils.metrics import Evaluator
from tqdm import tqdm
from pages.funds.modeling.unet import Unet
from pages.funds.modeling.dinknet import DinkNet34
from pages.funds.modeling.Segformer import Segformer
from pages.funds.modeling.DBRANet import DBRANet_4
from pages.funds.modeling.HCTNet import HCTNet
from pages.funds.modeling.MAResUnet import MAResUNet
from pages.funds.modeling.NLLinkNet import NL34_LinkNet
import albumentations as A
from einops import rearrange
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image
from torchvision import transforms as tf
import cv2
import random
from scipy.ndimage import zoom




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


def predict(img_name, st):

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--out_path', type=str, default='./run/ygq/',
                        help='mask image to save')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default='/Users/yanggq/Downloads/stream-seg/model/512_att_weight/checkpoint.pth.tar',  #  checkpoint   best_mAUPR
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='con_ce',
                        choices=['ce', 'con_ce', 'focal'],
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--dataroot', type=str, default='/Users/yanggq/Downloads/crop_DDR_512',  # /home/xcfq/Grading/data  '/home/xcfq/datasets
                        help='root path of datasets')
    parser.add_argument('--dataset', type=str, default='IDRiD_SAMESIZE',
                        choices=['DDR', 'IDRiD', 'eophtha', 'idrid_grad', 'IDRiD_SAMESIZE'],
                        help='dataset name (default: DeepGlobe)')
    parser.add_argument('--classes', type=int, default=4,
                        choices=[2, 4],
                        help='number of classes')
    parser.add_argument('--image-size', type=int, default=512,
                        help='base image size. DeepGlobe:1024.')


    args = parser.parse_args()

    model = HCTNet(image_size=(args.image_size, args.image_size),  num_classes=args.classes)

    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])

    save_path = os.path.join(args.out_path, 'Output')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    channel_mean = torch.tensor([0.485, 0.456, 0.406])
    channel_std = torch.tensor([0.229, 0.224, 0.225])
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]

    # 归一化和反归一化生成器
    denormalizer = tf.Normalize(mean=MEAN, std=STD)

    mAUPR = 0
    IoU = 0
    mIoU = 0
    F1 = 0
    mF1 = 0
    # evaluator = Evaluator(2)
    model.eval()
    print('Test:')
    print('best epoch:', ckpt['epoch'])


    name = img_name
    img = Image.open(args.dataroot + '/test/image/' + name).convert('RGB')
    img = np.asarray(img)
    labels = ["EX/", "HE/", "MA/", "SE/"]

    for label in labels:
        if label == "EX/":
            masks = Image.open(args.dataroot + '/test/label/' + label + name[0:-4] + '.tif')
            masks = tf.ToTensor()(masks)
        else:
            img_lable = Image.open(args.dataroot + '/test/label/' + label + name[0:-4] + '.tif')
            img_lable = tf.ToTensor()(img_lable)
            masks = [masks, img_lable]
            masks = torch.cat(masks, dim=0)
    masks = masks.permute(1, 2, 0).contiguous().numpy()  # c, h, w

    ves = Image.open(args.dataroot + '/test/vessel/' + name[0:-4] + '.gif')
    ves = np.asarray(ves)[:, :, np.newaxis]

    zoom_factors = (512 / img.shape[0], 512 / img.shape[1], 1)


    # 使用 zoom 进行插值
    img = zoom(img, zoom_factors, order=1)  # order=1 表示双线性插值
    ves = zoom(ves, zoom_factors, order=1)  # order=1 表示双线性插值
    masks = zoom(masks, zoom_factors, order=1)  # order=1 表示双线性插值

    img = tf.ToTensor()(img)
    norm = tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = norm(img)
    target = tf.ToTensor()(masks)
    ves = tf.ToTensor()(ves)
    image = image.unsqueeze(0)  # 形状变为 (1, c, h, w)
    target = target.unsqueeze(0)  # 形状变为 (1, c, h, w)
    ves = ves.unsqueeze(0)  # 形状变为 (1, c, h, w)

    stream_mask = []



    for n in range(args.classes):
        evaluator = Evaluator(2)
        if n == 0:  evaluator_all = Evaluator(2)
        all_targets = torch.empty((1, 512, 512)).cpu().numpy()
        all_outputs = torch.empty((1, 512, 512)).cpu().numpy()

        output = model(image, ves).cpu().data.numpy()  #
        o = np.zeros(output.shape)

        all_targets = target[:, n, :, :].cpu().data.numpy()  # b, 512, 512
        all_outputs = output[:, n, :, :]  # b, 512, 512
        
        o[output < 0.0066574435] = 0
        o[output >= 0.0066574435] = 1
        target_n = target.cpu().numpy()  #*255


        evaluator.add_batch(target_n[:, n, :, :], o[:, n, :, :])
        if n == 0: evaluator_all.add_batch(target_n, o)


        if not os.path.exists(save_path):
            os.makedirs(save_path)          
        mask = o * 255
        mask = np.array(mask)
        label = ["EX/", "HE/", "MA/", "SE/"]
        for c in range(4):
            save_out = save_path + "/" + label[c]
            if not os.path.exists(save_out):
                os.makedirs(save_out)
            cv2.imwrite(os.path.join(save_out, name), mask[0][c])

            stream_mask.append(mask[0][c])


        i = denormalizer(image[0]).cpu().numpy().transpose(1, 2, 0) * 255
        save_image = save_path + "/images"
        if not os.path.exists(save_image):
            os.makedirs(save_image)
        cv2.imwrite(os.path.join(save_image, name), i)



        all_targets = rearrange(all_targets, 'n h w -> (n h w)')
        all_targets = np.array([int(a) for a in all_targets])
        all_outputs = rearrange(all_outputs, 'n h w -> (n h w)')

        AUC = 0
        try:
            AUC = round(roc_auc_score(all_targets, all_outputs),  4)
        except ValueError:
            pass

        precision, recall, thresholds = precision_recall_curve(all_targets, all_outputs)
        AUPR = auc(recall, precision)
        mAUPR += AUPR
        AUPR = round(AUPR, 4)

        if args.classes == 2:
            labels = ["EX", "MA"]  # , "HE", "SE"
        elif args.classes == 4:
            labels = ["EX", "HE", "MA", "SE"]  # 
        IoU = evaluator.Intersection_over_Union()
        mIoU += IoU
        IoU = round(IoU, 4)
        Precision = round(evaluator.Pixel_Precision(), 4)
        F1 = evaluator.Pixel_F1()
        mF1 += F1
        F1 = round(F1, 4)
        print(labels[n] + ": AUPR:{}, AUC:{}, IoU:{}, F1:{}".format(AUPR, AUC, IoU, F1))  # MPA:{}, mIoU:{} , MPA, mIoU

        if n==0:
            IoU = round(evaluator_all.Intersection_over_Union(), 4)
            F1 = round(evaluator_all.Pixel_F1(), 4)

    """
    st.subheader('Output Image')
    cols = st.columns(4)
    cols[0].image(stream_mask[0], clamp=True, channels='GRAY', use_container_width=True, caption="EX病变")
    cols[1].image(stream_mask[1], clamp=True, channels='GRAY', use_container_width=True, caption="HE病变")
    cols[2].image(stream_mask[2], clamp=True, channels='GRAY', use_container_width=True, caption="MA病变")
    cols[3].image(stream_mask[3], clamp=True, channels='GRAY', use_container_width=True, caption="SE病变")
    print("mAUPR:{} IoU:{} mIoU:{}  F1:{} mF1:{}".format(round(mAUPR/args.classes, 4), round(IoU, 4), round(mIoU / args.classes, 4), round(F1, 4), round(mF1 / args.classes, 4)))
    """

    return stream_mask


