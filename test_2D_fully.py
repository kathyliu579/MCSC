import argparse
import os
import shutil
import h5py
import nibabel as nib
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, kernel):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, padding=1)
        #  init my layers
        nn.init.xavier_normal_(self.conv1.weight)

    def forward(self, x, dropout=True):
        x = self.conv1(x)
        return x


def calculate_metric_percase(pred, gt, clas):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    flag = 0 # if flag ==1, we will not count this value in metric
    
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    else:
        dice = metric.binary.dc(pred, gt)
        hd95 = 0
        asd = 0
        
        if np.sum(gt) != 0:
           flag = 1
           print(clas, 1) 
        return dice, 0, 0


def test_single_volume_confident(case, net, classifier, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    confidence_part = np.zeros_like(label, dtype="float32")

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]

        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            confidence, out = torch.max(torch.softmax(classifier(out_main[3]), dim=1), dim=1)
            out = out.squeeze(0).cpu().detach().numpy()
            confidence = confidence.squeeze(0).cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            confidence = zoom(confidence, (x / 256, y / 256), order=0)

            prediction[ind] = pred
            confidence_part[ind] = confidence

    first_metric = calculate_metric_percase(prediction == 1, label == 1, 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2, 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3, 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric, prediction, confidence_part, label


def test_single_volume(case, net, classifier, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                classifier(out_main[3]), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    if test_save_path != None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        img_itk.SetSpacing((1, 1, 10))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        prd_itk.SetSpacing((1, 1, 10))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        lab_itk.SetSpacing((1, 1, 10))

        sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}/{}/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)

    classifier = Classifier(in_dim=16, out_dim=4, kernel=3).cuda()

    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))

    save_classfier_path = os.path.join(
        snapshot_path, '{}_best_classifier1.pth'.format(FLAGS.model))
    
    # save_mode_path = os.path.join(
    #     snapshot_path, 'model1_iter_40000_dice_0.8128.pth')
    # save_classfier_path = os.path.join(
    #     snapshot_path, 'classifier1_iter_40000_dice_0.8128.pth')
    
    # save_mode_path = os.path.join(
    #     snapshot_path, 'model1_iter_30600_dice_0.8871.pth')
    # save_classfier_path = os.path.join(
    #     snapshot_path, 'classifier1_iter_30600_dice_0.8871.pth')

    net.load_state_dict(torch.load(save_mode_path))
    classifier.load_state_dict(torch.load(save_classfier_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    classifier.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    confidence_val = np.array([])
    pred_class_val = np.array([])
    true_class_val = np.array([])
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric, pred_class_part_val, confidence_part_val, gt = test_single_volume_confident(
            case, net, classifier, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        confidence_val = np.append(confidence_val, confidence_part_val)
        pred_class_val = np.append(pred_class_val, pred_class_part_val)
        true_class_val = np.append(true_class_val, gt)
    print('dice', first_total)
    
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]

    # print(confusion_matrix(true_class_val, pred_class_val))
    # class_names = ['0', 'rv', 'myo', 'lv']
    # print(classification_report(true_class_val, pred_class_val, target_names=class_names))
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)

    print(['rv', 'myo', 'lv'], metric)
    print((metric[0] + metric[1] + metric[2]) / 3)
