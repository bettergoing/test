import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.spatial.distance import directed_hausdorff


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype('int') + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape, f"Prediction shape {imgPredict.shape} does not match label shape {imgLabel.shape}!"
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def intersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def dice(self):
        intersection = np.diag(self.confusionMatrix)
        dice = 2 * intersection / (np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0))
        return dice


if __name__ == '__main__':

    NUM_CLASSES = 2
    NAME_CLASSES = ["background", "crack"]
    desktop_path = os.path.expanduser(r'G:/图像处理/第五篇_损伤修复/数据-终版/模型训练/定性结果')
    label_folder = os.path.join(desktop_path, 'label')
    pred_folder = os.path.join(desktop_path, 'TransUNet')
    TXT_FILE_NAME = "test.txt"
    txt_file_path = os.path.join(desktop_path, TXT_FILE_NAME)

    if not os.path.exists(label_folder) or not os.path.exists(pred_folder):
        print("Error: Please ensure 'label' and 'TransUNet' folders exist at the specified path.")
        exit()
    if not os.path.exists(txt_file_path):
        print(f"Error: TXT file '{TXT_FILE_NAME}' not found at the specified path.")
        exit()

    try:
        with open(txt_file_path, 'r') as f:
            image_basenames = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        exit()

    if not image_basenames:
        print(f"Error: The specified TXT file '{TXT_FILE_NAME}' is empty.")
        exit()

    print(f"Found {len(image_basenames)} files to process from '{TXT_FILE_NAME}'. Starting metrics calculation...")
    metric = SegmentationMetric(NUM_CLASSES)

    hausdorff_distances = []

    for basename in tqdm(image_basenames, desc="Calculating"):
        try:
            if basename.endswith('_ori'):
                image_id = basename[:-4]
            else:
                image_id = basename

            filename = image_id + '.png'

            label_path = os.path.join(label_folder, filename)
            pred_path = os.path.join(pred_folder, filename)

            if not os.path.exists(label_path) or not os.path.exists(pred_path):
                continue

            pred_img_pil = Image.open(pred_path).convert('L')
            label_img_pil = Image.open(label_path).convert('L')

            pred_img = np.array(pred_img_pil)
            label_img = np.array(label_img_pil)

            if np.any(label_img > (NUM_CLASSES - 1)):
                label_img[label_img > 0] = 1
            if np.any(pred_img > (NUM_CLASSES - 1)):
                pred_img[pred_img > 0] = 1

            metric.addBatch(pred_img, label_img)

            gt_binary = (label_img == 1).astype(np.uint8)
            pred_binary = (pred_img == 1).astype(np.uint8)

            gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pred_contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if gt_contours and pred_contours:
                gt_points = np.concatenate(gt_contours, axis=0).squeeze()
                pred_points = np.concatenate(pred_contours, axis=0).squeeze()

                if gt_points.ndim == 1:
                    gt_points = np.expand_dims(gt_points, axis=0)
                if pred_points.ndim == 1:
                    pred_points = np.expand_dims(pred_points, axis=0)

                hd1 = directed_hausdorff(gt_points, pred_points)[0]
                hd2 = directed_hausdorff(pred_points, gt_points)[0]
                hausdorff_distances.append(max(hd1, hd2))

        except Exception as e:
            print(f"\nError processing basename '{basename}': {e}")
            continue

    iou = metric.intersectionOverUnion()
    dice = metric.dice()

    mean_hd = np.nanmean(hausdorff_distances) if hausdorff_distances else float('inf')

    print("\nCalculation finished.")
    print("--------------------------------------------------")
    for i in range(NUM_CLASSES):
        try:
            print(f'===> {NAME_CLASSES[i]}: \t'
                  f'IoU: {iou[i] * 100:.2f}; \t'
                  f'Dice: {dice[i] * 100:.2f}')
        except:
            print(f'===> {NAME_CLASSES[i]}: Error calculating metrics (maybe division by zero)')

    print("--------------------------------------------------")
    print(f'===> Mean IoU: {np.nanmean(iou) * 100:.2f}')
    print(f'===> Mean Dice: {np.nanmean(dice) * 100:.2f}')
    print(f'===> Mean Hausdorff Distance: {mean_hd:.2f}')
    print("--------------------------------------------------")
