import os
import time
from PIL import Image
from tqdm import tqdm
from unet import Unet


def batch_predict_recursive(
    model: Unet,
    root_input_dir: str,
    root_output_dir: str,
) -> None:

    for subdir, _, files in os.walk(root_input_dir):
        rel_path = os.path.relpath(subdir, root_input_dir)
        output_subdir = os.path.join(root_output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)

        image_files = [f for f in files if f.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]

        if not image_files:
            continue

        print(f"Processing folder: {subdir} -> {output_subdir}")

        total_time = 0.0

        for img_name in tqdm(image_files, desc=f"Predicting in {rel_path}", leave=False):
            input_path = os.path.join(subdir, img_name)
            output_path = os.path.join(output_subdir, img_name)
            image = Image.open(input_path)
            start = time.perf_counter()
            result = model.detect_image(image)
            elapsed = time.perf_counter() - start
            total_time += elapsed
            result.save(output_path)

        avg_time = total_time / len(image_files)
        print(f"  Processed {len(image_files)} images in {total_time:.2f}s (avg: {avg_time:.3f}s/image)")


if __name__ == "__main__":
    ROOT_INPUT_DIR = r"G:\图像处理\第五篇_损伤修复\数据-终版\模型训练\定性结果\灰度"    # Folder containing subfolders with OCT images
    ROOT_OUTPUT_DIR = r"D:\Desktop\111"   # Where predicted results will be stored

    unet = Unet()

    batch_predict_recursive(unet, ROOT_INPUT_DIR, ROOT_OUTPUT_DIR)
