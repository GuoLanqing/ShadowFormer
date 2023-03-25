import os
from argparse import Namespace

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import get_arch, load_checkpoint


class DataLoaderInference(Dataset):
    def __init__(self, image_path, mask_path):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path

    def __getitem__(self, index):
        image = cv2.imread(self.image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.0

        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.float32(mask) / 255.0

        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor, os.path.basename(self.image_path)

    def __len__(self):
        return 1

def predict_output_image(args, input_image, shadow_file):
    dataset = DataLoaderInference(input_image, shadow_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model_restoration = get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)
    load_checkpoint(model_restoration, args.weights)

    # model_restoration.load_state_dict(torch.load(args.weights, map_location=args.device))
    model_restoration.to(args.device)
    model_restoration.eval()

    with torch.no_grad():
        for noisy, mask, noisy_filename in dataloader:
            print('sex')
            # if mask file is empty, then return original image
            print(noisy_filename)
            print(mask.sum())
            if mask.sum() == 0:
                return noisy.squeeze(0).permute(1, 2, 0).cpu().numpy()

            noisy = noisy.to(args.device)
            mask = mask.to(args.device)

            output = model_restoration(noisy, mask)
            output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

            return output_image


def process_images(input_dir, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if img_name.endswith("_mask.jpg") or img_name.endswith("_mask.png"):
            continue
        img_path = os.path.join(input_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + "_mask" + os.path.splitext(img_name)[-1]
        mask_path = os.path.join(input_dir, mask_name)
        if os.path.isfile(mask_path):
            output_image = predict_output_image(args, img_path, mask_path)
            output_image_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_image_path, (output_image * 255.0))


if __name__ == "__main__":
    args = Namespace(
        weights="log/ShadowFormer_istd/models/model_epoch_650.pth",
        input_dir="dataset/",
        output_dir="out",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        arch="ShadowFormer",
        embed_dim=32,
        win_size=10,
        token_projection="linear",
        token_mlp="leff",
        vit_dim=256,
        vit_depth=12,
        vit_nheads=8,
        vit_mlp_dim=512,
        vit_patch_size=16,
        global_skip=False,
        local_skip=False,
        vit_share=False,
        train_ps=320
    )

    process_images(args.input_dir, args.output_dir, args)
