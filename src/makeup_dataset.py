from torch.utils.data import Dataset
from torchvision import transforms
import os
import os.path as osp
from PIL import Image
import cv2
import numpy as np
import albumentations as A
import re
import random

class MakeupDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        size=1024,
    ):
        self.size = size
        src_dirs = [
            'dataset/FFHQ-v10/src',
            'dataset/FFHQ-v11/src',
            'dataset/FFHQ-v12/src',
            'dataset/FFHQ-v13/src',
            'dataset/FFHQ-v14/src',
            'dataset/FFHQ-v15/src',
            'dataset/FFHQ-v16/src',
            'dataset/FFHQ-v17/src',
            'dataset/FFHQ-v18/src',
            'dataset/FFHQ-v19/src',
        ]
        ref_dirs = [
            'dataset/FFHQ-v10/ref',
            'dataset/FFHQ-v11/ref',
            'dataset/FFHQ-v12/ref',
            'dataset/FFHQ-v13/ref',
            'dataset/FFHQ-v14/ref',
            'dataset/FFHQ-v15/ref',
            'dataset/FFHQ-v16/ref',
            'dataset/FFHQ-v17/ref',
            'dataset/FFHQ-v18/ref',
            'dataset/FFHQ-v19/ref',
        ]
        self.mask_dir = 'dataset/mask'

        self.src_paths = []
        self.ref_paths = []

        # 遍历每一组数据（ref 为基准）
        for src_dir, ref_dir in zip(src_dirs, ref_dirs):
            ref_filenames = sorted(os.listdir(ref_dir))
            for fname in ref_filenames:
                src_path = osp.join(src_dir, fname)
                ref_path = osp.join(ref_dir, fname)

                # 加入存在性检查
                if osp.exists(src_path):
                    self.src_paths.append(src_path)
                    self.ref_paths.append(ref_path)
                else:
                    print(f"[Warning] Missing src for: {fname}")

        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.aug1 = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.Rotate(limit=20,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0,
                     p=0.5)
        ], additional_targets={'image0': 'image', 'image1': 'image'})

        self.aug2 = A.Compose([
            A.Resize(height=1024, width=1024),
            A.Rotate(limit=20,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=0,
                     p=0.5),
            A.ElasticTransform(
                alpha=180,
                sigma=12,
                alpha_affine=60,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.9
            ),
            A.Affine(
                scale=(0.7, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                p=0.5
            ),
        ])

        self.face_ids = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13]

    def __len__(self):
        return len(self.ref_paths)
    
    def __getitem__(self, idx):
        gt_path = self.ref_paths[idx]
        src_path = self.src_paths[idx]
        mask_path = osp.join(self.mask_dir, os.path.basename(gt_path))

        gt_image = Image.open(gt_path).convert("RGB").resize((self.size, self.size))
        src_image = Image.open(src_path).convert("RGB").resize((self.size, self.size))
        ref_image = Image.open(gt_path).convert("RGB").resize((1024, 1024))
        parsing_img = Image.open(mask_path).convert('L').resize((1024, 1024), resample=Image.NEAREST)

        gt_image = np.array(gt_image)
        src_image = np.array(src_image)
        ref_image = np.array(ref_image)
        parsing_img = np.array(parsing_img)
        face_mask = np.isin(parsing_img, self.face_ids).astype(np.uint8)
        ref_image = ref_image * face_mask[..., None]

        # Image.fromarray(gt_image).save('0-0.png')
        # Image.fromarray(src_image).save('0-1.png')
        # Image.fromarray(ref_image).save('0-2.png')

        aug1_results = self.aug1(image=gt_image, image0=src_image)
        gt_image, src_image = aug1_results["image"], aug1_results["image0"]

        aug2_results = self.aug2(image=ref_image)
        ref_image = aug2_results["image"]

        # Image.fromarray(gt_image).save('1-0.png')
        # Image.fromarray(src_image).save('1-1.png')
        # Image.fromarray(ref_image).save('1-2.png')
        # import pdb; pdb.set_trace()

        gt_image = self.train_transforms(gt_image)
        src_image = self.train_transforms(src_image)
        ref_image = self.train_transforms(ref_image)

        prompt = "makeup."

        return gt_image, prompt, src_image, ref_image


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = MakeupDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for i, (makeup_image, prompt, nonmakeup_image, aug_makeup_image) in enumerate(dataloader):
        pass

