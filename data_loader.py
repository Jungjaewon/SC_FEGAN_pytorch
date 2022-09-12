import torch
import os.path as osp
import numpy as np
import glob
import random
import cv2
import math

from torch.utils import data
from torchvision import transforms as T
from PIL import Image, ImageDraw
from glob import glob
from collections import defaultdict


class Dataset(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, config, transform, transform_hed, mode='train'):
        """Initialize and preprocess the Polyevore dataset."""
        self.image_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], mode)
        self.img_size = config['TRAINING_CONFIG']['IMG_SIZE']
        self.ann_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], 'CelebAMask-HQ-mask-anno')
        self.transform = transform
        self.transform_hed = transform_hed
        self.mode = config['TRAINING_CONFIG']['MODE']
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.ann_dict = defaultdict(list)

        self.edge_drop_ratio = float(config['TRAINING_CONFIG']['EDGE_DROP_RATIO'])
        self.color_drop_ratio = float(config['TRAINING_CONFIG']['COLOR_DROP_RATIO'])
        self.face_drop_ratio = float(config['TRAINING_CONFIG']['FACE_DROP_RATIO'])
        self.mask_drop_ratio = float(config['TRAINING_CONFIG']['MASK_DROP_RATIO'])

        for img_path in glob(osp.join(self.ann_dir, '*', '*')):
            img_name = osp.basename(img_path)
            face_id = img_name.split('_')[0]
            self.ann_dict[face_id].append(img_path)

        img_list = glob(osp.join(self.image_dir, '*.jpg'))
        img_list = [osp.basename(x).replace('.jpg', '') for x in img_list]
        img_list = [x.split('_')[0] for x in img_list]
        self.img_list = list(set(img_list))

    def __getitem__(self, index):
        face_id = self.img_list[index]
        face = Image.open(osp.join(self.image_dir, f'{face_id}.jpg')).convert('RGB')
        color = Image.open(osp.join(self.image_dir, f'{face_id}_color.jpg')).convert('RGB')
        sketch = Image.open(osp.join(self.image_dir, f'{face_id}_hed.jpg')).convert('L')

        prob = random.uniform(0, 1)

        if face_id not in self.ann_dict:
            #mask = self.bbox2mask() if random.uniform(0, 1) > 0.5 else self.random_ff_mask()
            mask = self.random_ff_mask()
            mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        else:
            if prob < 0.5:
                mask = self.random_ff_mask()
                mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
                #print(f'ff mask : {mask.size()}')
            else:
                mask = self.face_mask(face_id)
                #print(f'face mask : {mask.size()}')

        face_gt = self.transform(face)
        face_in = face_gt * (1 - mask)

        if random.uniform(0, 1) < self.face_drop_ratio:
            if random.uniform(0, 1) < self.mask_drop_ratio:
                mask, noise = torch.zeros_like(mask), torch.zeros_like(mask)
                color, sketch = self.transform(color), self.transform_hed(sketch)
                if random.uniform(0, 1) < self.color_drop_ratio:
                    color = torch.zeros_like(color)
                elif random.uniform(0, 1) < self.edge_drop_ratio:
                    sketch = torch.zeros_like(sketch)
            else:
                color, sketch = self.transform(color) * mask, self.transform_hed(sketch) * mask
                noise = torch.randn_like(sketch) * mask
                if random.uniform(0, 1) < self.color_drop_ratio:
                    color = torch.zeros_like(color)
                elif random.uniform(0, 1) < self.edge_drop_ratio:
                    sketch = torch.zeros_like(sketch)
        else:
            color = self.transform(color) * mask
            color = torch.zeros_like(color) if random.uniform(0, 1) < self.color_drop_ratio else color

            sketch = self.transform_hed(sketch) * mask
            sketch = torch.zeros_like(sketch) if random.uniform(0, 1) < self.edge_drop_ratio else sketch

            noise = torch.randn_like(sketch) * mask

        face_id = torch.LongTensor([int(face_id)])

        return face_id, face_in, mask, sketch, color, noise, face_gt

    def __len__(self):
        """Return the number of images."""
        return len(self.img_list)

    def random_ff_mask(self, max_angle=360, max_len=100, max_width=80, times=15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        https://github.com/csqiangwen/DeepFillv2_Pytorch/blob/master/train_dataset.py
        """
        height, width = 512, 512
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times - 10, times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len - 20, max_len)
                brush_w = 5 + np.random.randint(max_width - 30, max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1,) + mask.shape).astype(np.float32)

    def face_mask(self, face_id):

        face_mask_pool = self.ann_dict[face_id]
        pick_m = random.randint(1, len(face_mask_pool) - 1)
        selected_mask = random.sample(face_mask_pool, pick_m)
        selected_mask = [Image.open(x).convert('1') for x in selected_mask]
        selected_mask = [T.Resize(512, 512)(x) for x in selected_mask]
        selected_mask = [T.ToTensor()(x) for x in selected_mask]
        selected_mask = [torch.reshape(x, (1, 1, 512, 512)) for x in selected_mask]

        return torch.cat(selected_mask, dim=1).max(dim=1, keepdim=True)[0][0]

    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape=512, margin=10, bbox_shape=80, times=6):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = list()
        times = np.random.randint(1, times)
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape # img_size
        width = shape # img_size
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)


def get_loader(config, mode='train'):
    """Build and return a data loader."""
    transform, transform_hed = list(), list()

    transform.append(T.Resize((config['TRAINING_CONFIG']['IMG_SIZE'], config['TRAINING_CONFIG']['IMG_SIZE'])))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform_hed.append(T.Resize((config['TRAINING_CONFIG']['IMG_SIZE'], config['TRAINING_CONFIG']['IMG_SIZE'])))
    transform_hed.append(T.ToTensor())
    transform_hed.append(T.Normalize(mean=(0.5), std=(0.5)))

    transform = T.Compose(transform)
    transform_hed = T.Compose(transform_hed)

    dataset = Dataset(config, transform, transform_hed, mode)
    batch_size = config['TRAINING_CONFIG']['BATCH_SIZE'] if mode == 'train' else 1

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
