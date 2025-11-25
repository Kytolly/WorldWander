
import os
from decord import VideoReader
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class ResizeByLongSide:
    def __init__(self, target_long, interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_long = target_long
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size  # PIL.Image.size = (width, height)
        long_side = max(w, h)
        scale = self.target_long / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)


class CustomTrainDataset(Dataset):
    def __init__(
        self,
        first_video_root,
        third_video_root,
        ref_image_root=None,
        height=512,
        width=512,
        resize_long=1505,
        sample_n_frames=49,
        stride=2,
        is_one2three=False,
        training_len=-1,
    ):  
        self.stride = stride
        self.training_len = training_len
        self.is_one2three = is_one2three
        #
        self.first_video_root = first_video_root
        self.third_video_root = third_video_root
        self.ref_image_root = ref_image_root
        #
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames
        #
        self.first_video_paths = [os.path.join(self.first_video_root, video) for video in sorted(os.listdir(self.first_video_root))]
        self.third_video_paths = [os.path.join(self.third_video_root, video) for video in sorted(os.listdir(self.third_video_root))]
        if self.is_one2three:
            self.ref_image_paths = [os.path.join(self.ref_image_root, image) for image in sorted(os.listdir(self.ref_image_root))]
        # video transforms
        self.train_video_transforms = transforms.Compose(
            [
                ResizeByLongSide(self.resize_long), 
                transforms.CenterCrop([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
        if self.is_one2three:
            # image transforms
            self.train_image_transforms = transforms.Compose(
                [   
                    transforms.Resize([self.height, self.width]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],  
                        std=[0.5, 0.5, 0.5]
                    )
                ]
            )
        # 
        self.first_video_len = len(self.first_video_paths)
        self.third_video_len = len(self.third_video_paths)
        if self.is_one2three:
            self.ref_image_len = len(self.ref_image_paths)
        #
        assert self.first_video_len == self.third_video_len, "mismatch in first videos and third videos"
        if self.is_one2three:
            assert self.third_video_len == self.ref_image_len, "mismatch in third videos and reference frames"

    def __len__(self):
        if self.training_len == -1:
            return self.first_video_len
        else:
            return self.training_len

    def __getitem__(self, index):
        index = index % self.first_video_len
        while True:
            # first video
            first_video_path = self.first_video_paths[index]
            first_video_reader = VideoReader(first_video_path)
            first_frame_length = len(first_video_reader)
            # third video
            third_video_path = self.third_video_paths[index]
            third_video_reader = VideoReader(third_video_path)
            third_frame_length = len(third_video_reader)
            try:
                assert first_frame_length == third_frame_length
                assert self.sample_n_frames <= first_frame_length
                break
            except AssertionError:
                print(f"Assertion failed! video_path={first_video_path}, video_path2={third_video_path}")
                index = random.randint(0, min(self.first_video_len, self.third_video_len) - 1)
                continue

        if self.is_one2three:
            ref_image_path = self.ref_image_paths[index]
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_pixel_values = self.train_image_transforms(ref_image)
        else:
            ref_pixel_values = []
        # random index
        max_start = max(first_frame_length - (self.sample_n_frames - 1) * self.stride, 0)
        start_index = np.random.randint(0, max_start - 3)
        frame_indices = start_index + np.arange(self.sample_n_frames) * self.stride
        # first video
        first_video = first_video_reader.get_batch(frame_indices).asnumpy() # F, H, W, C
        first_video = [Image.fromarray(frame) for frame in first_video]
        first_pixel_values = torch.stack([self.train_video_transforms(frame) for frame in first_video]) # F, C, H, W
        # third video
        third_video = third_video_reader.get_batch(frame_indices).asnumpy()
        third_video = [Image.fromarray(frame) for frame in third_video]
        third_pixel_values = torch.stack([self.train_video_transforms(frame) for frame in third_video]) # F, C, H, W

        # return dict
        return {
            'ref_pixel_values': ref_pixel_values, # C, H, W
            'first_pixel_values': first_pixel_values.permute(1, 0, 2, 3), # C, F, H, W
            'third_pixel_values': third_pixel_values.permute(1, 0, 2, 3), # C, F, H, W
            'prompts': 'Transform it into the third-person perspective.' if self.is_one2three else 'Transform it into the first-person perspective.',
        }


class CustomTestDataset(Dataset):
    def __init__(
        self,
        original_video_root,
        ref_image_root=None,
        height=512,
        width=512,
        sample_n_frames=49,
        stride=1,
        is_one2three=False,
        training_len=-1,
    ):  
        self.stride = stride
        self.training_len = training_len
        self.is_one2three = is_one2three
        #
        self.original_video_root = original_video_root
        if self.is_one2three:
            self.ref_image_root = ref_image_root
        #
        self.height = height
        self.width = width
        self.sample_n_frames = sample_n_frames
        #
        self.original_video_paths = [os.path.join(self.original_video_root, video) for video in sorted(os.listdir(self.original_video_root))]
        if self.is_one2three:
            self.ref_image_paths = [os.path.join(self.ref_image_root, image) for image in sorted(os.listdir(self.ref_image_root))]
        # video transforms
        self.test_video_transforms = transforms.Compose(
            [
                # ResizeByLongSide(1505), 
                # transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],  
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
        if self.is_one2three:
            # image transforms
            self.test_image_transforms = transforms.Compose(
                [
                    # ResizeByLongSide(1505), 
                    # transforms.CenterCrop((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],  
                        std=[0.5, 0.5, 0.5]
                    )
                ]
            )
        # 
        self.original_video_len = len(self.original_video_paths)
        if self.is_one2three:
            self.ref_image_len = len(self.ref_image_paths)
            assert self.original_video_len == self.ref_image_len, "mismatch in original videos and reference images"

    def __len__(self):
        if self.training_len == -1:
            return self.original_video_len
        else:
            return self.training_len

    def __getitem__(self, index):
        index = index % self.original_video_len

        if self.is_one2three:
            ref_image_path = self.ref_image_paths[index]
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_pixel_values = self.test_image_transforms(ref_image)
        else:
            ref_pixel_values = []

        original_video_path = self.original_video_paths[index]
        input_frame_list = [os.path.join(original_video_path, frame_path) for frame_path in sorted(os.listdir(original_video_path))]
        input_frame_list = [Image.open(frame_path).convert("RGB") for frame_path in input_frame_list]
        input_pixel_values = torch.stack([self.test_video_transforms(frame) for frame in input_frame_list]) # F, C, H, W
        
        # additional return path for save name
        path = original_video_path.split('/')[-1]
        
        # return
        return {
            'ref_pixel_values': ref_pixel_values, # C, H, W
            'input_pixel_values': input_pixel_values.permute(1, 0, 2, 3), # C, F, H, W
            'prompts': 'Transform it into the third-person perspective.' if self.is_one2three else 'Transform it into the first-person perspective.',
            'path': path,
        }


# for debug
if __name__ == '__main__':
    my_dataset = CustomTrainDataset(
        video_root='your-path1',
        video_root2='your-path2',
        first_video_root='your-path3',
        height=480,
        width=720,
        sample_n_frames=81,
    )
    my_dataloader = DataLoader(my_dataset,
                               batch_size=1,
                               shuffle=False)
    #
    for batch in my_dataset:
        breakpoint()
