import os
import json
import random
import numpy as np
from PIL import Image
from decord import VideoReader
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ResizeByLongSide:
    def __init__(self, target_long, interpolation=transforms.InterpolationMode.BILINEAR):
        self.target_long = target_long
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        long_side = max(w, h)
        scale = self.target_long / long_side
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation, antialias=True)

class CustomTrainDataset(Dataset):
    def __init__(
        self,
        json_index_path,           # 必需：主索引文件路径
        followbench_root,          # 必需：原始数据根目录 (e.g. .../FollowBench/train)
        warped_video_root,         # 必需：Warp 数据根目录 (e.g. .../WarpedVideo/train)
        height=512,
        width=512,
        resize_long=1505,
        sample_n_frames=49,
        stride=2,
        is_one2three=True,
        training_len=-1,
        **kwargs # 吸收多余参数
    ):  
        self.stride = stride
        self.training_len = training_len
        self.is_one2three = is_one2three
        
        self.followbench_root = followbench_root
        self.warped_video_root = warped_video_root
        
        self.height = height
        self.width = width
        self.resize_long = resize_long
        self.sample_n_frames = sample_n_frames
        
        # [核心修复] 加载 JSON 并转换为列表
        print(f"[Dataset] Loading index: {json_index_path}")
        with open(json_index_path, 'r') as f:
            data_raw = json.load(f)
            
        # 将 Dict 转换为 List，并排序以确保多卡训练时顺序一致
        if isinstance(data_raw, dict):
            # 按照 Case ID (Key) 排序
            self.data_list = [data_raw[k] for k in sorted(data_raw.keys())]
        elif isinstance(data_raw, list):
            self.data_list = data_raw
        else:
            raise ValueError(f"Unsupported JSON format: {type(data_raw)}")
        
        print(f"[Dataset] Found {len(self.data_list)} samples.")

        # 定义 Transforms
        self.video_transforms = transforms.Compose([
            ResizeByLongSide(self.resize_long), 
            transforms.CenterCrop([self.height, self.width]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if self.is_one2three:
            self.image_transforms = transforms.Compose([   
                transforms.Resize([self.height, self.width]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        # 如果指定了 training_len (用于 epoch 长度控制)，则返回该长度
        if self.training_len != -1:
            return self.training_len
        return len(self.data_list)

    def __getitem__(self, index):
        # 循环取数据，防止 index 越界
        item = self.data_list[index % len(self.data_list)]
        
        # [核心修复] 1. 使用您 JSON 中的正确键名解析路径
        # JSON 示例: "the first view": "train_case_08644/ego.mp4"
        ego_rel = item.get('the first view')
        exo_rel = item.get('the third view')
        ref_rel = item.get('reference')
        
        # 拼接完整路径
        ego_path = os.path.join(self.followbench_root, ego_rel)
        exo_path = os.path.join(self.followbench_root, exo_rel)
        
        # [核心修复] 2. 解析 Warped Video 路径
        # 逻辑：WarpedVideo 保持同样的目录结构，文件名固定为 ego_warped_video.mp4
        # ego_rel 是 "train_case_08644/ego.mp4"，dirname 是 "train_case_08644"
        case_folder = os.path.dirname(ego_rel) 
        warped_path = os.path.join(self.warped_video_root, case_folder, "ego_warped_video.mp4")
        
        ref_path = None
        if self.is_one2three and ref_rel:
            ref_path = os.path.join(self.followbench_root, ref_rel)

        # 3. 读取视频 (带异常捕获和重试)
        try:
            ego_vr = VideoReader(ego_path)
            exo_vr = VideoReader(exo_path)
            
            # 检查 Warped 视频是否存在
            if os.path.exists(warped_path):
                warped_vr = VideoReader(warped_path)
            else:
                # 训练时通常要求存在，如果不存在可以报错或生成全黑
                # 这里为了稳健，如果找不到就生成全黑占位 (需注意这会失去几何引导)
                # print(f"[Warning] Warped video not found: {warped_path}")
                warped_vr = None 
                
        except Exception as e:
            print(f"[Dataset Error] Failed to load {item}: {e}")
            # 随机换一个样本重试
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # 4. 采样索引 (确保时间对齐)
        len_ego = len(ego_vr)
        len_exo = len(exo_vr)
        len_warped = len(warped_vr) if warped_vr else len_ego
        
        min_len = min(len_ego, len_exo, len_warped)
        
        # 确保有足够的帧
        if min_len < self.sample_n_frames:
             # 如果视频太短，重试其他样本
             return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        # 随机起始点
        max_start = max(min_len - (self.sample_n_frames - 1) * self.stride, 0)
        start_idx = np.random.randint(0, max_start + 1)
        indices = start_idx + np.arange(self.sample_n_frames) * self.stride
        indices = np.clip(indices, 0, min_len - 1)

        # 5. 获取 Tensor
        def load_frames(vr, indices):
            frames = vr.get_batch(indices).asnumpy()
            return torch.stack([self.video_transforms(Image.fromarray(f)) for f in frames])

        ego_pixel = load_frames(ego_vr, indices).permute(1, 0, 2, 3)    # [C, F, H, W]
        exo_pixel = load_frames(exo_vr, indices).permute(1, 0, 2, 3)    # [C, F, H, W]
        
        if warped_vr:
            warped_pixel = load_frames(warped_vr, indices).permute(1, 0, 2, 3)
        else:
            # Fallback: 全黑 (C, F, H, W)
            warped_pixel = torch.zeros_like(ego_pixel)

        ref_pixel = []
        if self.is_one2three and ref_path:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_pixel = self.image_transforms(ref_img) # [C, H, W]
            except Exception:
                ref_pixel = torch.zeros((3, self.height, self.width))
        elif self.is_one2three:
             ref_pixel = torch.zeros((3, self.height, self.width))

        # Prompt 处理 (如果没有 prompt 字段则使用默认)
        prompt = item.get('prompt', 'Transform ego view to exo view.')

        return {
            'ref_pixel_values': ref_pixel,       # [C, H, W]
            'first_pixel_values': ego_pixel,     # [C, F, H, W]
            'third_pixel_values': exo_pixel,     # [C, F, H, W]
            'warped_pixel_values': warped_pixel, # [C, F, H, W]
            'prompts': prompt,
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