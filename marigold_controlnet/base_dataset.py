# %%
import io
import os
import random
import tarfile
from enum import Enum
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.depth_transform import DepthNormalizerBase, ScaleShiftDepthNormalizer
import pandas as pd
import cv2
import trimesh
import matplotlib.pyplot as plt
import builtins

def random_brush_mask(image_shape, num_strokes=[1,2,3], min_brush_size=20, max_brush_size=40, min_points=5, max_points=15):
    '''
    image_shape: (H, W)
    num_strokes: the possible number of strokes, will choose one randomly
    min_brush_size: the minimum size of the brush
    max_brush_size: the maximum size of the brush
    min_points: the minimum number of points in a stroke
    max_points: the maximum number of points in a stroke'''
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    num_strokes = np.random.choice(num_strokes)
    for _ in range(num_strokes):
        x, y = np.random.randint(0, image_shape[1]), np.random.randint(0, image_shape[0])
        # random resize the brush size with scale
        scale = np.random.uniform(0.5, 1.5)
        for _ in range(np.random.randint(min_points, max_points)):
            brush_size = int(np.random.randint(min_brush_size, max_brush_size)*scale)
            dx, dy = np.random.randint(-brush_size, brush_size, size=2)
            x, y = np.clip(x + dx, 0, image_shape[1] - 1), np.clip(y + dy, 0, image_shape[0] - 1)
            cv2.circle(mask, (x, y), brush_size, 1, -1)
    return mask

def video_to_image_list(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"can open video file: {video_path}")
        return None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        frames.append(image)
    cap.release()
    
    return frames

def center_crop_pil_image(input_image, target_width=720, target_height=480):
    w, h = input_image.size
    h_ratio = h / target_height
    w_ratio = w / target_width

    if h_ratio > w_ratio:
        h = int(h / w_ratio)
        if h < target_height:
            h = target_height
        input_image = input_image.resize((target_width, h), Image.Resampling.LANCZOS)
    else:
        w = int(w / h_ratio)
        if w < target_width:
            w = target_width
        input_image = input_image.resize((w, target_height), Image.Resampling.LANCZOS)
    return ImageOps.fit(input_image, (target_width, target_height), Image.BICUBIC)

class BaseDataset(Dataset):
    def __init__(
        self,
        filename_ls_path: str,
        dataset_dir: str,
        depth_transform: Union[DepthNormalizerBase, None] = ScaleShiftDepthNormalizer(),
        rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        random_mask_prob = 0.5,
        disp_name: str = 'DL3DV',
        H: int = 336,
        W: int = 512,
        **kwargs,
    ) -> None:
        super().__init__()
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.rgb_transform = rgb_transform
        self.random_mask_prob = random_mask_prob
        self.H = H
        self.W = W
        self.disp_name = disp_name
        df = pd.read_csv(self.filename_ls_path)
        self.file_list = df['path'].tolist()


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        index = index % builtins.len(self.file_list)
        file_sub_path = self.file_list[index]
        try:
            str_index = file_sub_path.find("videos")
            file_sub_path = file_sub_path[str_index+builtins.len("videos/"):-4]
            
            video_path = os.path.join(self.dataset_dir,'videos', f'{file_sub_path}.mp4')
            point_path = os.path.join(self.dataset_dir,'point_cloud', file_sub_path,'pc_0.ply')
            camera_path = os.path.join(self.dataset_dir,'point_cloud', file_sub_path,'camera.pt')
            mask_dir_path = os.path.join(self.dataset_dir,'mask', file_sub_path,'0')

            # handle rgb
            frame_pil = video_to_image_list(video_path)[0]
            frame_pil = center_crop_pil_image(frame_pil)
            frame_pil = frame_pil.crop((0,4,720,476))
            frame_pil = frame_pil.resize((self.W,self.H))
            rgb = np.array(frame_pil)
            rgb_norm = rgb
            rgb_norm = np.transpose(rgb_norm, (2, 0, 1)).astype(int)  # [rgb, H, W]
            rgb_norm = rgb_norm / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            # handle depth
            camera_param = torch.load(camera_path, map_location='cpu', weights_only=True)
            c2w = camera_param['c2ws'][0]
            # 3. 读取点云，计算得到深度图
            mesh = trimesh.load(point_path, process=False)
            points = mesh.vertices
            # 计算世界坐标系到相机坐标系的变换矩阵
            w2c = torch.zeros(4, 4).to(c2w)
            w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
            w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
            w2c[3, 3] = 1.0
            # 将点云转换为 PyTorch 张量
            points = torch.from_numpy(points).to(torch.float32)
            # print(f"{file_sub_path} 1:", points.min(), points.max())
            # 将点云转换为齐次坐标
            points_homo = torch.concat([points, torch.ones((points.shape[0], 1))], dim=1)
            # 将点云从世界坐标系转换到相机坐标系
            points_camera = torch.matmul(points_homo, w2c.T)
            # 提取深度值（相机坐标系下的 z 值）
            depth = points_camera[:, 2].reshape(1,self.H,self.W)
            # print(f"{file_sub_path} 2:", depth.min(), depth.max())
            depth_handled = self.depth_transform(depth)
            # handel mask
            if np.random.rand() < self.random_mask_prob:
                mask = random_brush_mask((self.H, self.W))
                mask = 255 * np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
                mask_handled = torch.from_numpy(mask).float().permute(2,0,1)
            else:
                masks = os.listdir(mask_dir_path)
                mask_path = np.random.choice(masks)
                mask = Image.open(os.path.join(mask_dir_path,mask_path)).convert('RGB')
                mask = np.array(mask.resize((self.W,self.H)))
                mask_handled = mask.transpose(2,0,1)
                mask_handled = torch.from_numpy(mask_handled).float()
            
            mask_handled = mask_handled / 255.0 * 2 - 1
            masked_depth = depth.clone()
            masked_depth[mask_handled[:1]==1] = 0.0
            masked_depth_handled = self.depth_transform(masked_depth)

            return {
                "rgb_int": rgb,
                "rgb_norm": torch.from_numpy(rgb_norm).float(),
                "depth": depth,
                "depth_handled": depth_handled,
                "mask": mask,
                "mask_handled": mask_handled,
                "masked_depth": masked_depth,
                "masked_depth_handled": masked_depth_handled,
                "rgb_name": os.path.basename(file_sub_path),
            }
        except:
            print(f'error in {index}, with file name: {file_sub_path}')
            return self.__getitem__(index+1)

def visualize_outputs(output_dict, save_path):
    """
    可视化 __getitem__ 函数返回的所有值，并显示值的范围。

    参数:
        output_dict (dict): __getitem__ 函数的返回值。
        save_path (str): 保存图像的路径。
    """
    # 提取数据
    rgb_int = output_dict["rgb_int"]  # [H, W, 3]
    rgb_norm = output_dict["rgb_norm"].numpy().transpose(1, 2, 0)  # [H, W, 3]
    depth = output_dict["depth"].squeeze().numpy()  # [H, W]
    depth_handled = output_dict["depth_handled"].squeeze().numpy()  # [H, W]
    mask = output_dict["mask"]  # [H, W, 3]
    mask_handled = output_dict["mask_handled"].squeeze().numpy()  # [H, W]
    masked_depth = output_dict["masked_depth"].squeeze().numpy()  # [H, W]
    masked_depth_handled = output_dict["masked_depth_handled"].squeeze().numpy()  # [H, W]

    # 创建画布
    plt.figure(figsize=(15, 10))

    # 绘制 rgb_int
    plt.subplot(3, 3, 1)
    plt.imshow(rgb_int)
    plt.title(f"RGB (int)\nRange: [{rgb_int.min():.2f}, {rgb_int.max():.2f}]")
    plt.axis('off')

    # 绘制 rgb_norm
    plt.subplot(3, 3, 2)
    plt.imshow((rgb_norm + 1) / 2)  # 将 [-1, 1] 映射到 [0, 1]
    plt.title(f"RGB (normalized)\nRange: [{rgb_norm.min():.2f}, {rgb_norm.max():.2f}]")
    plt.axis('off')

    # 绘制 depth
    plt.subplot(3, 3, 3)
    plt.imshow(depth, cmap='viridis')
    plt.title(f"Depth (raw linear)\nRange: [{depth.min():.2f}, {depth.max():.2f}]")
    plt.colorbar()
    plt.axis('off')

    # 绘制 depth_handled
    plt.subplot(3, 3, 4)
    plt.imshow(depth_handled, cmap='viridis')
    plt.title(f"Depth (handled)\nRange: [{depth_handled.min():.2f}, {depth_handled.max():.2f}]")
    plt.colorbar()
    plt.axis('off')

    # 绘制 mask
    plt.subplot(3, 3, 5)
    plt.imshow(mask)
    plt.title(f"Mask\nRange: [{mask.min():.2f}, {mask.max():.2f}]")
    plt.axis('off')

    # 绘制 mask_handled
    plt.subplot(3, 3, 6)
    plt.imshow((mask_handled.transpose(1, 2, 0) + 1) / 2)  # 将 [-1, 1] 映射到 [0, 1]
    plt.title(f"Mask (handled)\nRange: [{mask_handled.min():.2f}, {mask_handled.max():.2f}]")
    plt.axis('off')

    # 绘制 masked_depth
    plt.subplot(3, 3, 7)
    plt.imshow(masked_depth, cmap='viridis')
    plt.title(f"Masked Depth\nRange: [{masked_depth.min():.2f}, {masked_depth.max():.2f}]")
    plt.colorbar()
    plt.axis('off')

    # 绘制 masked_depth_handled
    plt.subplot(3, 3, 8)
    plt.imshow(masked_depth_handled, cmap='viridis')
    plt.title(f"Masked Depth (handled)\nRange: [{masked_depth_handled.min():.2f}, {masked_depth_handled.max():.2f}]")
    plt.colorbar()
    plt.axis('off')

    # 显示图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # 关闭图像以释放内存
    


if __name__ == "__main__":
# %%
    dataset = BaseDataset(
        filename_ls_path="/home/lishiyang/data/Marigold_Inpaint/data.csv",
        dataset_dir="/home/lishiyang/data/Marigold_Inpaint/DL3DV-ALL-960P",
        H=336,
        W=512,
        random_mask_prob=0.5,
    )
    len = len(dataset)
    for i in range(len):
        data = dataset[i]
        visualize_outputs(data, f"tmp/0{i}.png")
        os.makedirs(f"tmp/{i}", exist_ok=True)
        np.save(f"tmp/{i}/depth.npy", data["depth"].squeeze().numpy())
        Image.fromarray(data["mask"].astype(np.uint8)).save(f"tmp/{i}/mask.jpg")
        Image.fromarray(data["rgb_int"].astype(np.uint8)).save(f"tmp/{i}/rgb.jpg")        
