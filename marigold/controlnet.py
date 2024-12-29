# Last modified: 2024-05-24
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os

from typing import Dict, Optional, Union, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .marigold_pipeline import MarigoldPipeline, MarigoldDepthOutput
from diffusers.models.controlnet import ControlNetModel
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer
from src.util.seeding import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from torch.utils.data import DataLoader, TensorDataset

import json

class ControlNetMarigoldPipeline(MarigoldPipeline):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        controlnet: Optional[Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]]] = None,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__(
            unet,
            vae,
            scheduler,
            text_encoder,
            tokenizer,
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )
        if controlnet is None:
            print("ControlNet is not provided, use unet to initialize controlnet.")
            controlnet = ControlNetModel.from_unet(
                unet=unet,
                conditioning_channels=4,
            )
        self.register_modules(controlnet=controlnet)
    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        input_depth: Union[np.ndarray, torch.Tensor],
        input_mask: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image mask depth Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
            rgb = rgb.permute(0, 3, 1, 2)  # [1, rgb, H, W]
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong image input shape {input_size}, expected [1, rgb, H, W]"

        
        if isinstance(input_mask, Image.Image):
            input_mask = input_mask.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            mask = pil_to_tensor(input_mask)
            mask = mask.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_mask, torch.Tensor):
            mask = input_mask
            mask = mask.permute(0, 3, 1, 2)  # [1, rgb, H, W]
        else:
            raise TypeError(f"Unknown input type: {type(input_mask) = }")
        assert (
            4 == mask.dim() and 3 == mask.shape[-3]
        ), f"Wrong mask input shape {input_size}, expected [1, rgb, H, W]"

        if isinstance(input_depth, np.ndarray):
            depth = torch.from_numpy(input_depth)
            depth = depth.unsqueeze(0)  # [1, H, W]
            depth = depth.unsqueeze(0)  # [1, 1, H, W]
        elif isinstance(input_depth, torch.Tensor):
            depth = input_depth
        else:
            raise TypeError(f"Unknown input type: {type(input_depth) = }")
        assert (
            4 == depth.dim() and 1 == depth.shape[-3]
        ), f"Wrong depth input shape {input_size}, expected [1, 1, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
            depth = resize_max_res(
                depth,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
            mask = resize_max_res(
                mask,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        mask_norm: torch.Tensor = mask / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        mask_norm = mask_norm.to(self.dtype)
        
        assert mask_norm.min() >= -1.0 and mask_norm.max() <= 1.0
        
        depth_norm: torch.Tensor = depth / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        depth_norm = depth_norm.to(self.dtype)
        assert depth_norm.min() >= -1.0 and depth_norm.max() <= 1.0
        
        

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_depth = depth_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_mask = mask_norm.expand(ensemble_size, -1, -1, -1)
        
        single_rgb_dataset = TensorDataset(duplicated_rgb, duplicated_depth, duplicated_mask)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img, batched_depth, batched_mask) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                depth_in=batched_depth,
                mask_in=batched_mask,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        depth_in: torch.Tensor,
        mask_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        depth_in = depth_in.to(device)
        mask_in = mask_in.to(device)
        

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)
        #PROB: encode condition?
        # depth_latent = self.encode_depth(depth_in)
        # mask_latent = self.encode_rgb(mask_in)
        
        #TODO: use know depth?
        # Initial depth map (noise)
        depth_latent =  torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important
            controlnet_input = torch.cat(
                [depth_in, mask_in], dim=1
            )
            
            # predict the controlnet residual
            #TODO: control_scale value?
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                unet_input,
                t,
                encoder_hidden_states=batch_empty_text_embed,
                controlnet_cond=controlnet_input,
                conditioning_scale=1.0,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed,
                down_block_additional_residuals=down_block_res_samples, 
                mid_block_additional_residual=mid_block_res_sample, 
                
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean 
    
    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked
    
    # TODO: how to simplify this?
    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        model.controlnet = model.controlnet.to(*args, **kwargs)
        return model


if "__main__" == __name__:
    half_precision = False
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.warning(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe = MarigoldPipeline.from_pretrained(
        'checkpoint/marigold-depth-v1-0', variant=variant, torch_dtype=dtype
    )
    config = json.load(open("marigold_controlnet/config.json"))
    controlnet = ControlNetModel.from_unet(pipe.unet)
    print(controlnet)
