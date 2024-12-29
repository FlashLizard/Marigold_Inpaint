import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from marigold import ControlNetMarigoldPipeline
from marigold.marigold_pipeline import MarigoldDepthOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence

from accelerate import Accelerator
import pdb



class ControlNetMarigoldTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: ControlNetMarigoldPipeline,
        train_dataloader: DataLoader,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accelerator,
        val_dataloader: DataLoader = None,
        vis_dataloader: DataLoader = None,
    ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.cfg: OmegaConf = cfg
        lr = self.cfg.lr
        optimizer = Adam(model.controlnet.parameters(), lr=lr)
        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)
        model, val_dataloader, vis_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, val_dataloader, vis_dataloader, optimizer, lr_scheduler)
        model.to(self.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model: ControlNetMarigoldPipeline = model
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loader: DataLoader= val_dataloader
        self.vis_loader: DataLoader = vis_dataloader

        # Adapt input layers
        # if 8 != self.model.unet.config["in_channels"]:
        #     self._replace_unet_conv_in()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone()

        self.model.unet.enable_xformers_memory_efficient_attention()
        self.model.controlnet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(False)
        self.model.controlnet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def train(self, t_end=None):
        logging.info("Start training")

        # device = self.device
        # self.model.to(device)
        
        
        accelerator = self.accelerator
        self.val_loader.batch_size = 1
        if  accelerator.is_main_process:
            if self.in_evaluation:
                logging.info(
                    "Last evaluation was not finished, will do evaluation before continue training."
                )
            self.validate()
            self.train_metrics.reset()
        
        #PROB: how to accumulate in accelerator
        # accumulated_step = 0

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                with accelerator.accumulate(self.model):
                
                    # self.model.unet.train()
                    self.model.controlnet.train()
                    device = accelerator.device

                    # globally consistent random generators
                    if self.seed is not None:
                        local_seed = self._get_next_seed()
                        rand_num_generator = torch.Generator(device=device)
                        rand_num_generator.manual_seed(local_seed)
                    else:
                        rand_num_generator = None

                    # >>> With gradient accumulation >>>

                    # Get data
                    rgb = batch["rgb_norm"].to(device)
                    depth_mask = batch["mask_handled"].to(device)
                    depth_gt_for_latent = batch["depth_handled"].to(device) 
                    depth = batch["masked_depth_handled"].to(device)

                    # PROB: what the use of the mask
                    # if self.gt_mask_type is not None:
                    #     valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    #     invalid_mask = ~valid_mask_for_latent
                    #     valid_mask_down = ~torch.max_pool2d(
                    #         invalid_mask.float(), 8, 8
                    #     ).bool()
                    #     valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                    # else:
                    #     raise NotImplementedError

                    batch_size = rgb.shape[0]

                    with torch.no_grad():
                        # Encode image
                        rgb_latent = self.model.encode_rgb(rgb)  # [B, 4, h, w]
                        # Encode GT depth
                        gt_depth_latent = self.encode_depth(
                            depth_gt_for_latent,
                            self.model
                        )  # [B, 4, h, w]
                        
                        
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.scheduler_timesteps,
                        (batch_size,),
                        device=device,
                        generator=rand_num_generator,
                    ).long()  # [B]

                    # Sample noise
                    if self.apply_multi_res_noise:
                        strength = self.mr_noise_strength
                        if self.annealed_mr_noise:
                            # calculate strength depending on t
                            strength = strength * (timesteps / self.scheduler_timesteps)
                        noise = multi_res_noise_like(
                            gt_depth_latent,
                            strength=strength,
                            downscale_strategy=self.mr_noise_downscale_strategy,
                            generator=rand_num_generator,
                            device=device,
                        )
                    else:
                        noise = torch.randn(
                            gt_depth_latent.shape,
                            device=device,
                            generator=rand_num_generator,
                        )  # [B, 4, h, w]

                    # Add noise to the latents (diffusion forward process)
                    noisy_latents = self.training_noise_scheduler.add_noise(
                        gt_depth_latent, noise, timesteps
                    )  # [B, 4, h, w]

                    # Text embedding
                    text_embed = self.empty_text_embed.to(device).repeat(
                        (batch_size, 1, 1)
                    )  # [B, 77, 1024]

                    # Concat rgb and depth latents
                    cat_latents = torch.cat(
                        [rgb_latent, noisy_latents], dim=1
                    )  # [B, 8, h, w]
                    cat_latents = cat_latents.float()
                    
                    controlnet_input = torch.cat(
                        [depth, depth_mask], dim=1
                    )
                    
                    # predict the controlnet residual
                    #TODO: control_scale value?
                    down_block_res_samples, mid_block_res_sample = self.model.controlnet(
                        cat_latents,
                        timesteps,
                        encoder_hidden_states=text_embed,
                        controlnet_cond=controlnet_input,
                        conditioning_scale=1.0,
                        return_dict=False,
                    )
                    
                    # Predict the noise residual
                    model_pred = self.model.unet(
                        cat_latents, timesteps, text_embed,
                        down_block_additional_residuals=down_block_res_samples, 
                        mid_block_additional_residual=mid_block_res_sample, 
                    ).sample  # [B, 4, h, w]
                    if torch.isnan(model_pred).any():
                        logging.warning("model_pred contains NaN.")

                    # Get the target for loss depending on the prediction type
                    if "sample" == self.prediction_type:
                        target = gt_depth_latent
                    elif "epsilon" == self.prediction_type:
                        target = noise
                    elif "v_prediction" == self.prediction_type:
                        target = self.training_noise_scheduler.get_velocity(
                            gt_depth_latent, noise, timesteps
                        )  # [B, 4, h, w]
                    else:
                        raise ValueError(f"Unknown prediction type {self.prediction_type}")

                    # Masked latent loss
                    #PROB: use ?
                    # if self.gt_mask_type is not None:
                    #     latent_loss = self.loss(
                    #         model_pred[valid_mask_down].float(),
                    #         target[valid_mask_down].float(),
                    #     )
                    # else:
                    #     latent_loss = self.loss(model_pred.float(), target.float())
                    latent_loss = self.loss(model_pred.float(), target.float())

                    loss = latent_loss.mean()
                    # loss.backward()
                    accelerator.backward(loss)
                    # Practical batch end
                    # Perform optimization step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    loss = loss.detach()
                    loss = loss.item()
                    loss = accelerator.gather(torch.tensor(loss).to(device))
                    loss = loss.mean()
                    
                    if accelerator.is_main_process:
                        self.train_metrics.update("loss", loss.cpu())
                        self.effective_iter += 1
                        self.n_batch_in_epoch += 1
                        # Log to tensorboard
                        accumulated_loss = self.train_metrics.result()["loss"]
                        tb_logger.log_dic(
                            {
                                f"train/{k}": v
                                for k, v in self.train_metrics.result().items()
                            },
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "lr",
                            self.lr_scheduler.get_last_lr()[0],
                            global_step=self.effective_iter,
                        )
                        tb_logger.writer.add_scalar(
                            "n_batch_in_epoch",
                            self.n_batch_in_epoch,
                            global_step=self.effective_iter,
                        )
                        logging.info(
                            f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                        )
                        self.train_metrics.reset()

                        # Per-step callback
                        #TODO: not only main process
                        self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        if accelerator.is_main_process:
                            self.save_checkpoint(
                                ckpt_name=self._get_backup_ckpt_name(),
                                save_train_state=False,
                            )
                            logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        if accelerator.is_main_process:
                            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                            logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                        # <<< Effective batch end <<<
            # Epoch end
            if accelerator.is_main_process:
                self.n_batch_in_epoch = 0

    def encode_depth(self, depth_in, model):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        accelerator = self.accelerator
        if accelerator.is_main_process and self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            if accelerator.is_main_process:
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            if accelerator.is_main_process:
                self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            accelerator.is_main_process
            and self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if accelerator.is_main_process and self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        accelerator = self.accelerator
        val_dataset_name = self.val_loader.dataset.disp_name
        val_metric_dic = self.validate_single_dataset(
            data_loader=self.val_loader, metric_tracker=self.val_metrics
        )
        if accelerator.is_main_process:
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=self.val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            main_eval_metric = val_metric_dic[self.main_val_metric]
            if (
                "minimize" == self.main_val_metric_goal
                and main_eval_metric < self.best_metric
                or "maximize" == self.main_val_metric_goal
                and main_eval_metric > self.best_metric
            ):
                self.best_metric = main_eval_metric
                logging.info(
                    f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                )
                # Save a checkpoint
                self.save_checkpoint(
                    ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                )

    def visualize(self):
        vis_dataset_name = self.vis_loader.dataset.disp_name
        vis_out_dir = os.path.join(
            self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
        )
        os.makedirs(vis_out_dir, exist_ok=True)
        _ = self.validate_single_dataset(
            data_loader=self.vis_loader,
            metric_tracker=self.val_metrics,
            save_to_dir=vis_out_dir
        )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        if self.accelerator.is_main_process:
            metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            # Read input image
            rgb_int = batch["rgb_int"]  # [B, 3, H, W]
            mask_in = batch["mask"]
            # GT depth
            depth_raw_ts = batch["depth"].squeeze()
            depth_raw = depth_raw_ts.cpu().numpy()
            depth_raw_ts = depth_raw_ts.to(self.accelerator.device)
            depth_in = batch["masked_depth"]
            
            #PROB: use?
            # valid_mask_ts = batch["valid_mask_raw"].squeeze()
            # valid_mask = valid_mask_ts.numpy()
            # valid_mask_ts = valid_mask_ts.to(self.device)
            valid_mask = np.ones_like(depth_raw).squeeze().astype(int)
            valid_mask_ts = torch.from_numpy(valid_mask).to(self.accelerator.device)
            
            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                #TODO: change to self.device
                # generator = torch.Generator(device=self.device)
                generator = torch.Generator(self.accelerator.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                depth_in,
                mask_in,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            # Clip to dataset min max
            #TODO: use?
            # depth_pred = np.clip(
            #     depth_pred,
            #     a_min=data_loader.dataset.min_depth,
            #     a_max=data_loader.dataset.max_depth,
            # )

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                # _metric = torch.tensor(met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()).to(self.accelerator.device)
                # pdb.set_trace()
                # _metric = self.accelerator.gather(_metric)
                if self.accelerator.is_main_process:
                    sample_metric.append(_metric.__str__())
                    # metric_tracker.update(_metric_name, _metric.cpu())
                    metric_tracker.update(_metric_name, _metric)

            # Save as 16-bit uint png
            if self.accelerator.is_main_process and save_to_dir is not None:
                img_name = batch["rgb_name"][0]
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                masked_depth_handled = batch['masked_depth_handled'].squeeze()
                depth_handled = batch['depth_handled'].squeeze()
                depth_handled_16bit = ((depth_handled + 1) / 2 * 65535).cpu().numpy().astype(np.uint16)
                masked_depth_handled_16bit = ((masked_depth_handled + 1) / 2 * 65535).cpu().numpy().astype(np.uint16)
                depth_np_16bit = (pipe_out.depth_np * 65535).astype(np.uint16)
                spacing_width = 10

                # 创建一个空白图像用于拼接
                height = depth_handled_16bit.shape[0]
                total_width = depth_handled_16bit.shape[1] + masked_depth_handled_16bit.shape[1] + depth_np_16bit.shape[1] + 2 * spacing_width
                combined_image = np.zeros((height, total_width), dtype=np.uint16)

                # 将三张图像拼接到一起，并添加间隔
                combined_image[:, :depth_handled_16bit.shape[1]] = depth_handled_16bit
                combined_image[:, depth_handled_16bit.shape[1] + spacing_width: depth_handled_16bit.shape[1] + spacing_width + masked_depth_handled_16bit.shape[1]] = masked_depth_handled_16bit
                combined_image[:, depth_handled_16bit.shape[1] + masked_depth_handled_16bit.shape[1] + 2 * spacing_width:] = depth_np_16bit
                
                Image.fromarray(combined_image).save(png_save_path, mode="I;16")

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save controlnet
        controlnet_path = os.path.join(ckpt_dir, "controlnet")
        self.model.controlnet.save_pretrained(controlnet_path, safe_serialization=False)
        logging.info(f"ControlNet is saved to: {controlnet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "controlnet", "diffusion_pytorch_model.bin")
        self.model.controlnet.load_state_dict(
            torch.load(_model_path)
        )
        logging.info(f"ControlNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return