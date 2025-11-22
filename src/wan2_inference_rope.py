import os

import pytorch_lightning as L
import argparse
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import imageio.v3 as iio

from datasets.custom_dataset import CustomTestDataset
from torch.utils.data import DataLoader
from models.wan2.custom_pipeline_rope import CustomWanPipeline
from src.wan2_trainer_rope import WorldWanderTrainSystem

import torch
from diffusers.utils import export_to_video
from diffusers.utils import numpy_to_pil
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler


class WorldWanderInferenceSystem(WorldWanderTrainSystem):
    # custom load ckpt
    def load_state_dict(self, state_dict, strict: bool = True):
        # only load the lora
        self.transformer.load_state_dict(state_dict['lora'], strict=False)

    def on_predict_epoch_start(self):
        self.pred_pipeline = CustomWanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=UniPCMultistepScheduler.from_config(
                FlowMatchEulerDiscreteScheduler.from_pretrained(self.hparams.model_id, subfolder="scheduler").config,
                flow_shift=5,
            ),
        )
        self.pred_path = self.hparams.pred_path
        os.makedirs(self.pred_path, exist_ok=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # data process
        first_pixel_values = batch["first_pixel_values"] # B, C, F, H, W
        third_pixel_values = batch["third_pixel_values"] # B, C, F, H, W
        ref_pixel_values = batch['ref_pixel_values'].unsqueeze(2) # # B, C, 1, H, W
        ori_ref_pixel_values = ref_pixel_values
        prompts = batch["prompts"]
        path = batch["path"]
        # ---------------------------------------------------------------------------------
        if self.hparams.dataset.is_one2three:
            video_gt = third_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
            video_gt = video_gt.permute(0, 2, 3, 1).cpu().numpy()
            # additional
            ref_pixel_values = self.vae.encode(ref_pixel_values).latent_dist.sample()
            ref_pixel_values = (ref_pixel_values - self.latents_mean) / self.latents_std
            # input
            meta = first_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            meta = ((meta + 1) * 0.5).clamp(0, 1)
            meta = meta.permute(0, 2, 3, 1).cpu().numpy()
            #
            first_pixel_values = self.vae.encode(first_pixel_values).latent_dist.sample() # [B, C, F, H, W]
            first_pixel_values = (modefirst_pixel_valuesl_input - self.latents_mean) / self.latents_std # scaling
            attention_kwargs = {
                'encoder_condition_states': first_pixel_values,
                'encoder_ref_states': ref_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
        else:
            video_gt = first_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
            video_gt = video_gt.permute(0, 2, 3, 1).cpu().numpy()
            #
            meta = third_pixel_values.squeeze(0).permute(1, 0, 2, 3)
            meta = ((meta + 1) * 0.5).clamp(0, 1)
            meta = meta.permute(0, 2, 3, 1).cpu().numpy()
            #
            third_pixel_values = self.vae.encode(third_pixel_values).latent_dist.sample() # [B, C, F, H, W]
            third_pixel_values = (third_pixel_values - self.latents_mean) / self.latents_std # scaling
            attention_kwargs = {
                'encoder_condition_states': third_pixel_values,
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
        #
        video_generate = self.pred_pipeline(
            prompt=prompts,
            height=self.hparams.dataset.height,
            width=self.hparams.dataset.width,
            num_frames=self.hparams.dataset.sample_n_frames,
            guidance_scale=5.0,
            attention_kwargs=attention_kwargs,
        )
        video_generate = video_generate.frames[0]    
        #
        concatenated_video = np.concatenate([meta, video_generate, video_gt], axis=1)
        
        # save concat video
        pred_video_path = os.path.join(self.pred_path, f"{path[0]}.mp4")
        export_to_video(concatenated_video, output_video_path=pred_video_path, fps=self.hparams.dataset.fps)

        # save image
        # pred_video_path = os.path.join(self.pred_path, f"{path[0]}")
        # os.makedirs(pred_video_path, exist_ok=True)
        # for i, frame in enumerate(video_generate):
        #     image = numpy_to_pil(frame)[0]
        #     image.save(os.path.join(pred_video_path, f"{i:05d}.png"))

        return

def main(opt):
    L.seed_everything(opt.seed)
    test_dataset = CustomTestDataset(
        first_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video",
        third_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video",
        ref_image_root="/mnt/nfs/workspace/sqj/kkk3",
        #
        height=opt.dataset.height,
        width=opt.dataset.width,
        sample_n_frames=opt.dataset.sample_n_frames,
        stride=1,
        is_one2three=opt.dataset.is_one2three,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=opt.dataset.num_workers,
        drop_last=opt.dataset.drop_last,
        pin_memory=opt.dataset.pin_memory,
        shuffle=False,
    )
    breakpoint()
    system = WorldWanderInferenceSystem.load_from_checkpoint(opt.ckpt_path, opt=opt)
    trainer = L.Trainer(
        logger=False,
        precision=opt.training.precision,
        log_every_n_steps=1,
        accelerator=opt.training.accelerator, # 
        strategy=opt.training.strategy,
        benchmark=opt.training.benchmark,
        num_nodes=opt.num_nodes,
    )
    trainer.predict(system, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan2-2_lora_three2one.yaml", help="path to the yaml config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--pred_path", type=str, default="", help="save path for inference")
    # ----------------------------------------------------------------------
    args, extras = parser.parse_known_args()
    args = vars(args)
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )
    # ----------------------------------------------------------------------
    main(opt)
