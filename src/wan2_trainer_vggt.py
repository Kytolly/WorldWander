import sys
import os
# 确保当前目录在 path 中以便导入模块
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from src.wan2_trainer import WorldWanderTrainSystem, main
from models.wan2.transformer_wan import CustomWanTransformer3DModel
from tools.util import masks_like, resolve_strategy
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from models.wan2.custom_pipeline import CustomWanPipeline # 我们稍后需要继承这个来实现验证

class VGGTTrainSystem(WorldWanderTrainSystem):
    def configure_model(self):
        # 1. 如果已经配置过，直接跳过 (复用父类逻辑中的标志位)
        if self.is_configured:
            return

        # 2. 调用父类初始化，加载 Tokenizer, Text Encoder, VAE (这些不需要改)
        # 注意：我们这里手动执行父类的前半部分逻辑，但拦截 Transformer 的加载
        self.is_configured = True
        
        # --- 复制自父类 (保持不变) ---
        from transformers import AutoTokenizer, UMT5EncoderModel
        from diffusers import AutoencoderKLWan
        from tools.my_schedule import FlowMatchScheduler
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.hparams.model_id, subfolder="text_encoder", torch_dtype=torch.float32)
        self.vae = AutoencoderKLWan.from_pretrained(self.hparams.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler.set_timesteps(1000, training=True)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        
        self.register_buffer('latents_mean', torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1))
        self.register_buffer('latents_std', torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1))
        # ---------------------------

        # 3. [关键修改] 加载 Transformer 并扩充通道
        print("\n[VGGT] Initializing Transformer with 32 input channels (16 Noisy + 16 Warped)...")
        
        # A. 加载配置并修改
        config = CustomWanTransformer3DModel.load_config(self.hparams.model_id, subfolder="transformer")
        config['in_channels'] = 32 # 核心修改
        
        # B. 用新配置初始化模型 (此时权重是随机的)
        self.transformer = CustomWanTransformer3DModel.from_config(config)
        
        # C. 加载预训练权重并处理维度不匹配
        # 先加载原始的 16 通道模型权重到内存
        print("[VGGT] Loading pretrained weights and applying Zero-Init...")
        original_model = CustomWanTransformer3DModel.from_pretrained(
            self.hparams.model_id, subfolder="transformer", torch_dtype=torch.float32
        )
        original_state_dict = original_model.state_dict()
        
        # 获取新模型的 state_dict
        new_state_dict = self.transformer.state_dict()
        
        # 遍历权重进行迁移
        for key in new_state_dict.keys():
            # 处理输入层卷积
            if key == 'patch_embedding.weight':
                # original shape: [dim, 16, t, h, w]
                # new shape:      [dim, 32, t, h, w]
                old_weight = original_state_dict[key]
                
                # 复制前16通道，后16通道置零
                new_state_dict[key][:, :16, ...] = old_weight
                new_state_dict[key][:, 16:, ...] = 0.0 # Zero-Init
                
            # 处理 Bias (通常不用改，但直接复制)
            elif key == 'patch_embedding.bias':
                new_state_dict[key] = original_state_dict[key]
                
            # 其他层直接复制
            elif key in original_state_dict:
                new_state_dict[key] = original_state_dict[key]
            else:
                print(f"[Warning] Key {key} not found in pretrained model (expected for LoRA/New layers)")

        # 将处理好的权重加载进新模型
        self.transformer.load_state_dict(new_state_dict)
        
        # 释放内存
        del original_model
        torch.cuda.empty_cache()

        # 4. 后续设置 (LoRA, Gradient Checkpointing) - 保持与父类一致
        if self.hparams.use_lora:
            self.transformer.requires_grad_(False)
            
        if self.hparams.training.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if self.hparams.use_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=self.hparams.training.rank,
                lora_alpha=self.hparams.training.rank,
                init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
            )
            self.transformer.add_adapter(transformer_lora_config)
            
        print("[VGGT] Model configuration complete.\n")
        
    def process_data(self, batch, batch_idx):
        # 1. 调用父类处理 Ego, Exo, Ref, Text
        # 注意：这里假设父类方法返回的是 tuple。
        # 如果父类 process_data 比较复杂不好拆解，建议直接复制父类代码过来修改
        
        # 为了稳妥，这里完整复现 process_data 逻辑并加入 warped 处理
        first_pixel_values = batch["first_pixel_values"] # [B, C, F, H, W]
        third_pixel_values = batch["third_pixel_values"] # [B, C, F, H, W]
        
        # [VGGT 新增] 获取 Warped Video
        warped_pixel_values = batch.get("warped_pixel_values") # [B, C, F, H, W]
        if warped_pixel_values is None:
            raise ValueError("Dataset must provide 'warped_pixel_values' for VGGT training.")

        ref_pixel_values = batch['ref_pixel_values'].unsqueeze(2) if self.hparams.dataset.is_one2three else None
        prompts = batch["prompts"]
        
        # Drop Text Logic
        if self.hparams.use_drop_text:
            import random
            prompts = [prompt if random.random() > 0.1 else '' for prompt in prompts]

        # VAE Encoding (with gradient disabled context just in case, though VAE is frozen)
        with torch.no_grad():
            # First (Ego)
            first_latents = self.vae.encode(first_pixel_values).latent_dist.sample()
            first_latents = (first_latents - self.latents_mean) / self.latents_std
            
            # Third (Exo/Target)
            third_latents = self.vae.encode(third_pixel_values).latent_dist.sample()
            third_latents = (third_latents - self.latents_mean) / self.latents_std
            
            # [VGGT 新增] Warped
            warped_latents = self.vae.encode(warped_pixel_values).latent_dist.sample()
            warped_latents = (warped_latents - self.latents_mean) / self.latents_std
            
            # Ref
            if self.hparams.dataset.is_one2three:
                ref_latents = self.vae.encode(ref_pixel_values).latent_dist.sample()
                ref_latents = (ref_latents - self.latents_mean) / self.latents_std
            else:
                ref_latents = None

        # Text Encoding
        prompt_embeds = self.encode_prompt(prompts)

        return first_latents, third_latents, ref_latents, prompt_embeds, warped_latents
    
    def training_step(self, batch, batch_idx):
        # 解包数据
        first_latents, third_latents, ref_latents, prompt_embeds, warped_latents = self.process_data(batch, batch_idx)        
        
        batch_size = third_latents.shape[0]
        noise = torch.randn_like(third_latents)
        timestep_id = torch.randint(0, self.train_scheduler.num_train_timesteps, (batch_size,))
        timestep = self.train_scheduler.timesteps[timestep_id].to(dtype=third_latents.dtype, device=self.device)
        
        # 1. 加噪
        latent_noisy = self.train_scheduler.add_noise(third_latents, noise, timestep)
        
        # 2. [VGGT 核心] 拼接输入
        # latent_noisy: [B, 16, F, H, W]
        # warped_latents: [B, 16, F, H, W]
        # input: [B, 32, F, H, W]
        transformer_input = torch.cat([latent_noisy, warped_latents], dim=1)
        
        # 3. 准备 Condition (Prompt + Ego Video)
        _, mask2 = masks_like(noise, zero=False)
        
        # 构造 Attention Kwargs
        if self.hparams.dataset.is_one2three:
            attention_kwargs = {
                'encoder_condition_states': first_latents, # Ego Video
                'encoder_ref_states': ref_latents,         # Ref Image
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
            # One2Three 的 mask 逻辑
            timestep_all = (timestep.view(batch_size, 1, 1, 1) * mask2[0][:, 0, :, ::2, ::2]).flatten(1)
            timestep_all = torch.concat([torch.zeros_like(mask2[0][:, 0, 0, ::2, ::2].flatten(1)), torch.zeros_like(timestep_all), timestep_all], dim=-1)
        else:
            attention_kwargs = {
                'encoder_condition_states': third_latents, # 注意：原代码这里可能是写错了或者是 Three2One 的特殊逻辑
                # 但对于 Ego2Exo 任务，Condition 应该是 First (Ego)
                # 假设你是在做 Ego -> Exo，这里的 condition 应该是 first_latents
                'encoder_condition_states': first_latents, 
                'use_collaborative_position_encoding': self.hparams.use_collaborative_position_encoding,
            }
            timestep_all = (timestep.view(batch_size, 1, 1, 1) * mask2[0][:, 0, :, ::2, ::2]).flatten(1)
            timestep_all = torch.concat([torch.zeros_like(timestep_all), timestep_all], dim=-1)

        # 4. Forward
        v_target = self.train_scheduler.training_target(third_latents, noise, timestep)
        
        v_pred = self.transformer(
            hidden_states=transformer_input, # 传入 32 通道
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_all,
            return_dict=False,
            attention_kwargs=attention_kwargs,
        )[0]
        
        # 5. Loss
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float(), reduction='none')
        weight = self.train_scheduler.training_weight(timestep).to(loss.device)
        loss = (loss * weight[:, None, None, None, None]).mean()
        
        # Log
        self.log("train/loss_vggt", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, on_step=True, logger=True, sync_dist=True)

        return loss
    
    def on_validation_epoch_start(self):
        # 暂时只打印日志，不初始化 Pipeline，因为标准 Pipeline 无法处理 32 通道输入
        self.print(f"Validation started at step {self.global_step}. Note: Visual generation is customized for VGGT.")
        self.val_path = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'val_samples')
        os.makedirs(self.val_path, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        # 在训练初期，我们可能只需要计算 Validation Loss 即可
        # 如果需要生成视频，必须手写采样循环 (Sampling Loop)
        # 这里为了简化，我们先计算 Loss，确保模型没跑偏
        
        loss = self.training_step(batch, batch_idx) # 复用 training_step 算 loss
        self.log("val/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        
        # 如果你非常需要看到生成的视频，你需要实现一个 custom sampling loop
        # 这涉及 scheduler.step 和 iterative denoising，代码量较大
        # 建议先跑通训练，确认 Loss 下降
        return loss
    
if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wan2-2_lora_one2three_realworld.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_path", type=str, default="")
    
    args, extras = parser.parse_known_args()
    args = vars(args)
    
    # 动态获取显卡数量
    num_gpus = torch.cuda.device_count()
    
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": num_gpus}),
    )
    
    # 替换原本的 main 逻辑
    from src.wan2_trainer import CustomTrainDataset, DataLoader, WandbLogger, CustomProgressBar, CustomModelCheckpoint, L, resolve_strategy
    
    # 1. 实例化数据集 (你的 Dataset 已经能自动处理 Warped 路径了)
    train_dataset = CustomTrainDataset(
        json_index_path=opt.dataset.json_index_path,
        followbench_root=opt.dataset.followbench_root,
        warped_video_root=opt.dataset.warped_video_root,
        height=opt.dataset.height,
        width=opt.dataset.width,
        resize_long=opt.dataset.resize_long,
        sample_n_frames=opt.dataset.sample_n_frames,
        stride=opt.dataset.stride,
        is_one2three=opt.dataset.is_one2three
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.training.batch_size,
        num_workers=opt.dataset.num_workers,
        shuffle=True,
        drop_last=True
    )
    
    # 验证集 (简单复用训练集的前几个样本)
    val_dataloader = DataLoader(
        train_dataset, 
        batch_size=1, 
        shuffle=False 
    )

    # 2. 实例化 VGGT System
    system = VGGTTrainSystem(opt)

    # 3. Logger & Callbacks
    logger = WandbLogger(
        project=opt.experiment_project, 
        name=opt.experiment_name, 
        save_dir=opt.output_root
    )
    
    checkpoint_callback = CustomModelCheckpoint(
        dirpath=os.path.join(opt.output_root, opt.experiment_name, 'checkpoints'),
        filename="{step}",
        every_n_train_steps=opt.training.save_val_interval_steps,
        save_top_k=-1
    )

    # 4. Trainer
    trainer = L.Trainer(
        logger=logger,
        max_steps=opt.training.max_steps,
        accelerator=opt.training.accelerator,
        strategy=resolve_strategy(opt.training.strategy),
        callbacks=[CustomProgressBar(), checkpoint_callback],
        num_nodes=opt.num_nodes,
        devices=num_gpus,
        precision="16-mixed" # 建议开启混合精度
    )

    # 5. Fit
    trainer.fit(system, train_dataloader, val_dataloader)