import json
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
from torchvision import transforms as TF

# 引入库
try:
    from decord import VideoReader, cpu
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError:
    print("错误: 请确保安装了 decord 和 vggt")
    exit(1)

def preprocess_batch_joint_pad(ego_rgb_np, exo_rgb_np, target_bs, target_size=518):
    """
    联合预处理，并强制 Pad 到固定的 Batch Size 以利用 H20 的 torch.compile
    """
    to_tensor = TF.ToTensor()
    
    def _process(rgb_list):
        imgs = []
        for i in range(len(rgb_list)):
            img = Image.fromarray(rgb_list[i]) # 已经是 resize 后的
            t = to_tensor(img)
            
            # Center Crop
            c_h, c_w = t.shape[1], t.shape[2]
            if c_h > target_size:
                sy = (c_h - target_size) // 2
                t = t[:, sy : sy + target_size, :]
            if c_w > target_size:
                sx = (c_w - target_size) // 2
                t = t[:, :, sx : sx + target_size]
            imgs.append(t)
        return torch.stack(imgs)

    ego_real = _process(ego_rgb_np) # (B_real, 3, H, W)
    exo_real = _process(exo_rgb_np) # (B_real, 3, H, W)
    
    current_bs = len(ego_real)
    
    # 拼接
    joint_real = torch.cat([ego_real, exo_real], dim=0) # (2*B_real, ...)
    
    # Pad 逻辑
    if current_bs < target_bs:
        pad_len = target_bs - current_bs
        # 创建全 0 的 Pad Tensor (2*pad_len, ...)
        # 注意：需要同时 Pad Ego 和 Exo 部分，保持结构一致性以便切片
        # 实际上 VGGT 内部不区分 Ego/Exo，我们只要保证总长度对齐
        # 简单策略：直接在 joint_real 后面补零
        pad_tensor = torch.zeros((2 * pad_len, 3, target_size, target_size), dtype=joint_real.dtype)
        joint_padded = torch.cat([joint_real, pad_tensor], dim=0)
    else:
        joint_padded = joint_real

    return joint_padded, current_bs

def extract_features_turbo(model, ego_path, exo_path, output_prefix_ego, output_prefix_exo, device, batch_size=64, step=2):
    # 检查完成
    if os.path.exists(f"{output_prefix_ego}_depth.npy") and os.path.exists(f"{output_prefix_exo}_extrinsic.npy"):
        return True

    try:
        # 1. Peek 尺寸
        vr_peek = VideoReader(ego_path, ctx=cpu(0))
        h_orig, w_orig, _ = vr_peek[0].shape
        len_ego = len(vr_peek)
        del vr_peek
        
        vr_peek_exo = VideoReader(exo_path, ctx=cpu(0))
        len_exo = len(vr_peek_exo)
        del vr_peek_exo
        
        min_len = min(len_ego, len_exo)
        if min_len < 10: return None

        # 2. Resize 参
        target_s = 518
        if w_orig >= h_orig:
            nh, nw = target_s, int(round(w_orig * (target_s / h_orig) / 14) * 14)
        else:
            nw, nh = target_s, int(round(h_orig * (target_s / w_orig) / 14) * 14)

        # 3. Re-open
        vr_ego = VideoReader(ego_path, ctx=cpu(0), width=nw, height=nh)
        vr_exo = VideoReader(exo_path, ctx=cpu(0), width=nw, height=nh)

    except Exception as e:
        print(f"[Err] Init failed {ego_path}: {e}")
        return None

    results = {k: [] for k in ["ego_depth", "ego_K", "ego_E", "exo_depth", "exo_K", "exo_E"]}
    dtype = torch.float16
    
    # 4. 循环处理 (带 Stride)
    # 假设 min_len=600, step=2 -> 处理 0, 2, 4... 共 300 帧
    # total_steps = 300
    indices_all = list(range(0, min_len, step))
    total_processing_frames = len(indices_all)
    
    try:
        for i in range(0, total_processing_frames, batch_size):
            # 获取当前 Batch 的帧索引
            batch_indices = indices_all[i : min(i + batch_size, total_processing_frames)]
            
            # 读取
            batch_ego_np = vr_ego.get_batch(batch_indices).asnumpy()
            batch_exo_np = vr_exo.get_batch(batch_indices).asnumpy()
            
            # 预处理 + Padding
            # images shape 固定为 (1, 2*batch_size, 3, 518, 518)
            images, real_bs = preprocess_batch_joint_pad(batch_ego_np, batch_exo_np, batch_size, target_s)
            images = images.to(device).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                agg_tokens, ps_idx = model.aggregator(images)
                depth, _ = model.depth_head(agg_tokens, images, ps_idx)
                pose_enc = model.camera_head(agg_tokens)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                
                # 拆分并去除 Padding
                # 输出 shape: (1, 2*batch_size, ...)
                # 有效数据: 前 real_bs 是 Ego, 随后 real_bs 是 Exo
                # 注意 Padding 是加在最后的，所以 joint 结构是 [Ego_Real, Exo_Real, Pad_Zeros]
                # 抱歉，preprocess_batch_joint_pad 里的逻辑是 [Ego_Real, Exo_Real, Pad]
                # 所以我们直接取前 2*real_bs 即可? 
                # 不对，preprocess 里的 cat 逻辑是 cat([ego, exo], dim=0) -> cat([joint, pad])
                # 所以前 real_bs 是 Ego, 第 [real_bs : 2*real_bs] 是 Exo。
                
                depth_np = depth.squeeze().float().cpu().numpy()
                ext_np = extrinsic.squeeze().float().cpu().numpy()
                int_np = intrinsic.squeeze().float().cpu().numpy()
                
                # 保存有效数据
                results["ego_depth"].append(depth_np[:real_bs])
                results["exo_depth"].append(depth_np[real_bs : 2*real_bs])
                
                results["ego_E"].append(ext_np[:real_bs])
                results["exo_E"].append(ext_np[real_bs : 2*real_bs])
                
                results["ego_K"].append(int_np[:real_bs])
                results["exo_K"].append(int_np[real_bs : 2*real_bs])

    except Exception as e:
        print(f"[Err] Inference {ego_path}: {e}")
        return None

    # 合并保存
    for k in results:
        results[k] = np.concatenate(results[k], axis=0)

    np.save(f"{output_prefix_ego}_depth.npy", results["ego_depth"])
    np.save(f"{output_prefix_ego}_intrinsic.npy", results["ego_K"])
    np.save(f"{output_prefix_ego}_extrinsic.npy", results["ego_E"])
    
    np.save(f"{output_prefix_exo}_depth.npy", results["exo_depth"])
    np.save(f"{output_prefix_exo}_intrinsic.npy", results["exo_K"])
    np.save(f"{output_prefix_exo}_extrinsic.npy", results["exo_E"])

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--output_index", type=str, required=True)
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=64)
    # 新增 step 参数
    parser.add_argument("--step", type=int, default=2, help="Frame stride (1=60fps, 2=30fps, 3=20fps)")
    args = parser.parse_args()

    with open(args.index_path, 'r') as f:
        data = json.load(f)

    device = "cuda"
    print(f"Loading VGGT on {device}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    
    # 关键加速：H20 + Compile
    # mode="reduce-overhead" 配合 Fixed Batch Size 是最快的
    try:
        print("Compiling model (wait ~2 mins)...")
        model = torch.compile(model, mode="reduce-overhead")
    except: pass

    print(f"Turbo Processing {len(data)} cases (Step={args.step})...")
    count = 0
    
    for case_id, items in tqdm(data.items()):
        if "the first view" not in items: continue
        
        ego_p = os.path.join(args.root_dir, items["the first view"])
        exo_p = os.path.join(args.root_dir, items["the third view"])
        
        p_ego = os.path.splitext(ego_p)[0]
        p_exo = os.path.splitext(exo_p)[0]
        
        success = extract_features_turbo(
            model, ego_p, exo_p, p_ego, p_exo, 
            device, args.batch_size, args.step
        )
        
        if success:
            items["ego_depth"] = os.path.relpath(f"{p_ego}_depth.npy", args.root_dir)
            items["ego_intrinsic"] = os.path.relpath(f"{p_ego}_intrinsic.npy", args.root_dir)
            items["ego_extrinsic"] = os.path.relpath(f"{p_ego}_extrinsic.npy", args.root_dir)
            items["exo_depth"] = os.path.relpath(f"{p_exo}_depth.npy", args.root_dir)
            items["exo_intrinsic"] = os.path.relpath(f"{p_exo}_intrinsic.npy", args.root_dir)
            items["exo_extrinsic"] = os.path.relpath(f"{p_exo}_extrinsic.npy", args.root_dir)
            items["frame_stride"] = args.step # 记录步长，方便后续训练对齐
        
        count += 1
        if count % 20 == 0:
            with open(args.output_index, 'w') as f:
                json.dump(data, f, indent=4)

    with open(args.output_index, 'w') as f:
        json.dump(data, f, indent=4)
    print("All Done.")

if __name__ == "__main__":
    '''
    python scripts/preprocess_vggt_joint.py \
      --batch_size 64 \
      --step 3 \
      --output_index /opt/liblibai-models/user-workspace2/users/xqy/project/WorldWander/FollowBench/train/index_vggt.json \
      --index_path /opt/liblibai-models/user-workspace2/users/xqy/project/WorldWander/FollowBench/train/index_clean.json \
      --root_dir /opt/liblibai-models/user-workspace2/users/xqy/project/WorldWander/FollowBench/train
    '''
    main()