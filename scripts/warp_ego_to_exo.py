import json
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse

# -----------------------------------------------------------------------------
# 3D 渲染核心函数 (PyTorch 加速)
# -----------------------------------------------------------------------------
def render_warped_depth(ego_depth, ego_K, ego_E, exo_K, exo_E, height, width, device="cuda"):
    """
    输入:
        ego_depth: (H, W)
        ego_K: (3, 3)
        ego_E: (4, 4) World-to-Camera (VGGT 输出通常是 W2C)
        exo_K: (3, 3)
        exo_E: (4, 4) World-to-Camera
    输出:
        warped_depth: (H, W) Exo 视角下的深度图
    """
    # 1. 准备像素网格 (H, W)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), 
        indexing='ij'
    )
    # 齐次像素坐标 (3, N)
    pixels = torch.stack([x_grid.flatten(), y_grid.flatten(), torch.ones_like(x_grid.flatten())], dim=0).float()
    
    # 过滤无效深度
    depth_flat = ego_depth.flatten()
    valid_mask = depth_flat > 0.1 # 过滤极近噪点
    
    pixels = pixels[:, valid_mask]
    z = depth_flat[valid_mask]
    
    # 2. Unproject: Pixel -> Ego Camera (3D)
    # P_ego = Z * K_inv * p_pixel
    ego_K_inv = torch.inverse(ego_K)
    p_ego = torch.matmul(ego_K_inv, pixels) * z
    
    # 3. Transform: Ego Camera -> World -> Exo Camera
    # P_exo = E_exo * E_ego_inv * P_ego
    # 注意: 如果 E 是 World-to-Camera
    ego_E_inv = torch.inverse(ego_E) # Cam-to-World
    
    # 齐次化 (4, N)
    p_ego_homo = torch.cat([p_ego, torch.ones((1, p_ego.shape[1]), device=device)], dim=0)
    
    # 变换矩阵 T = E_exo @ E_ego_inv
    T_rel = torch.matmul(exo_E, ego_E_inv)
    p_exo_homo = torch.matmul(T_rel, p_ego_homo)
    
    # 4. Project: Exo Camera -> Pixel
    # p_pixel = K_exo * (P_exo / Z_exo)
    x_exo = p_exo_homo[0, :]
    y_exo = p_exo_homo[1, :]
    z_exo = p_exo_homo[2, :]
    
    # 过滤相机背后的点
    valid_z = z_exo > 0.1
    x_exo = x_exo[valid_z]
    y_exo = y_exo[valid_z]
    z_exo = z_exo[valid_z]
    
    # 投影
    fx, fy = exo_K[0, 0], exo_K[1, 1]
    cx, cy = exo_K[0, 2], exo_K[1, 2]
    
    u = (x_exo * fx / z_exo) + cx
    v = (y_exo * fy / z_exo) + cy
    
    u = torch.round(u).long()
    v = torch.round(v).long()
    
    # 过滤出界点
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z_exo[valid_uv]
    
    # 5. Z-Buffer (画家算法: 远的先画，近的覆盖)
    # 按 Z 降序排序
    sort_idx = torch.argsort(z, descending=True)
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    z_sorted = z[sort_idx]
    
    # 写入画布
    canvas = torch.zeros(height * width, device=device, dtype=torch.float32)
    flat_indices = v_sorted * width + u_sorted
    canvas[flat_indices] = z_sorted # 重复索引会被覆盖，最后留下的是最小 Z (近的)
    
    return canvas.reshape(height, width)

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def process_case(case_data, root_dir, output_dir, device):
    # 加载数据
    def load_npy(rel_path):
        return np.load(os.path.join(root_dir, rel_path))

    try:
        ego_depths = torch.from_numpy(load_npy(case_data["ego_depth"])).to(device)
        ego_Ks = torch.from_numpy(load_npy(case_data["ego_intrinsic"])).to(device)
        ego_Es = torch.from_numpy(load_npy(case_data["ego_extrinsic"])).to(device)
        exo_Ks = torch.from_numpy(load_npy(case_data["exo_intrinsic"])).to(device)
        exo_Es = torch.from_numpy(load_npy(case_data["exo_extrinsic"])).to(device)
    except Exception as e:
        print(f"Skipping case due to missing files: {e}")
        return

    num_frames, H, W = ego_depths.shape
    
    # 准备视频输出
    case_name = os.path.basename(os.path.dirname(case_data["ego_depth"]))
    save_path = os.path.join(output_dir, f"{case_name}_warped.mp4")
    out_video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H), isColor=True)
    
    # 逐帧 Warp
    for t in range(num_frames):
        warped_depth = render_warped_depth(
            ego_depths[t], ego_Ks[t], ego_Es[t], 
            exo_Ks[t], exo_Es[t], 
            H, W, device
        )
        
        # 可视化 (归一化 + 伪彩色)
        d_np = warped_depth.cpu().numpy()
        d_vis = np.zeros((H, W, 3), dtype=np.uint8)
        
        mask = d_np > 0
        if mask.any():
            d_valid = d_np[mask]
            # 鲁棒归一化 (2% - 98%)
            v_min, v_max = np.percentile(d_valid, 2), np.percentile(d_valid, 98)
            d_norm = (d_np - v_min) / (v_max - v_min + 1e-5)
            d_norm = np.clip(d_norm, 0, 1)
            d_uint8 = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_INFERNO)
            d_vis[mask] = d_color[mask] # 仅对有效区域上色，背景保持黑
            
        out_video.write(d_vis)
        
    out_video.release()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True, help="Path to index_vggt.json")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--output_dir", type=str, default="warped_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.index_path, 'r') as f:
        data = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 处理前 5 个 Case 看看效果
    count = 0
    for case_id, items in tqdm(data.items()):
        if "ego_depth" in items and "exo_extrinsic" in items:
            process_case(items, args.root_dir, args.output_dir, device)
            count += 1
            if count >= 5: break