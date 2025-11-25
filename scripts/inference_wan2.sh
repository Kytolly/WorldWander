export PYTHONPATH=$(pwd)


# three2one, synthetic scenarios
CUDA_VISIBLE_DEVICES=1 python src/wan2_inference.py --config=configs/wan2-2_lora_three2one_synthetic.yaml \
                        --ckpt_path="ckpts/wan2-2_lora_three2one_synthetic.ckpt" \
                        --original_video_root="examples/synthetic_scenarios/third_video" \
                        --pred_path="pred_results/three2one_synthetic" \
                        --seed=1234


# one2three, synthetic scenarios
# CUDA_VISIBLE_DEVICES=3 python src/wan2_inference.py --config=configs/wan2-2_lora_one2three_synthetic.yaml \
#                         --ckpt_path "/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-one2three/5e5-no-mask-rope-copy-stride2-v1/checkpoints/step=5500.ckpt" \
#                         --original_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video" \
#                         --ref_image_root="/mnt/nfs/workspace/sqj/kkk3" \
#                         --pred_path "pred_results/one2three" \
#                         --seed=1234

# three2one, real-world scenarios
# CUDA_VISIBLE_DEVICES=6 python src/wan2_inference.py --config=configs/wan2-2_lora_three2one_realworld.yaml \
#                         --ckpt_path="ckpts/wan2-2_lora_three2one_synthetic.ckpt" \
#                         --original_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video" \
#                         --pred_path="pred_results/tmp" \
#                         --seed=1234

                        