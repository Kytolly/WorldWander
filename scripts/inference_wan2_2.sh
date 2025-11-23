export PYTHONPATH=$(pwd)

# three2one
# CUDA_VISIBLE_DEVICES=2 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_three2one.yaml \
#                         --seed=1234 \
#                         --ckpt_path="/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-three2one-r-128/5e5-rope-copy-no-mask-stride-2-v1/checkpoints/step=7000.ckpt" \
#                         --first_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video" \
#                         --third_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video" \
#                         --ref_image_root="/mnt/nfs/workspace/sqj/kkk3" \
#                         --pred_path="pred_results/three2one"


# one2three
CUDA_VISIBLE_DEVICES=3 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_three2one.yaml \
                        --seed=1234 \
                        --ckpt_path "/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-one2three-r-128/5e5-no-mask-rope-copy-stride2-v1/checkpoints/step=6500.ckpt" \
                        --first_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video" \
                        --third_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video" \
                        --ref_image_root="/mnt/nfs/workspace/sqj/kkk3" \
                        --pred_path "pred_results/one2three"
