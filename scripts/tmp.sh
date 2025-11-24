export PYTHONPATH=$(pwd)


# three2one, synthetic scenarios
CUDA_VISIBLE_DEVICES=1 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_three2one_synthetic.yaml \
                        --seed=1234 \
                        --ckpt_path="/mnt/nfs/workspace/sqj/code/WorldWander/trash/checkpoint.ckpt" \
                        --first_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video" \
                        --third_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video" \
                        --ref_image_root="/mnt/nfs/workspace/sqj/kkk3" \
                        --pred_path="pred_results/three2one_synthetic"


# one2three, synthetic scenarios
# CUDA_VISIBLE_DEVICES=3 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_one2three_synthetic.yaml \
#                         --seed=1234 \
#                         --ckpt_path "/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-one2three-r-128/5e5-no-mask-rope-copy-stride2-v1/checkpoints/step=6500.ckpt" \
#                         --first_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/First_Video" \
#                         --third_video_root="/mnt/nfs/workspace/sqj/EgoExoTestCrop/Third_Video" \
#                         --ref_image_root="/mnt/nfs/workspace/sqj/kkk3" \
#                         --pred_path "pred_results/one2three"

