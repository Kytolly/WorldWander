export PYTHONPATH=$(pwd)

# three2one
CUDA_VISIBLE_DEVICES=3 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_three2one.yaml \
                        --seed=1234 \
                        --ckpt_path "/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-three2one/5e5-rope-copy-no-mask-stride-2-v1/checkpoints/step=6000.ckpt" \
                        --pred_path "pred_results/three2one"


# one2three
# CUDA_VISIBLE_DEVICES=7 python src/wan2_inference_rope.py --config=/mnt/nfs/workspace/sqj/InteractionVideo/configs/wan2-2_game_lora_one2three_high_rank.yaml \
#                         --seed=1234 \
#                         --ckpt_path "/mnt/nfs/workspace/sqj/InteractionVideo/outputs/wan2-one2three-r-128/5e5-no-mask-rope-copy-stride2-v1/checkpoints/step=6000.ckpt" \
#                         --pred_path "pred_samples/teaser_appendix3"
