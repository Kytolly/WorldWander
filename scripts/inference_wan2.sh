export PYTHONPATH=$(pwd)

# three2one
CUDA_VISIBLE_DEVICES=3 python src/wan2_inference_rope.py --config=configs/wan2-2_lora_three2one.yaml \
                        --seed=1234 \
                        --ckpt_path "your/ckpt/path" \
                        --pred_path "pred_results/three2one"


# one2three
CUDA_VISIBLE_DEVICES=3 python src/wan2_inference_rope.py --config=/mnt/nfs/workspace/sqj/InteractionVideo/configs/wan2-2_game_lora_one2three_high_rank.yaml \
                        --seed=1234 \
                        --ckpt_path "your/ckpt/path" \
                        --pred_path "pred_results/one2three"
