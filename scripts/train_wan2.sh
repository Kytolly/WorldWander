export PYTHONPATH=$(pwd)

# three2one
CUDA_VISIBLE_DEVICES=2 python src/wan2_trainer_rope.py --config=configs/wan2-2_lora_three2one.yaml --seed=1234


# one2three
CUDA_VISIBLE_DEVICES=2 python src/wan2_trainer_rope.py --config=configs/wan2-2_lora_one2three.yaml --seed=1234
