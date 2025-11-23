export PYTHONPATH=$(pwd)


# three2one, synthetic scenarios
# CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer_rope.py --config=configs/wan2-2_lora_three2one_synthetic.yaml --seed=1234


# one2three, synthetic scenarios
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/wan2_trainer_rope.py --config=configs/wan2-2_lora_one2three_synthetic.yaml --seed=1234
