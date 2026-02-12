export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

mkdir -p logs
# python batched_experiment.py --different-len-delta 0 > logs/batched_experiment.log 2>&1
python batched_experiment.py --max-batch-pow 5 > logs/batched_experiment.log 2>&1
