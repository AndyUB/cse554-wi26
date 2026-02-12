export CUDA_VISIBLE_DEVICES=6
export PYTHONUNBUFFERED=1

mkdir -p logs
python single_batch_experiment.py > logs/single_batch.log 2>&1