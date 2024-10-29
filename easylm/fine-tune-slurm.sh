#!/bin/bash

#SBATCH --job-name=llama-jax-fine-tune
#SBATCH --nodes=1
#SBATCH --partition=n1t4
#SBATCH --time=1:10:00

export CONDA_BASE=/opt/conda
source $CONDA_BASE/bin/activate base
conda activate llama_jax
cd /data_bucket
tar zxvf easylm.tgz
cd $SLURM_SUBMIT_DIR
export PYTHONPATH=/data_bucket:$PYTHONPATH
python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,64,1' \
    --dtype='fp32' \
    --total_steps=25000 \
    --log_freq=50 \
    --save_model_freq=0 \
    --save_milestone_freq=250 \
    --llama.base_model='llama_7b' \
    --load_dataset_state='' \
    --load_checkpoint='' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=20 \
    --optimizer.adamw_optimizer.lr_decay_steps=250 \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='ivanzhouyq/RedPajama-Tiny' \
    --train_dataset.huggingface_dataset.name='default' \
    --train_dataset.huggingface_dataset.seq_length=20 \
    --train_dataset.huggingface_dataset.batch_size=10 \
    --train_dataset.huggingface_dataset.streaming=True \
    --checkpointer.save_optimizer_state=True \
    --logger.online=False \
    --logger.prefix='easylm' \
    --logger.project="llama_jax_7b" \
    --logger.output_dir="/data_bucket/output" 