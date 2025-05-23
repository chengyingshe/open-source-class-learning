# run_baichuan_7b.sh
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" = "" ]; then
    OUTPUT=./output_step1_baichuan_7b
fi

if [ "$ZERO_STAGE" = "" ]; then
    ZERO_STAGE=3
fi

mkdir -p $OUTPUT

deepspeed --master_port=29501 main.py \
   --data_path data/MyDataset/huatuo_knowledge_graph_qa.json \
   --data_split 10,0,0 \
   --model_name_or_path models/baichuan_7b \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
