for task_id in 1 2 3 4 5 7
do
    if [ $task_id == 1 ]
    then
        epoch=2
        len=128
    fi
    if [ $task_id == 2 ]
    then
        epoch=2
        len=128
    fi
    if [ $task_id == 3 ]
    then
        epoch=1
        len=10
    fi
    if [ $task_id == 4 ]
    then
        epoch=2
        len=128
    fi
    if [ $task_id == 5 ]
    then
        epoch=2
        len=128
    fi
    if [ $task_id == 7 ]
    then
        epoch=2
        len=128
    fi
    deepspeed --master_port=29501 main_profile.py \
        --model_path ../FlanT5-XXL/ \
        --emb_model_path ../bge-base-en-v1.5/ \
        --train_file ../LaMP_time_${task_id}_id/train_aug_input.json \
        --dev_file ../LaMP_time_${task_id}_id/dev_profile.json \
        --max_input_len 256 \
        --max_his_len 512 \
        --max_new_len ${len} \
        --output_dir output_${task_id} \
        --optim adamw_torch \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --warmup_ratio 0.05 \
        --num_train_epochs ${epoch} \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --logging_dir ./log/ \
        --logging_steps 10 \
        --evaluation_strategy steps \
        --save_strategy epoch \
        --save_only_model True \
        --eval_steps 0.1 \
        --log_level warning \
        --deepspeed dp.json \
        --report_to none \
        --save_total_limit 1 \
        --bf16 True \
    > output.log 2>&1
done
