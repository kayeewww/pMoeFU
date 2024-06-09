#!/bin/bash

# 定义输出日志文件的路径
LOG_FILE="./log/no_moe$(date +%Y-%m-%d_%H-%M-%S).log"
python Fed_Unlearn_main.py  >> $LOG_FILE 2>&1


# params=("sample" "client")

# for param in "${params[@]}"
# do
#    # 创建日志文件名，包括日期和当前参数值
#    LOG_FILE="./log/${param}600_global_epoch_unlearn_with_tfidf$(date +%Y-%m-%d_%H-%M-%S).log"

#    # 运行Python脚本，并传入当前的参数值，同时重定向输出到日志文件
# #    python Fed_Unlearn_main.py --arg1 "$param" > "$LOG_FILE" &
# #    python Fed_Unlearn_main.py --fats_method "$param" > "$LOG_FILE"
#    python Fed_Unlearn_main.py --global_epoch 600 --local_epoch 10 --fats_method '$param' --rouc 0.2 --rous 0.1 --K 2  --M 600  >> $LOG_FILE 2>&1

# done


# 运行Python脚本并将输出重定向到日志文件
# python Fed_Unlearn_main.py --global_epoch 600 --local_epoch 10 --fats_method 'sample' --rouc 0.2 --rous 0.1 --K 2  --M 600  >> $LOG_FILE 2>&1
#python Fed_Unlearn_main.py --global_epoch 600 --local_epoch 10 --fats_method 'client' --rouc 0.2 --rous 0.1 --K 2  --M 600  >> $LOG_FILE 2>&1


