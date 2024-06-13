#!/bin/bash

# 定义输出日志文件的路径
LOG_FILE="./log/cifar10_600G_2K_dif_M/4clients_$(date +%Y-%m-%d_%H-%M-%S).log"
python Fed_Unlearn_main.py  >> $LOG_FILE 2>&1


#params=( '3' '4' '5' '6' '7' '8' '9' '10')
#
#for param in "${params[@]}"
#do
#
#  LOG_FILE="./log/cifar10_600G_2K_dif_M/4clients_$(date +%Y-%m-%d_%H-%M-%S).log"
#
#  python Fed_Unlearn_main.py --global_epoch 600 --local_epoch 2 --forget_clients_num 4 --rouc 0.2 --K 2  --M 3000  >> "$LOG_FILE"2>&1
#
#done