#!/bin/bash
#
# 定义输出日志文件的路径
#LOG_FILE="./log/save/cifar_1_clients_$(date +%Y-%m-%d_%H-%M-%S).log"
#python Fed_Unlearn_main.py --model 'cnn' --data_name 'cifar10' --n_data 100 --n_data_val 200 --n_data_test 200 --frac 5 --N_client 10 --mix_epoch 500 --finetune_epoch 500 --global_epoch 1250 --local_epoch 3 --opt 0 --p 1.0 --local_lr 1e-5 --overlap   >> $LOG_FILE 2>&1


params=(1 2 3 4 5 6 7 8 9)

for param in "${params[@]}"
do
  ARGS="--forget_clients_num $param"
  LOG_FILE="./log/save/cifar10_${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
  python Fed_Unlearn_main.py --model 'cnn' --data_name 'cifar10' --n_data 100 --n_data_val 200 --n_data_test 200 --frac 5 --N_client 10 --mix_epoch 500 --finetune_epoch 500 --global_epoch 1250 --local_epoch 3 --opt 0 --p 1.0 --local_lr 1e-5 --overlap  $ARGS >> "$LOG_FILE" 2>&1
done

## 定义参数列表
#params=(1)
#
##rouc_list_1=(0.3)
##K_list_1=(3)
##M_list_1=(3000)
#
#rouc_list_2=(0.3)
#K_list_2=(2)
#M_list_2=(2000)
#
##params3=(1)
##rouc_list_3=(1)
##K_list_3=(10)
##M_list_3=(3000)
##
##params4=(7)
##rouc_list_4=(0.7)
##K_list_4=(7)
##M_list_4=(3000)
#
##params5=(1 2 3 4 5 6 7 9)
##rouc_list_5=(1)
##K_list_5=(10)
##M_list_5=(3000)
#
## 循环遍历第一组参数
##for i in "${!rouc_list_1[@]}"
##do
##  rouc=${rouc_list_1[i]}
##  K=${K_list_1[i]}
##  M=${M_list_1[i]}
##  for param in "${params[@]}"
##  do
##    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
##    LOG_FILE="./log/sh1/experiment1_rouC${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
##    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
##  done
##done
#
#
### 循环遍历第三组参数
##for i in "${!rouc_list_3[@]}"
##do
##  rouc=${rouc_list_3[i]}
##  K=${K_list_3[i]}
##  M=${M_list_3[i]}
##  for param in "${params3[@]}"
##  do
##    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
##    LOG_FILE="./log/sh/experiment3_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
##    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
##  done
##done
##
### 循环遍历第四组参数
##for i in "${!rouc_list_4[@]}"
##do
##  rouc=${rouc_list_4[i]}
##  K=${K_list_4[i]}
##  M=${M_list_4[i]}
##  for param in "${params4[@]}"
##  do
##    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
##    LOG_FILE="./log/sh1/experiment4_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
##    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
##  done
##done
#
## 循环遍历第五组参数
##for i in "${!rouc_list_5[@]}"
##do
##  rouc=${rouc_list_5[i]}
##  K=${K_list_5[i]}
##  M=${M_list_5[i]}
##  for param in "${params5[@]}"
##  do
##    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
##    LOG_FILE="./log/sh1/experiment5_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
##    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
##  done
##done
#
## 循环遍历第二组参数
#for i in "${!rouc_list_2[@]}"
#do
#  rouc=${rouc_list_2[i]}
#  K=${K_list_2[i]}
#  M=${M_list_2[i]}
#  for param in "${params[@]}"
#  do
#    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
#    LOG_FILE="./log/sh1/_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
#    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
#  done
#done
