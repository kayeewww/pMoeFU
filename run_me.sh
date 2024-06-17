#!/bin/bash
#
# 定义输出日志文件的路径
#LOG_FILE="./log/cifar10_600G_2K_dif_M/9clients_$(date +%Y-%m-%d_%H-%M-%S).log"
#python Fed_Unlearn_main.py  >> $LOG_FILE 2>&1


#params=(9)
#
#for param in "${params[@]}"
#do
#  ARGS="--forget_clients_num $param"
#  LOG_FILE="./log/cifar10_600G_3000M_dif_K/ii${param}_clients_K5_$(date +%Y-%m-%d_%H-%M-%S).log"
#  python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
#done

# 定义参数列表
params=(1 2 3 4 5 6 7 8 9)

rouc_list_1=(0.2 0.3)
K_list_1=(2 3)
M_list_1=(3000 3000)

rouc_list_2=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
K_list_2=(2 2 2 2 2 2 2 2)
M_list_2=(2000 1500 1200 1000 857 750 66 600)

params3=(1)
rouc_list_3=(0.5 0.6 0.7 0.8 0.9 1)
K_list_3=(5 6 7 8 9 10)
M_list_3=(3000 3000 3000 3000 3000)

params4=(7)
rouc_list_4=(0.7)
K_list_4=(7)
M_list_4=(3000)

params5=(1 2 3 4 5 6 7 8 9)
rouc_list_5=(0.8 0.9 1)
K_list_5=(8 9 10)
M_list_5=(3000)

# 循环遍历第一组参数
for i in "${!rouc_list_1[@]}"
do
  rouc=${rouc_list_1[i]}
  K=${K_list_1[i]}
  M=${M_list_1[i]}
  for param in "${params[@]}"
  do
    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
    LOG_FILE="./log/sh1/experiment1_rouC${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
  done
done


# 循环遍历第三组参数
for i in "${!rouc_list_3[@]}"
do
  rouc=${rouc_list_3[i]}
  K=${K_list_3[i]}
  M=${M_list_3[i]}
  for param in "${params3[@]}"
  do
    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
    LOG_FILE="./log/sh/experiment3_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
  done
done

# 循环遍历第四组参数
for i in "${!rouc_list_4[@]}"
do
  rouc=${rouc_list_4[i]}
  K=${K_list_4[i]}
  M=${M_list_4[i]}
  for param in "${params4[@]}"
  do
    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
    LOG_FILE="./log/sh/experiment4_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
  done
done

# 循环遍历第五组参数
for i in "${!rouc_list_5[@]}"
do
  rouc=${rouc_list_5[i]}
  K=${K_list_5[i]}
  M=${M_list_5[i]}
  for param in "${params5[@]}"
  do
    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
    LOG_FILE="./log/sh/experiment3_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
  done
done

# 循环遍历第二组参数
for i in "${!rouc_list_2[@]}"
do
  rouc=${rouc_list_2[i]}
  K=${K_list_2[i]}
  M=${M_list_2[i]}
  for param in "${params[@]}"
  do
    ARGS="--rouc $rouc --K $K --M $M --forget_clients_num $param"
    LOG_FILE="./log/sh/experiment2_rouc${rouc}_K${K}_M${M}_forget${param}clients_$(date +%Y-%m-%d_%H-%M-%S).log"
    python Fed_Unlearn_main.py $ARGS >> "$LOG_FILE" 2>&1
  done
done
