import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# ourself libs
from model_initiation import model_init
from data_preprocess import data_init
from FL_base import test
import Fed_Retrain
from membership_inference import train_attack_model, attack

from Fed_Unlearn_base import federated_learning_unlearning

# from membership_inference import train_attack_model, attack

"""Step 0. Initialize Federated Unlearning parameters"""


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 25
        self.data_name = 'mnist'  # cifar10, cifar100
        self.model_name = 'resnet56'  # 44 resnet20, resnet32, resnet44, resnet56, vgg11, vgg13, vgg16, vgg19
        self.global_epoch = 1
        self.local_epoch = 10

        self.save_acc = 80  ###
        self.save_re_acc = 75  ###

        # Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.005
        self.test_batch_size = 64
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = True
        self.model_file = 'seed_acc80.06_epoch125_2024-03-02 15-08-23.pth'

        # 要剪枝的类别。
        # TODO tfidf mask来替换unlearn_class
        self.unlearn_class = 2  # If you want to forget, change None to the client index
        self.sparsity = 0.05

        self.if_retrain = False
        # 如果设置为 False，表示在遗忘操作后执行重新训练。在重新训练期间，与遗忘的客户端相关的数据将被丢弃
        # 如果设置为 "True"，全局模型将使用 FL-Retrain 函数重新训练，并丢弃 forget_client_IDx 编号对应的用户数据。

        self.if_unlearning = False
        # 如果设置为 False，global_train_once 函数不会跳过需要遗忘的用户；如果设置为 True，global_train_once 会在训练过程中跳过被遗忘的用户。

        self.forget_local_epoch_ratio = 0.5
        # self.mia_oldGM = False
        # 当一个用户被选中遗忘时，其他用户需要在各自的数据集中进行多轮在线训练，以获得模型收敛的大方向，从而提供模型收敛的大方向。
        # forget_local_epoch_ratio*local_epoch 是我们需要获得各局部模型收敛方向时的局部训练轮数
        self.forget_client_idx=None


def Federated_Unlearning():
    """Step 1.Set the parameters for Federated Unlearning"""
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)
    # kwargs for data loader
    print(60 * '=')
    print("Step1. Federated Learning Settings \n We use dataset: " + FL_params.data_name + (
        " for our Federated Unlearning experiment.\n"))

    """Step 2. construct the necessary user private data set required for federated learning, as well as a common test set"""
    print(60 * '=')
    print("Step2. Client data loaded, testing data loaded!!!\n       Initial Model loaded!!!")
    # 加载数据
    init_global_model = model_init(FL_params.data_name, FL_params.model_name)
    client_all_loaders, test_loader = data_init(FL_params)

    print(init_global_model)
    # print(client_all_loaders)

    # 从100个客户端中随机抽取25个作为实验对象
    selected_clients = np.random.choice(range(FL_params.N_total_client), size=FL_params.N_client, replace=False)
    client_loaders = list()
    for idx in selected_clients:
        client_loaders.append(client_all_loaders[idx])

    """
    This section of the code gets the initialization model init Global Model
    User data loader for FL training Client_loaders and test data loader Test_loader
    User data loader for covert FL training, Shadow_client_loaders, and test data loader Shadow_test_loader
    """

    """Step 3. Select a client's data to forget，1.Federated Learning, 2.Unlearning(FedEraser), and 3.(Accumulating)Unlearing without calibration"""
    print(60 * '=')
    print("Step3. Fedearated Learning and Unlearning Training...")
    # FedAvg, FedEraser, FedAccum,
    old_GMs, unlearn_GMs, uncali_unlearn_GMs, _, _ = federated_learning_unlearning(init_global_model, client_loaders,
                                                                                   test_loader, FL_params)
    # print(old_GMs, unlearn_GMs, uncali_unlearn_GMs) #都是torch.size
    if (FL_params.if_retrain == True):

        t1 = time.time()
        # FedRetrain
        retrain_GMs = Fed_Retrain(init_global_model, client_loaders, test_loader, FL_params)
        params_diff = np.abs(np.array(unlearn_GMs[-1])) - np.abs(np.array(retrain_GMs[-1]))
        histogram, bins = np.histogram(params_diff, bins=10)
        plt.bar(bins[:-1], histogram, width=np.diff(bins), align='edge')
        plt.xlabel('Deviation')
        plt.ylabel('Frequency')
        t2 = time.time()
        print("Time using = {} seconds".format(round(t2 - t1), 3))

        # Evaluation
    fedavg_test_loss, fedavg_test_acc = test(old_GMs[-1], test_loader)
    federaser_test_loss, federaser_test_acc = test(unlearn_GMs[-1], test_loader)
    fedaccum_test_loss, fedaccum_test_acc = test(uncali_unlearn_GMs[-1], test_loader)
    fedretrain_test_loss, fedretrain_test_acc = test(retrain_GMs[-1], test_loader)

    target_loader = client_loaders[FL_params.forget_client_idx]
    fedavg_target_loss, fedavg_target_acc = test(old_GMs[-1], target_loader)
    federaser_target_loss, federaser_target_acc = test(unlearn_GMs[-1], target_loader)
    fedaccum_target_loss, fedaccum_target_acc = test(uncali_unlearn_GMs[-1], target_loader)
    fedretrain_target_loss, fedretrain_target_acc = test(retrain_GMs[-1], target_loader)

    print(5 * "*" + "  Result Summary  " + 5 * "*")
    print("[FedEraser] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_test_loss,
                                                                                     federaser_test_acc))
    print("[FedAccum] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_test_loss,
                                                                                    fedaccum_test_acc))
    print("[FedRetrain] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_test_loss,
                                                                                      fedretrain_test_acc))
    print("[FedAvg] Test set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_test_loss, fedavg_test_acc))
    print("\n")
    print("[FedEraser] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(federaser_target_loss,
                                                                                       federaser_target_acc))
    print("[FedAccum] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedaccum_target_loss,
                                                                                      fedaccum_target_acc))
    print("[FedRetrain] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedretrain_target_loss,
                                                                                        fedretrain_target_acc))
    print("[FedAvg] Target set: Average loss = {:.8f}, Average acc = {:.4f}".format(fedavg_target_loss,
                                                                                    fedavg_target_acc))

    """Step 4  The member inference attack model is built based on the output of the Target Global Model on client_loaders and test_loaders.In this case, we only do the MIA attack on the model at the end of the training"""

    """MIA:Based on the output of oldGM model, MIA attack model was built, and then the attack model was used to attack unlearn GM. If the attack accuracy significantly decreased, it indicated that our unlearn method was indeed effective to remove the user's information"""
    print(60 * '=')
    print("Step4. Membership Inference Attack aganist GM...")

    T_epoch = -1
    # MIA setting:Target model == Shadow Model
    old_GM = old_GMs[T_epoch]
    attack_model = train_attack_model(old_GM, client_loaders, test_loader, FL_params)

    print("\nEpoch  = {}".format(T_epoch))
    print("Attacking against FL Standard  ")
    target_model = old_GMs[T_epoch]
    (PRE_old, REC_old, F1_old) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    if (FL_params.if_retrain == True):
        print("Attacking against FL Retrain  ")
        target_model = retrain_GMs[T_epoch]
        (PRE_retrain, REC_retrain, F1_retrain) = attack(target_model, attack_model, client_loaders, test_loader,
                                                        FL_params)

    print("Attacking against FL Unlearn  ")
    target_model = unlearn_GMs[T_epoch]
    (PRE_unlearn, REC_unlearn, F1_unlearn) = attack(target_model, attack_model, client_loaders, test_loader, FL_params)

    print("Attacking against FL Unlearn without calibration  ")
    target_model = uncali_unlearn_GMs[T_epoch]
    (PRE_uncali_unlearn, REC_uncali_unlearn, F1_uncali_unlearn) = attack(target_model, attack_model, client_loaders,
                                                                         test_loader, FL_params)


if __name__ == '__main__':
    Federated_Unlearning()
