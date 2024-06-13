import torch
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
#ourself libs
import total_variance as total_variance
from data_preprocess import data_set, model_init
from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain, test
from class_pruner import acculumate_feature, calculate_cp, get_threshold_by_sparsity, select_least_important_clients

def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    """


    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model

    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()  # Model Params： oldCM - oldGM_t
    new_param_update = dict()  # Model Params： newCM - newGM_t

    new_global_model_state = global_model_after_forget.state_dict()  # newGM_t

    return_model_state = dict()  # newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    # print("lai",len(old_client_models),len(new_client_models))
    assert len(old_client_models) == len(new_client_models)

    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0 * global_model_before_forget.state_dict()[layer]

        return_model_state[layer] = 0 * global_model_before_forget.state_dict()[layer]

        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii + 1)  # Model Params： oldCM
        new_param_update[layer] /= (ii + 1)  # Model Params： newCM

        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[
            layer]  # 参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[
            layer]  # 参数： newCM - newGM_t

        step_length = torch.norm(old_param_update[layer])  # ||oldCM - oldGM_t||
        step_direction = new_param_update[layer] / torch.norm(
            new_param_update[layer])  # (newCM - newGM_t)/||newCM - newGM_t||

        return_model_state[layer] = new_global_model_state[layer] + step_length * step_direction

    return_global_model = copy.deepcopy(global_model_after_forget)

    return_global_model.load_state_dict(return_model_state)

    return return_global_model

    # return forget_global_model


def unlearning_without_cali(old_global_models, old_client_models, FL_params):
    """


    Parameters
    ----------
    old_client_models : list of DNN models
        All user local update models are saved during the federated learning and training process that is not forgotten.
    FL_params : parameters
        All parameters in federated learning and federated forgetting learning

    Returns
    -------
    global_models : List of DNN models
        In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round

    """
    """
    The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
                 For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
    The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
    """
    if (FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
    print('FL_params.forget_client_idx: ', FL_params.forget_client_idx)

    # if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
    #     raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    forget_client = FL_params.forget_client_idx

    for epoch in range(FL_params.global_epoch):
        # print('old_client', len(old_client_models))
        print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))
        temp = []
        temp.append(old_client_models)  # [epoch*FL_params.N_client: epoch*FL_params.N_client+FL_params.N_client]
        temp.append(old_client_models)
        # print('temp',len(temp))、
        for k_client in forget_client:
            temp.pop(k_client)
        # print('#' * 4, 'Remove {} clients datapoint from global'.format(forget_client), '#' * 4)
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    uncali_global_models = list()
    # uncali_global_models.append(copy.deepcopy(old_global_models[-1]))

    epoch = 0
    for epoch in range(FL_params.local_epoch):
        uncali_global_model = fedavg(old_client_models[epoch])
        uncali_global_models.append(copy.deepcopy(uncali_global_model))
    # print('uncali_global_models: ', len(uncali_global_models))

    """
    new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)

    For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
    For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
    For uncalibrated federated forgotten learning, the parameter update of the unforgotten user in standard federated learning is used to directly overlay the new global model to obtain the next round of new global model.
    """
    old_param_update = dict()  # (oldCM_t - oldGM_t)
    return_model_state = dict()  # newGM_t+1

    for epoch in range(FL_params.global_epoch):
        if (epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))

        current_global_model = uncali_global_models[epoch]  # newGM_t
        current_client_models = old_client_models[epoch]  # oldCM_t
        old_global_model = old_global_models[epoch]  # oldGM_t
        # global_model_before_forget = old_global_models[epoch]#old_GM_t

        for layer in current_global_model.state_dict().keys():
            # State variable initialization
            old_param_update[layer] = 0 * current_global_model.state_dict()[layer]
            return_model_state[layer] = 0 * current_global_model.state_dict()[layer]

            for ii in range(len(current_client_models)):
                old_param_update[layer] += current_client_models[ii].state_dict()[layer]
            old_param_update[layer] /= (ii + 1)  # oldCM_t

            old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[
                layer]  # 参数： oldCM_t - oldGM_t

            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[
                layer]  # newGM_t + (oldCM_t - oldGM_t)

        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)

        uncali_global_models.append(return_global_model)

    return uncali_global_models
