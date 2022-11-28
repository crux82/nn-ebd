from it.policy_EC_EU import init_policyEC_EU, compute_accuracies
import numpy as np
import os
import scipy.stats as sts
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from it.models.ENN_Model import Model_Ethics



########################
## TRAINING FUNCTIONS ##

def train_model(X, y_train_BE_fold, model, train_loader, optimizer, criterionBE, criterionEE_EA, fold, device):
    
    # Init two policies: EC and EU - Only train step
    #init only with the length of the fold as I do not need to calculate the policy by length of the entire dataset
    init_list_var = init_policyEC_EU(X)
    output_BE_list, output_EE_list, output_EA_list = [], [], []

    for data, target_BE, target_EE, target_EA in train_loader:

        data, target_BE, target_EE, target_EA = data.to(device), target_BE.to(device), target_EE.to(device), target_EA.to(device)
        
        optimizer.zero_grad()
        
        output_BE, output_EE, output_EA = model(data)

        loss_BE = criterionBE(output_BE, target_BE)
        loss_EE = criterionEE_EA(output_EE, target_EE)
        loss_EA = criterionEE_EA(output_EA, target_EA)

        #Sum loss
        loss = loss_BE + loss_EE + loss_EA

        loss.backward()

        optimizer.step()

        #to print loss -> loss.item(). Maybe with if on ephocs %.


        ########################
        ### Compute policies ###
        ########################
        output_BE_list += output_BE.tolist()
        output_EE_list += output_EE.tolist()
        output_EA_list += output_EA.tolist()

    acc_unconstr, acc_constr = compute_accuracies(output_BE_list, output_EE_list, output_EA_list, 
                                                    y_train_BE_fold[fold], init_list_var)

    return model, acc_unconstr, acc_constr#, w_train_acc_cons, w_train_acc_uncons






##########################
## VALIDATION FUNCTIONS ##

def load_ethical_signature_validation(base_path, number_of_values, number_of_classes, val_inds):
    number_of_instances = len(val_inds)
    ethical_signature = np.zeros(shape=(number_of_instances,2*number_of_classes*number_of_values))

    with open(os.path.join(base_path,"overall_ethical_signature.txt"), "r+") as f:
        lines = f.readlines()
        for i,z in zip(val_inds, range(number_of_instances)):
            line = lines[i]
            vector_tokens = line.split(":")[1].lstrip().rstrip().split(",")
            for v in range(len(vector_tokens)):
                ethical_signature[z][v] = vector_tokens[v]
    return ethical_signature

def init_oracle_path_validation(root_path, dir_name, exp_config, dataset_name):
    sm_factor, tweaking_factor_benefit, tweaking_factor_risk = exp_config['alfa'], exp_config['beta_b'], exp_config['beta_r']
    smoot = str(sm_factor).replace(".", "")+"_smoothing"
    assert tweaking_factor_benefit == tweaking_factor_risk, "tweaking factors are different"
    tweak = str(tweaking_factor_benefit).replace(".", "")
    tweak = tweak+"_"+tweak+"_tweaking"
    oracle_path = os.path.join(root_path, "resources", dataset_name, "data", "enriched_data", dir_name, 'Ethics_2', 'Steep', 
                               smoot, tweak)
    
    return oracle_path

def compute_confusion_matrix_validation(predictions, oracles):
    confusion_matrix = np.zeros(shape=(2,2))
    for i in range(predictions.shape[0]):
        row_index = predictions[i]
        col_index = oracles[i]
        confusion_matrix[int(row_index)][int(col_index)] += 1
    return confusion_matrix


def compute_business_metrics_validation(predictions, oracles, dataset_name):
    confusion_matrix = compute_confusion_matrix_validation(predictions=predictions, oracles=oracles)

    if dataset_name == "german_credit": #TODO: reverse the target values in the dataset, no longer do this in the code
        tp = confusion_matrix[0,0]
        fp = confusion_matrix[0,1]
        fn = confusion_matrix[1,0]
        tn = confusion_matrix[1,1]

    else:
        tp = confusion_matrix[1,1]
        fp = confusion_matrix[1,0]
        fn = confusion_matrix[0,1]
        tn = confusion_matrix[0,0]
        #print("tp", tp, "fp", fp, "fn", fn, "tn", tn)


    precision_c_0 = np.divide(tp, tp+fp)
    if np.isnan(precision_c_0):
        precision_c_0 = 1
    recall_c_0 = np.divide(tp,tp+fn)
    if np.isnan(recall_c_0):
        recall_c_0 = 1
    precision_c_1 = np.divide(tn,tn+fn)
    if np.isnan(precision_c_1):
        precision_c_1 = 1
    recall_c_1 = np.divide(tn,tn+fp)
    if np.isnan(recall_c_1):
        recall_c_1 = 1
    f_1_c_0 = np.divide(2*precision_c_0*recall_c_0, precision_c_0+recall_c_0)
    f_1_c_1 = np.divide(2*precision_c_1*recall_c_1, precision_c_1+recall_c_1)
    f_1_mean = np.divide(f_1_c_0+f_1_c_1, 2)
    accuracy = np.divide(tp+tn, tp+tn+fp+fn)
    business_metrics_dict = dict()
    business_metrics_dict.clear()
    business_metrics_dict['Precision C 0'] = precision_c_0
    business_metrics_dict['Recall C 0'] = recall_c_0
    business_metrics_dict['F1 C 0'] = f_1_c_0
    business_metrics_dict['Precision C 1'] = precision_c_1
    business_metrics_dict['Recall C 1'] = recall_c_1
    business_metrics_dict['F1 C 1'] = f_1_c_1
    business_metrics_dict['F1 mean'] = f_1_mean
    business_metrics_dict['Accuracy'] = accuracy

    #print("precision_c_0", precision_c_0, "recall_c_0", recall_c_0, "precision_c_1", precision_c_1, "recall_c_1", recall_c_1)
    #print("f_1_c_0", f_1_c_0, "f_1_c_1", f_1_c_1)
    return business_metrics_dict


def check_probability_vector(vect):
    if np.min(vect) < 0:
        return -1
    if np.abs(np.sum(vect)-1)>0.001:
        return -2
    return 0

def compute_symmetric_kl(pdf_1, pdf_2):
    #Check if input vectors are distributions (up to fixed threshold)
    check_code_1 = check_probability_vector(pdf_1)
    if check_code_1 < 0:
        if check_code_1 == -1:
            print("First vector has at least one negative element")
        if check_code_1 == -2:
            print("First vector does not sum to 1")
            print(str(np.sum(pdf_1)))
            print("len ->"+str(pdf_1.shape[0]))
        raise ValueError

    check_code_2 = check_probability_vector(pdf_2)
    if check_code_2 < 0:
        if check_code_2 == -1:
            print("Second vector has at least one negative element")
        if check_code_2 == -2:
            print("Second vector does not sum to 1")
        raise ValueError

    first_term = sts.entropy(pdf_1, pdf_2)
    second_term = sts.entropy(pdf_2, pdf_1)

    return np.divide(first_term+second_term, 2)

def compute_ethics_metrics_validation(ethical_signatures, decisions, alfa, beta_b, beta_r):
    number_of_decisions = 2
    number_of_values = 5
    number_of_instances = ethical_signatures.shape[0]
    observed_ethical_signatures = np.zeros(shape=(number_of_instances, ethical_signatures.shape[1]//number_of_decisions))

    ethical_optimum = np.array([0.005,0.015,0.12,0.36,0.5, 0.5, 0.36, 0.12, 0.015, 0.005])
    ethical_minimum = np.array([0.5, 0.36, 0.12, 0.015, 0.005, 0.005,0.015,0.12,0.36,0.5])

    for o in range(number_of_instances):
        s_i = 2*number_of_values*int(decisions[o])
        f_i = 2*number_of_values+s_i
        observed_ethical_signatures[o,:] = ethical_signatures[o, s_i:f_i]

    eth_compliances = np.zeros(shape=number_of_instances)

    for o in range(number_of_instances):

        observed_signature = observed_ethical_signatures[o]

        div_wrt_opt = compute_symmetric_kl(pdf_1=observed_signature[:number_of_values], pdf_2=ethical_optimum[:number_of_values])
        div_wrt_opt += compute_symmetric_kl(pdf_1=observed_signature[number_of_values:], pdf_2=ethical_optimum[number_of_values:])
        div_wrt_min = compute_symmetric_kl(pdf_1=observed_signature[:number_of_values], pdf_2=ethical_minimum[:number_of_values])
        div_wrt_min += compute_symmetric_kl(pdf_1=observed_signature[number_of_values:], pdf_2=ethical_minimum[number_of_values:])

        eth_compliances[o] = (div_wrt_min >= div_wrt_opt)

    return np.mean(eth_compliances)


def print_statistics(train_acc, valid_acc, e, avg_train_losses, avg_valid_losses, type_policy, EPOCHS):
    # calculate average loss over an epoch

    avg_train_losses.append(train_acc)
    avg_valid_losses.append(valid_acc)
    
    epoch_len = len(str(EPOCHS))

    if EPOCHS//10 == 0 or e%(EPOCHS//10) == 0:
        print_msg = (f'[{e:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                    f'{type_policy}_train_acc: {train_acc:.5f} ' +
                    f'{type_policy}_valid_acc: {valid_acc:.5f}')
        
        print(print_msg)


    return avg_train_losses, avg_valid_losses


def val_model(X, y_val_BE_fold, model, val_loader, fold, device):

    # Init two policies: EC and EU - Only valid step
    #init only with the length of the fold as I do not need to calculate the policy by length of the entire dataset
    init_list_var = init_policyEC_EU(X)
    output_BE_list, output_EE_list, output_EA_list = [], [], []

    for data, target_BE, target_EE, target_EA in val_loader:
        
        data, target_BE, target_EE, target_EA = data.to(device), target_BE.to(device), target_EE.to(device), target_EA.to(device)


        # forward pass: compute predicted outputs by passing inputs to the model
        output_BE, output_EE, output_EA = model(data)


        ########################
        ### Compute policies ###
        ########################
        output_BE_list += output_BE.tolist()
        output_EE_list += output_EE.tolist()
        output_EA_list += output_EA.tolist()
        

    acc_unconstr, acc_constr = compute_accuracies(output_BE_list, output_EE_list, output_EA_list, 
                                                  y_val_BE_fold[fold], init_list_var)

    return model, acc_unconstr, acc_constr





####################
## TEST FUNCTIONS ##

def test_model(model, test_loader, device):
    y_predBE_list, y_predEE_list, y_predEA_list = [], [], []

    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_predBE, y_test_predEE, y_test_predEA = model(X_batch)

        y_test_predBE = y_test_predBE.tolist()[0]
        y_test_predEE = y_test_predEE.tolist()[0]
        y_test_predEA = y_test_predEA.tolist()[0]

        #lista di predizioni per un singolo fold
        y_predBE_list.append(y_test_predBE)
        y_predEE_list.append(y_test_predEE)
        y_predEA_list.append(y_test_predEA)
    predictions_EbDNN = [np.array(y_predBE_list), np.array(y_predEE_list), np.array(y_predEA_list)]
    
    return predictions_EbDNN #, model


def print_early_stop_graph(train_acc, valid_acc, type_policy):

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Acc')
    plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Acc')

    # find position of lowest validation loss
    maxposs = valid_acc.index(max(valid_acc))+1 
    plt.axvline(maxposs, linestyle='--', color='r',label='Early Stopping Checkpoint '+str(type_policy))

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0.4, 0.8) # consistent scale
    plt.xlim(0, len(train_acc)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    #fig.savefig('loss_plot.png', bbox_inches='tight')



####################
## INIT FUNCTIONS ##

def init_all_train_loop(exp_config, dataset_name, X, LEARNING_RATE, number_of_features):

    nof = number_of_features
    # if dataset_name == "german_credit":
    #     val_input_sizes = [40, 400, 400, 400, 52]
    #     val_output_sizes = [400, 400, 2, 50, 50]
    # else:
    val_input_sizes = [nof, nof, nof*10, nof*10, 52]
    val_output_sizes = [nof, nof*10, 2, 50, 50]

    model = Model_Ethics(input_sizes = val_input_sizes, output_sizes = val_output_sizes, dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterionBE = nn.CrossEntropyLoss()
    criterionEE_EA = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    criterionBE = criterionBE.to(device)
    criterionEE_EA = criterionEE_EA.to(device)


    model_unconstr = Model_Ethics(input_sizes = val_input_sizes, output_sizes = val_output_sizes, dropout_rate=0.2)
    model_unconstr = model_unconstr.to(device)

    model_constr = Model_Ethics(input_sizes = val_input_sizes, output_sizes = val_output_sizes, dropout_rate=0.2)
    model_constr = model_constr.to(device)


    #Load configurations parameters to print them
    sm_factor, tweaking_factor_benefit, tweaking_factor_risk = exp_config['alfa'], exp_config['beta_b'], exp_config['beta_r']
    print("\n##################################################################")
    print("### Start train with config alfa, beta_b, beta_r", sm_factor, tweaking_factor_benefit, tweaking_factor_risk, "###")


    model_list_c = [] #save the 10 models corresponding to 10 folds for files and for the next test phase (constrained)
    model_list_u = []
  
    # Init two policies: EC and EU #
    init_list_var = init_policyEC_EU(X)

    return model, optimizer, criterionBE, criterionEE_EA, device, model_unconstr, model_constr, sm_factor, tweaking_factor_benefit, tweaking_factor_risk, model_list_c, model_list_u, init_list_var, sm_factor, tweaking_factor_benefit



