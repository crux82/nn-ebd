from tqdm import tqdm
#from tqdm.notebook import tqdm
import random
import torch
import numpy as np
import os
import copy
import gc


from it.utils.EarlyStopping_utils import EarlyStopping
from it.utils.ENN_train_utils import init_all_train_loop, train_model, val_model, print_statistics, print_early_stop_graph, test_model
from it.policy_EC_EU import run_policyEC_EU, save_policyEC_EU
from it.utils.stats_utils import calcola_differenza_probabilita, statistiche_popolazione, count_predictions, plot_bias_feature
from it.utils.MLP_learning_utils import train_mlp, val_mlp, test_mlp, test_cluster_MLP

def train_eval_folds_ENN(PATIENCE_VALUE, SEED, FOLDS, dataset_name, X, LEARNING_RATE, EPOCHS, train_loader_fold_config, val_loader_fold_config, test_loader_fold_config, network_output_dir_list, 
                        exp_configs, y_train_BE_fold, y_val_BE_fold, dir_name, enable_early_stop, number_of_features, 
                        do_validation):
    verbose = False #False only config and fold bar, True config & fold & eary stop  data and plots
    model_list_u_config, model_list_c_config = [], [] # save the models for the test phase. 
                                                  # each configuration has 10 folds, and therefore 10 models. model_list_u_config will contain 
                                                  # all unconstrained models. model_list_c_config will contain all models 
                                                  # constrained
    patience = PATIENCE_VALUE
    configurations_dictionary_list = []
    dict_lime_model_c = {}

    for train_loader_fold, val_loader_fold, test_loader_fold, network_output_dir, exp_config in tqdm(zip(train_loader_fold_config, val_loader_fold_config, test_loader_fold_config, network_output_dir_list, exp_configs), total = len(exp_configs)):

        #TODO to delete?
        #Get reproducible results ###################
        random.seed(SEED)                           #
        np.random.seed(SEED)                        #
        torch.manual_seed(SEED)                     #
        torch.cuda.manual_seed(SEED)                #
        torch.backends.cudnn.deterministic = True   #
        #############################################

        ### Init all variables ###

        model, optimizer, criterionBE, criterionEE_EA, device, model_unconstr, model_constr, _, _, _, model_list_c,\
            model_list_u, init_list_var_test, alfa, beta = init_all_train_loop(exp_config, dataset_name, X, LEARNING_RATE, 
                                                                                number_of_features)
        
        # Config variables for a single configuration #
        # for train_loader, val_loader, test_loader, test_cluster_loader, fold in zip(train_loader_fold, val_loader_fold, test_loader_fold, test_cluster_loader_fold, range(FOLDS)):
        for train_loader, val_loader, test_loader, fold in zip(train_loader_fold, val_loader_fold, test_loader_fold, range(FOLDS)):
        
            print("### Train, val and test fold", fold)

            # to track the average training and validation accuracy per epoch as the model trains
            avg_train_acc_constr, avg_val_acc_constr = [], [] 
            avg_train_acc_unconstr, avg_val_acc_unconstr = [], [] 
            

            # initialize the early_stopping object
            early_stopping_constr = EarlyStopping(patience=patience, verbose=False)
            early_stopping_unconstr = EarlyStopping(patience=patience, verbose=False)

            #variables for early stop
            es_u, es_c = False, False 

            #Starts epochs for a configuration
            for e in tqdm(range(1, EPOCHS+1)):

                ###################
                ### Start train ###
                ###################
                model.train()
                model, train_acc_unconstr, train_acc_constr = train_model(X, y_train_BE_fold, model, train_loader, optimizer, 
                                                                          criterionBE, criterionEE_EA, fold, device)

                ##################
                ### Start eval ###
                ##################
                model.eval() # prep model for evaluation
                model, val_acc_unconstr, val_acc_constr = val_model(X, y_val_BE_fold, model, val_loader, fold, device)
                # print train/val stats #
                avg_train_acc_unconstr, avg_val_acc_unconstr = print_statistics(train_acc_unconstr, val_acc_unconstr, e, avg_train_acc_unconstr, avg_val_acc_unconstr, "unconstrained", EPOCHS)
                avg_train_acc_constr, avg_val_acc_constr = print_statistics(train_acc_constr, val_acc_constr, e, avg_train_acc_constr, avg_val_acc_constr, "constrained", EPOCHS)

                #############################################################################
                # early_stopping needs the validation acc to check if it has decresed,      #
                # and if it has, it will make a checkpoint of the current model             #

                early_stopping_unconstr(val_acc_unconstr, model, "unconstr", dataset_name, dir_name)
                early_stopping_constr(val_acc_constr, model, "constr", dataset_name, dir_name)

                if enable_early_stop:                                                       #
                    if early_stopping_unconstr.early_stop and es_u == False:                #
                        epoch_len = len(str(EPOCHS))                                        #
                        print_msg = (f'[{e:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +         #
                                    f'unc_train_acc: {train_acc_unconstr:.5f} ' +           #
                                    f'unc_valid_acc: {val_acc_unconstr:.5f}')               #

                        if verbose:                                                         #
                            print("Early stopping.", print_msg)                             #

                        es_u = True                                                         #
                    
                    
                    if early_stopping_constr.early_stop and es_c == False:                  #
                        epoch_len = len(str(EPOCHS))                                        #
                        print_msg = (f'[{e:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +         #
                                    f'constr_train_acc: {train_acc_constr:.5f} ' +          #
                                    f'constr_valid_acc: {val_acc_constr:.5f}')

                        if verbose:                                                         #
                            print("Early stopping.", print_msg)                             #

                        #when early_stop_sentinel=2, then the stop is reached for both policies
                        es_c = True
                        #We also make the first early stop = True, in the case here we found a good one for the constrained
                        #es_u = True

                    
                    if es_u and es_c:                                                       #
                        break                                                               #
                #############################################################################

            if do_validation == False:
                #simply get the latest model
                model_unconstr = copy.deepcopy(model)
                model_constr = copy.deepcopy(model)
            else:
                #load the best model
                path_checkpoint = os.path.join("resources", dataset_name, "data", "enriched_data", dir_name)    
                model_unconstr.load_state_dict(torch.load(os.path.join(path_checkpoint, 'unconstr_checkpoint' + '.pt')))
                model_constr.load_state_dict(torch.load(os.path.join(path_checkpoint, 'constr_checkpoint' + '.pt')))

            model_list_u.append(copy.deepcopy(model_unconstr)) #save the model for the test phase
            model_list_c.append(copy.deepcopy(model_constr)) #save the model for later (Lime) and test phase
            

            #print graph
            if verbose:
                print_early_stop_graph(avg_train_acc_unconstr, avg_val_acc_unconstr, "unconstrained")
                print_early_stop_graph(avg_train_acc_constr, avg_val_acc_constr, "constrained")


            # We create a dictionary for each fold of each configuration. We will need it later to choose the best alpha 
            # of each fold given a fixed beta
            if enable_early_stop:
                val_acc_unconstr = early_stopping_unconstr.best_score
                val_acc_constr = early_stopping_constr.best_score

            dict_configurations = {
                "alfa": alfa,
                "beta": beta,
                "model_u": copy.deepcopy(model_unconstr),
                "model_c": copy.deepcopy(model_constr),
                "val_acc_unconstr": val_acc_unconstr,
                "val_acc_constr": val_acc_constr,
                "fold": fold,
                "test_loader": test_loader,
                "net_output_dir": network_output_dir
            }

            configurations_dictionary_list.append(copy.copy(dict_configurations))
            dict_configurations.clear()

            ##############################################
            # reset model for next cross validation step #
            ##############################################
            for name, module in model.named_children():
                if name == "relu" or name == "dropout" or name == "softmax":
                    continue
                module.reset_parameters()

        print("\n##################################################################\n")

        model_list_c_config.append(model_list_c)
        model_list_u_config.append(model_list_u)


        #We create a dictionary with alpha, beta and model for LIME
        chiave = str(alfa)+str(beta)
        dict_lime_model_c[chiave] =  model_list_c

        
        import gc
        del model
        del model_unconstr
        del model_constr
        del optimizer
        del criterionBE
        del criterionEE_EA
        del device

        gc.collect()

    return configurations_dictionary_list, dict_lime_model_c, init_list_var_test

def train_test_novalidation(X, y, LEARNING_RATE, y_train_BE_fold, SEED, FOLDS, EPOCHS, dataset_name, train_loader_fold_config, test_loader_fold_config, network_output_dir_list, 
                                y_test_BE_fold, exp_configs, number_of_features, bias_features):
    new_exp_config = []
    net_output_dir = ""
    model_list_u_config, model_list_c_config = [], [] #save models for files and test phase. each configuration has 10 folds, and therefore 10 models.

    
    for train_loader_fold, test_loader_fold, network_output_dir, exp_config in \
            tqdm(zip(train_loader_fold_config, test_loader_fold_config, network_output_dir_list, exp_configs), total = len(exp_configs)):

        #Get reproducible results ###################
        random.seed(SEED)                           #
        np.random.seed(SEED)                        #
        torch.manual_seed(SEED)                     #
        torch.cuda.manual_seed(SEED)                #
        torch.backends.cudnn.deterministic = True   #
        #############################################

        ### Init all variables ###

        model, optimizer, criterionBE, criterionEE_EA, device, model_unconstr, model_constr, _, _, _, model_list_c,\
             model_list_u, init_list_var_test, alfa, beta = init_all_train_loop(exp_config, dataset_name, X, LEARNING_RATE, 
                                                                                number_of_features)

        # Config variables for a single configuration #
        # for train_loader, val_loader, test_loader, test_cluster_loader, fold in zip(train_loader_fold, val_loader_fold, test_loader_fold, test_cluster_loader_fold, range(FOLDS)):
        for train_loader, test_loader, fold in zip(train_loader_fold, test_loader_fold, range(FOLDS)):
        
            print("### Train and test fold", fold)

            #Starts epochs for a configuration
            for e in tqdm(range(1, EPOCHS+1)):

                ###################
                ### Start train ###
                ###################
                model.train()
                model, _, _ = train_model(X, y_train_BE_fold, model, train_loader, optimizer, criterionBE, 
                                            criterionEE_EA, fold, device)
                                                                          
            #simply get the latest model
            model_unconstr = copy.deepcopy(model)
            model_constr = copy.deepcopy(model)

            ##################
            ### Start test ###
            ##################
            model_unconstr.eval()
            model_constr.eval()

            predictions_EbDNN_u = test_model(model_unconstr, test_loader, device)
            predictions_EbDNN_c = test_model(model_constr, test_loader, device)

            ########################
            ### Compute policies ###
            ########################
            init_list_var_test = run_policyEC_EU(
                predictions_u = predictions_EbDNN_u,
                predictions_c = predictions_EbDNN_c,
                test_inds = y_test_BE_fold[fold].index.tolist(),
                init_list_var = init_list_var_test
                )

            ##############################################
            # reset model for next cross validation step #
            ##############################################
            for name, module in model.named_children():
                if name == "relu" or name == "dropout" or name == "softmax":
                    continue
                module.reset_parameters()
        
        #FP rate and FN rate calculation for constrained model
        predictions_test_tmp = [np.argmax(r) for r in init_list_var_test[6]] #calculation of final predictions


        stats_list = []
        for bias_feature in bias_features:
            rates, totals = count_predictions(predictions_test_tmp, y, X.copy(), bias_feature, dataset_name)
            stats_list.extend(rates)
            stats_list.extend(totals)
            
        tmp_exp_config = exp_configs[0].copy()
        tmp_exp_config['alfa'], tmp_exp_config['beta_b'], tmp_exp_config['beta_r'] = alfa, beta, beta
        tmp_exp_config['stats'] = stats_list

        new_exp_config.append(tmp_exp_config.copy())
        tmp_exp_config.clear()


        save_policyEC_EU(output_path = network_output_dir, # path related to configuration
                        save_var = init_list_var_test) #init_list_var_test)

        print("Controllo", network_output_dir.split("/output/")[0])
        net_output_dir = network_output_dir.split("/output/")[0] #I'll just take an initial piece, that's all I need.

        print("\n##################################################################\n")

        model_list_c_config.append(model_list_c)
        model_list_u_config.append(model_list_u)

        import gc
        del model
        del model_unconstr
        del model_constr
        del optimizer
        del criterionBE
        del criterionEE_EA
        del device

        gc.collect()
        print("d1 new_exp_config", new_exp_config)

    return new_exp_config, net_output_dir


def test_folds_ENN(X, y, X_reduced, bias_features, alfa_values, beta_values, dataset_name, configurations_dictionary_list, FOLDS, dict_lime_model_c, y_test_BE_fold, index_to_drop, reduced_oracle_actived, reduced_oracle, predictions, reduced_predictions, exp_configs, init_list_var_test):
    model_c_lime_config = [] #here go the constrained models with the best alpha for files

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    new_exp_config = []
    dict_sum_acc_alfa = {}
    for b_v in beta_values: #for each beta and each alpha
        dict_sum_acc_alfa.clear()
        
        for a_v in alfa_values:
            dict_sum_acc_alfa[a_v] = 0
        
        for d in configurations_dictionary_list:
            if d["beta"]==b_v:
                alfa = copy.copy(d["alfa"])
                w_acc_constr = copy.copy(d["val_acc_constr"])
                
                #practically for each beta there will be several alphas. For each alpha, since we set beta, there will be 10 folds. 
                # So here we are adding up the accuracy of 10 folds, all corresponding to one alpha. 
                # We want to find that alpha which out of 10 folds gives us on average the best accuracy.
                dict_sum_acc_alfa[alfa] += w_acc_constr
        
        for a_v in alfa_values:
            dict_sum_acc_alfa[a_v] = copy.copy(dict_sum_acc_alfa[a_v])/FOLDS #average ##error
        
        best_alfa_value = max(dict_sum_acc_alfa, key=dict_sum_acc_alfa.get) #I take the alpha with the best average accuracy. I will use this for the test
        print("with beta", b_v, "best alfa is", best_alfa_value)

        #I list the model for LIME
        chiave = str(best_alfa_value)+str(b_v)
        model_c_lime_config.append(dict_lime_model_c[chiave])

        for fold in range(FOLDS): #for each fold
            #print("fold", fold)

            for d in configurations_dictionary_list:
                if d["beta"] == b_v and d["alfa"] == best_alfa_value and d['fold'] == fold:
                    #so we have for a specific fold of a beta fixed the right model to test. We load all the values
                    model_unconstr = copy.copy(d["model_u"])
                    model_constr = copy.copy(d["model_c"])
                    test_loader = copy.copy(d["test_loader"])
                    right_fold = copy.copy(d["fold"])
                    network_output_dir = copy.copy(d["net_output_dir"])
                    print("I have chosen b_v", b_v, "best_alfa_value", best_alfa_value, "fold", fold)



            ##################
            ### Start test ###
            ##################
            model_unconstr.eval()
            model_constr.eval()

            predictions_EbDNN_u = test_model(model_unconstr, test_loader, device)
            predictions_EbDNN_c = test_model(model_constr, test_loader, device)

            ########################
            ### Compute policies ###
            ########################
            init_list_var_test = run_policyEC_EU(
                predictions_u = predictions_EbDNN_u,
                predictions_c = predictions_EbDNN_c,
                test_inds = y_test_BE_fold[right_fold].index.tolist(),
                init_list_var = init_list_var_test
                )

        reduced_init_list_var_test = [[]]*7 #7 and not 3, for code compatibility
        temp_list_var_test = [[],[],[]]
        for r in range(len(y)):
            if r not in index_to_drop:
                temp_list_var_test[0].append(init_list_var_test[4][r])
                temp_list_var_test[1].append(init_list_var_test[5][r])
                temp_list_var_test[2].append(init_list_var_test[6][r])
        reduced_init_list_var_test[4] = np.array(temp_list_var_test[0])
        reduced_init_list_var_test[5] = np.array(temp_list_var_test[1])
        reduced_init_list_var_test[6] = np.array(temp_list_var_test[2])


        #################
        ###   Stats   ###

        if reduced_oracle_actived:
            #####probs difference calculation#####
            calcola_differenza_probabilita(reduced_init_list_var_test)

            #FP rate and FN rate calculation for unconstrained model
            predictions_test_tmp_unconstr = [np.argmax(r) for r in reduced_init_list_var_test[5]] #calculation of final predictions

            #FP rate and FN rate calculation for constrained model
            predictions_test_tmp = [np.argmax(r) for r in reduced_init_list_var_test[6]] #calculation of final predictions

            stats_list = []
            for bias_feature in bias_features:
                rates, totals = count_predictions(predictions_test_tmp, reduced_oracle, X_reduced.copy(), bias_feature, dataset_name)
                stats_list.extend(rates)
                stats_list.extend(totals)
            # FPR, FPR_bf, FPR_not_bf, FNR, FNR_bf, FNR_not_bf = rate_list
            # tot_p, tot_n, tot_bf_p, tot_not_bf_p, tot_bf_n, tot_not_bf_n, ppv_bf, ppv_not_bf = total_pos_neg_list
        else:
            #####calcolo differenza probs#####
            calcola_differenza_probabilita(init_list_var_test)

            #FP rate and FN rate calculation for constrained model
            predictions_test_tmp_unconstr = [np.argmax(r) for r in init_list_var_test[5]] #calculation of final predictions

            #FP rate and FN rate calculation for constrained model
            predictions_test_tmp = [np.argmax(r) for r in init_list_var_test[6]] #calculation of final predictions

            stats_list = []
            for bias_feature in bias_features:
                rates, totals = count_predictions(predictions_test_tmp, y, X.copy(), bias_feature, dataset_name)
                stats_list.extend(rates)
                stats_list.extend(totals)

        tmp_exp_config = exp_configs[0].copy()
        tmp_exp_config['alfa'], tmp_exp_config['beta_b'], tmp_exp_config['beta_r'] = best_alfa_value, b_v, b_v
        tmp_exp_config['stats'] = stats_list

        new_exp_config.append(tmp_exp_config.copy())
        tmp_exp_config.clear()

        ################################
        ### Writes policies to files ###
        print("network_output_dir", network_output_dir)
        if reduced_oracle_actived:
            save_policyEC_EU(output_path = network_output_dir, # path related to configuration
                            save_var = reduced_init_list_var_test) #init_list_var_test)
        else:
            save_policyEC_EU(output_path = network_output_dir, # path related to configuration
                            save_var = init_list_var_test) #init_list_var_test)

        network_output_dir = network_output_dir.split("/output/")[0] #We will just take an initial piece, that's all we need.


    return new_exp_config, network_output_dir



def train_val_test_MLP(X, y, model, dataset_df, dataset_name, FOLDS, train_loader_folds, val_loader_folds, 
                    test_loader_folds, y_test_folds, EPOCHS, device, optimizer, criterion, bias_features, 
                    reduced_oracle_actived, test_cluster_loader_folds, patience, dir_name):
    #Init some vars. TODO comment better
    index_to_drop = [] #It will contain the indices of the instances deleted from the original dataset to form the unbiased dataset.
    model_list = []
    predictions = np.zeros(shape=dataset_df.shape[0])
    predictions_cluster = np.zeros(shape=dataset_df.shape[0])

    #Train and test every folds
    for fold in range(FOLDS):
        print("\nStart train with fold", fold)
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        for e in tqdm(range(1, EPOCHS+1)):
            #############
            ### TRAIN ###
            model.train()
            model = train_mlp(model, train_loader_folds[fold], EPOCHS, device, optimizer, criterion)
            #model_list.append(copy.deepcopy(model))

            ############
            ### EVAL ###  
            model.eval()
            model, mlp_val_accuracy = val_mlp(model, val_loader_folds[fold], EPOCHS, device)

            early_stopping(mlp_val_accuracy, model, "mlp", dataset_name, dir_name)
            if early_stopping.early_stop:
                break

        #load the best model
        path_checkpoint = os.path.join("resources", dataset_name, "data", "enriched_data", dir_name)
        model.load_state_dict(torch.load(os.path.join(path_checkpoint, 'mlp_checkpoint' + '.pt')))


        ############
        ### TEST ###
        model.eval()
        y_pred_list = test_mlp(model, test_loader_folds[fold], device)

        # index label of every folds
        index_y = y_test_folds[fold].index

        # put all results in predictions, to calculate metrics
        for ind, label in zip(index_y, y_pred_list):
            predictions[ind] = label

        #reset model for next cross validation step
        for name, module in model.named_children():
            if name == "relu" or name == "dropout" or name == "softmax":
                continue
            module.reset_parameters()

    #TODO: Enter these values in the final results
    mlp_fpr_fnr = []
    for bias_feature in bias_features:
        mlp_rates, _ = count_predictions(predictions, y.copy(), X.copy(), bias_feature, dataset_name)
        #mlp_rates are 8 values: bias_FPR, bias_FNR, FPR_bf, FPR_not_bf, FNR_bf, FNR_not_bf
        mlp_rates = np.delete(mlp_rates, [0,1])

        mlp_fpr_fnr.extend(mlp_rates)

    reduced_oracle, reduced_preds = [], []
    if reduced_oracle_actived:
        reduced_preds, reduced_oracle, index_to_drop = test_cluster_MLP(y.copy(), FOLDS, device, model_list, \
                                                        test_cluster_loader_folds, y_test_folds, predictions_cluster, index_to_drop)
        
    #X_reduced calculation and statistical calculation
    X_reduced = X.copy().drop(X.copy().index[index_to_drop])
    # if reduced_oracle_actived:
    #     rate_list, _ = count_predictions(reduced_preds, reduced_oracle, X.copy(), bias_features, dataset_name)
    #     FPR, FPR_bf, FPR_not_bf, FNR, FNR_bf, FNR_not_bf = rate_list

    del model
    del optimizer
    del criterion
    del device
    gc.collect()


    return index_to_drop, X_reduced, reduced_oracle, predictions, reduced_preds, mlp_fpr_fnr


