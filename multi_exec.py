from config import *

from it.utils.data_utils import load_dataset, split_features_target, create_clustered_dataset
from it.cross_validation import KFold_CrossValidate_MLP, KFold_CrossValidate_ENN, KFold_CrossValidate_ENN_novalidation
from it.models.BasicMLPModel import init_MLP_model
from it.dataloaders import init_DataLoaders_MLP, init_DataLoaders_ENN
from run_ethics_1 import ethics
from it.load_oracles import load_reconstruction_oracle, load_revision_oracle
from it.models.ENN_Model import init_ENN_model
from run_metrics_2 import run_metrics
from it.train_eval_test_loop import train_eval_folds_ENN, train_test_novalidation, test_folds_ENN, train_val_test_MLP

###IF YOU RUN CODE ON COLAB, don't use simply tqdm
#from tqdm.notebook import tqdm as tqdm  
import numpy as np
import random
import torch


def multi_config_exec(SEED):
    # Set dir_name. Can't be done in config because we need the SEED (which changes at every run)
    dir_name = str(today) + "_" + str(active_ef) + "_SEED_" +str(SEED)

    # Some print
    print("TM are", active_ef)
    print("SEED is", SEED)
    print("Dataset name is", dataset_name)
    print("Augmentation=", DATA_AUGMENT)
    print("Max ethic epochs", ETH_NET_EPOCHS, "and max MLP_EPOCHS", MLP_EPOCHS)

    # Let's fix the randomicity, this way is going to be possible to get reproducible results. 
    # Therefore we set the random seed for Python, Numpy and PyTorch.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ####################################################################
    ####################################################################
    ####################################################################
    ####################          DATA         #########################
    print("Load data...")

    #Load data from csv
    dataset_df, dataset_size = load_dataset(root_path, dataset_name)

    #Split features in X and target y.
    X, y = split_features_target(dataset_df, dataset_name)

    number_of_features = X.shape[1]

    #K-fold cross validation
    #   - 90% (including 90% train and 10% validation)
    #   - 10% (test)
    X_train_folds, X_val_folds, X_test_folds, y_train_folds, y_val_folds, y_test_folds = KFold_CrossValidate_MLP(dataset_df, 
                                X.copy(), y.copy(), FOLDS, SEED, DATA_AUGMENT, AUG_OPTIONS)

    # No longer used
     # We create the "clustered" dataset for the future unbiased test set
     # That is, Wwe create 6 new instances for each instance of the dataset: each with a different race
    X_test_clustered_fold = []
    if dataset_name == "compas" and reduced_oracle_actived == True:
        X_test_clustered_fold = create_clustered_dataset(X_test_folds)


    ####################################################################
    ####################################################################
    ####################################################################
    ####################          MLP          #########################
    print("MLP...")

    #Model parameters: you can edit them in the config.py file
    EPOCHS = MLP_EPOCHS
    BATCH_SIZE = BATCH_SIZE_MLP
    LEARNING_RATE = 0.001

    #Init MLP model, optimizer and criterion
    model, optimizer, criterion = init_MLP_model(number_of_features, LEARNING_RATE)

    #Check if GPU is active and place model and criterion on to the device by using the .to method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using", device)
    model, criterion = model.to(device), criterion.to(device)

    #Init Dataloaders
    train_loader_folds, val_loader_folds, test_loader_folds, test_cluster_loader_folds = init_DataLoaders_MLP(X_train_folds, 
                    y_train_folds, X_val_folds, y_val_folds, X_test_folds, X_test_clustered_fold, FOLDS, reduced_oracle_actived, 
                    BATCH_SIZE)

    #Train and test
    index_to_drop, X_reduced, reduced_oracle, predictions, reduced_preds, mlp_fpr_fnr = train_val_test_MLP(X, y, model, 
                    dataset_df, dataset_name, FOLDS, train_loader_folds, val_loader_folds, test_loader_folds, y_test_folds, 
                    EPOCHS, device, optimizer, criterion, bias_features, reduced_oracle_actived, test_cluster_loader_folds, 
                    PATIENCE_VALUE, dir_name)


    ####################################################################
    ####################################################################
    ####################################################################
    ####################         ETHICS        #########################
    print("Ethics...")

    # Compute the oracles of ethical signatures
    network_output_dir_list, oracles_output_dir_list, path_csv_enriched, exp_configs = ethics(values, root_path, active_ef,
                                                                    dataset_size, dataset_name, dataset_y_name, dir_name)

    print("ethics exp_configs", exp_configs)


    ####################################################################
    ####################################################################
    ####################################################################
    ###########     Load Oracles & Kfold-cross validation    ###########
    print("Load Oracles and Kfold-cross validation")

    #Load Business oracle
    business_oracle = y.copy()

    #Load reconstruction oracle
    pd_rec_oracle = load_reconstruction_oracle(business_oracle, oracles_output_dir_list, reduced_oracle_actived,
                    index_to_drop, number_of_decisions, number_of_eth_values)

    #Load revision oracle
    pd_revi_oracle_list = load_revision_oracle(business_oracle, oracles_output_dir_list, reduced_oracle_actived,
                    index_to_drop, number_of_decisions, number_of_eth_values)
    
    #Kfold cross-validation
    if do_validation == False:
        y_val_EA_fold_config, X_val_fold, y_val_BE_fold, y_val_EE_fold = [], [], [], []
        #Kfold cross-validation ENN German Credit
        #To be as faithful as possible to the old execution in Keras, I do not create a validation test for german credit daatset
        X_train_fold, y_train_BE_fold, y_train_EE_fold, X_test_fold, y_test_BE_fold, y_train_EA_fold_config = \
            KFold_CrossValidate_ENN_novalidation(dataset_df, X.copy(), y.copy(), pd_rec_oracle, pd_revi_oracle_list, FOLDS, 
            SEED, DATA_AUGMENT, AUG_OPTIONS)  
    else:
        #In all other cases I create a train, validation and test set.
        X_train_fold, y_train_BE_fold, y_train_EE_fold, X_val_fold, y_val_BE_fold, y_val_EE_fold, X_test_fold, \
            y_test_BE_fold, y_train_EA_fold_config, y_val_EA_fold_config = \
            KFold_CrossValidate_ENN(dataset_df, X.copy(), business_oracle, pd_rec_oracle, pd_revi_oracle_list, FOLDS, 
            SEED, DATA_AUGMENT, AUG_OPTIONS)


    ####################################################################
    ####################################################################
    ####################################################################
    ####################          ENN          #########################
    print("ENN...")

    #Model parameters: you can edit them in the config.py file
    EPOCHS = ETH_NET_EPOCHS
    BATCH_SIZE = BATCH_SIZE_ENN
    LEARNING_RATE = 0.001
    
    #Init Model, optimizer and criterions
    model, optimizer, criterionBE, criterionEE_EA = init_ENN_model(dataset_name, number_of_features, LEARNING_RATE)

    #Check if GPU is active and place model and criterion on to the device by using the .to method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, criterionBE, criterionEE_EA = model.to(device), criterionBE.to(device), criterionEE_EA.to(device)

    #Create and initialize Dataloaders. We’ll use a batch_size = 1 for test dataloader.
    train_loader_fold_config, val_loader_fold_config, test_loader_fold_config = \
                init_DataLoaders_ENN(X_train_fold, y_train_BE_fold, y_train_EE_fold, y_train_EA_fold_config,
                y_val_EA_fold_config, X_val_fold, y_val_BE_fold, y_val_EE_fold, X_test_fold, dataset_name, FOLDS, BATCH_SIZE, 
                do_validation)

    if do_validation == False:
        #Train-Test german-credit dataset, NO VALIDATION
        new_exp_config, network_output_dir = train_test_novalidation(X, LEARNING_RATE, y_train_BE_fold, SEED, FOLDS, EPOCHS, 
                dataset_name, train_loader_fold_config, test_loader_fold_config, network_output_dir_list, y_test_BE_fold, 
                exp_configs)
    else:
        #Train, eval ...
        configurations_dictionary_list, dict_lime_model_c, init_list_var_test = train_eval_folds_ENN(PATIENCE_VALUE, SEED, FOLDS,
                dataset_name, X, LEARNING_RATE, EPOCHS, train_loader_fold_config, val_loader_fold_config, test_loader_fold_config,
                network_output_dir_list, exp_configs, y_train_BE_fold, y_val_BE_fold, dir_name, enable_early_stop, 
                number_of_features, do_validation)
        
        # ... and test folds ENN of all other datasets
        new_exp_config, network_output_dir = test_folds_ENN(X, y, X_reduced, bias_features, alfa_values, beta_values,
                dataset_name, configurations_dictionary_list, FOLDS, dict_lime_model_c, y_test_BE_fold, index_to_drop, 
                reduced_oracle_actived, reduced_oracle, predictions, reduced_preds, exp_configs, init_list_var_test)


    ####################################################################
    ####################################################################
    ####################################################################
    ####################          METRICS           ####################
    print("Compute metrics...")
    
    if reduced_oracle_actived:
        number_of_instances = len(reduced_oracle)
        #reduced_preds are the predictions generated by the mlp
        run_metrics(reduced_preds, path_csv_enriched, root_path, new_exp_config, [network_output_dir], \
                    number_of_instances, dataset_name, reduced_oracle, index_to_drop, mlp_fpr_fnr, bias_features)
    else:
        number_of_instances = len(y)
        run_metrics(predictions, path_csv_enriched, root_path, new_exp_config, [network_output_dir], \
                    number_of_instances, dataset_name, y, [], mlp_fpr_fnr, bias_features)

    print("Done!")
