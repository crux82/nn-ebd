import os
from datetime import datetime as dt

from it import compute_rules_tmp as ethicsComp


def run_experiment(root_path, config_dict, dataset_size, dataset_name, dataset_y_name):
    """
    EXPECTED KEYS IN DICT (config_dict):
    alfa - the smoothing factor on business decision
    beta_b, beta_r - tweaking factors for benefit and risks (should be an array of two real numbers)
    ethics_system - it should be a string with the "name" of the applied ethics
    ethics_mode - either 'Steep' or 'Inverted'
    active_ef - it should be a list of strings with the name of the ethical features
    root - it should be the root path (where the base dataset is)
    today - the current day in format DD_MM_YYYY
    epochs - the number of epochs for training

    An example of config_dict is:
    config_dict = {'alfa': 0.1, 'beta_b': 1.0, 'beta_r': 1.0, 'ethics_system': 'Ethics_2', 'ethics_mode': 'Steep', 
                    'active_ef': ['youthFostering', 'socialBenefit'], 'root': 'resources\\german_credit' or compas, 
                    'today': '27_05_2021', 'epochs': 1000, 'input_type': 'Business_only'},
    """

    #Init variables from config_dict
    input_type = config_dict['input_type'] #'Business_only'
    sm_factor, tweaking_factor_benefit, tweaking_factor_risk = config_dict['alfa'], config_dict['beta_b'], config_dict['beta_r']
    ethics_system, ethics_mode = config_dict['ethics_system'], config_dict['ethics_mode'] #'Ethics_2', 'Steep'
    active_ethical_features = config_dict['active_ef'] #['youthFostering', 'socialBenefit']
    root_dir = os.path.join(root_path,config_dict['root']) # 'resources\\german_credit' or compas
    today = config_dict['today'] #'27_05_2021'
    epochs = config_dict['epochs'] #1000

    #Other init
    current_time = dt.now().microsecond #datetime, e.g. 540526
    smoothing_string = str(sm_factor).replace(".", "") + "_smoothing" #01_smoothing
    tweaking_string = str(tweaking_factor_benefit).replace(".", "") + "_" + str(tweaking_factor_risk).replace(".",
                                                                                "") + "_tweaking" #10_10__tweaking

    print("---------- Executing -> ", smoothing_string, "&", tweaking_string, "----------")
    
    #output path oracles
    oracles_output_dir = os.path.join(root_dir, "data", "enriched_data", today, ethics_system, ethics_mode,
                                      smoothing_string, tweaking_string)

    #calculate the oracles: create the files overall_ethical_signature.txt, reconstruction_oracle.txt, 
    # revision_oracle.txt, business_oracle.txt. (compute_rules_temp as ethicsComp)
    ethicsComp.compute_ethics(  
        smoothing_factor=sm_factor, 
        ethics_mode=ethics_mode, 
        tweaking_factor=[tweaking_factor_benefit, tweaking_factor_risk],
        output_dir=oracles_output_dir,
        active_ethical_features=active_ethical_features, 
        dataset_size = dataset_size,
        root_path = root_path,
        dataset_name = dataset_name,
        dataset_y_name = dataset_y_name)

    # In the output folder are the results of training and evaluation
    network_output_dir = os.path.join(oracles_output_dir, "output", input_type, str(epochs) + "_epochs", str(current_time))
    

    return oracles_output_dir, oracles_output_dir, network_output_dir #returns the path used to fetch some files inside

