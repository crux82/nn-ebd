import os
from datetime import datetime
from tqdm import tqdm
#from tqdm.notebook import tqdm

from it.launcher_probEbDNLearner import run_experiment


def create_exp_configs(values, ethics_system, ethics_mode, active_ef, root, today, epochs, input_type):
    # exp_configs = [
    #     {'alfa': 0.3, 'beta_b': 0.01, 'beta_r': 0.01, 'ethics_system': ethics_system, 'ethics_mode': ethics_mode, 
    #     'active_ef': active_ef, 'root': root, 'today': today, 'epochs': epochs, 'input_type': input_type}
    # ]
    exp_configs = []

    for v in values:
        line = {'alfa': v[0], 'beta_b': v[1], 'beta_r': v[2], 'ethics_system': ethics_system, 'ethics_mode': ethics_mode, 
            'active_ef': active_ef, 'root': root, 'today': today, 'epochs': epochs, 'input_type': input_type}
        exp_configs.append(line)
    
    return exp_configs
        

def ethics(values, root_path, active_ef, dataset_size, dataset_name, dataset_y_name, today = datetime.today().strftime("%d_%m_%Y__%H-%M-%S")):
    
    ###Config
    ethics_system, ethics_mode, input_type, epochs = 'Ethics_2', 'Steep', 'Business_only', 1000
    root = "resources" + os.sep + dataset_name

    # We create the vector of the different configurations
    exp_configs = create_exp_configs(values, ethics_system, ethics_mode, active_ef, root, today, epochs, input_type)

    # for each configuration, train the modelo
    network_output_dir_list = []
    oracles_output_dir_list = []

    for exp_config in tqdm(exp_configs):
        path_csv_enriched, oracles_output_dir, network_output_dir = run_experiment(root_path=root_path, config_dict=exp_config, dataset_size = dataset_size, dataset_name=dataset_name, dataset_y_name=dataset_y_name)
        network_output_dir_list.append(network_output_dir)
        oracles_output_dir_list.append(oracles_output_dir)
    
    return network_output_dir_list, oracles_output_dir_list, path_csv_enriched, exp_configs
