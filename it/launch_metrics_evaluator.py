from it.metrics import compute_metrics_csv, compute_baselines_metrics, compute_ethical_dilemmas
from it.utils.data_utils import init_results_dir
import os

def launcher_metrics_eval(root_path, exp_configs, sign_path, number_of_instances, dataset_name, y_oracle, index_to_drop, dir_path_name, mlp_fpr_fnr, bias_features):

    experiment_path = os.path.join(root_path,"resources", dataset_name, "data", "enriched_data", exp_configs[0]['today'])
    results_path = os.path.join(experiment_path, "results")
    save_path_eth_dilemmas=os.path.join(results_path,"ethical_dilemmas.txt")
    root_dir=os.path.join(root_path,"resources", dataset_name, "data", "enriched_data")

    init_results_dir(
        experiment_path,
        sign_path
        ) 

    compute_ethical_dilemmas(
        ethical_signature_path=results_path, 
        save_path=save_path_eth_dilemmas, 
        number_of_values=5, 
        number_of_instances=number_of_instances, 
        number_of_decisions=2,
        index_to_drop=index_to_drop
        )

    eth_compl_mlp = compute_baselines_metrics(
        root_dir=root_path, 
        save_path=results_path, 
        sign_path=sign_path, 
        number_of_instances=number_of_instances,
        dataset_name = dataset_name,
        y_oracle = y_oracle,
        index_to_drop=index_to_drop,
        dir_path_name = dir_path_name,
        mlp_fpr_fnr = mlp_fpr_fnr,
        bias_features = bias_features
        )

    compute_metrics_csv(
        root_dir=root_dir,  
        tokens_list=exp_configs, 
        eth_compl_save_path=results_path, 
        number_of_instances=number_of_instances,
        y_oracle = y_oracle,
        sign_path= sign_path,
        dataset_name=dataset_name,
        index_to_drop=index_to_drop,
        bias_features=bias_features,
        eth_compl_mlp = eth_compl_mlp)
