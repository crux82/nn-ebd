import os

from it.launch_metrics_evaluator import launcher_metrics_eval
import pandas as pd


def run_metrics(predictions, path_csv_enriched, root_path, exp_configs, network_output_dir_list, number_of_instances, dataset_name, y_oracle, index_to_drop, mlp_fpr_fnr, bias_features):
    root = "resources" + os.sep + dataset_name
    print("exp_configs", exp_configs)
    dir_path_name = exp_configs[0]['today']

    ecgc_csv = pd.read_csv(os.path.join(path_csv_enriched, "enriched_categorical_" + dataset_name + ".csv"), sep = ";")
    
    ecgc_csv = ecgc_csv.drop(ecgc_csv.index[index_to_drop])
    ecgc_csv.reset_index(inplace = True, drop = True)
    
    ecgc_csv["basic MLP predictions"] = predictions
    ecgc_csv.to_csv(path_or_buf=os.path.join(root_path, root, "data", "enriched_data", dir_path_name, "final_ethics1_ee_cat_gs_with_predictions.csv" ), sep=";", header=True, index=False)


    # calculates the metrics. network_output_dir_list[-1] is the path of the ethical signature
    launcher_metrics_eval(root_path, exp_configs, network_output_dir_list[-1], number_of_instances, dataset_name, y_oracle, index_to_drop, dir_path_name, mlp_fpr_fnr, bias_features)
