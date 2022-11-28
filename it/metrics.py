from audioop import bias
import pandas as pd
import numpy as np
from it.utils.data_utils import check_probability_vector, load_decisions, load_ethical_signature
from it.utils.data_utils import parse_float
import scipy.stats as sts
import os
from tqdm import tqdm
#from tqdm.notebook import tqdm
########################
#
# BUSINESS METRICS
#
########################

# CONFUSION MATRIX
#                  Oracle=0  Oracle=1
# Prediction = 0  |   tp   |   fp    |
# Prediction = 1  |   fn   |   tn    |

def compute_confusion_matrix(predictions, oracles):
    confusion_matrix = np.zeros(shape=(2,2))
    for i in range(predictions.shape[0]):
        row_index = predictions[i]
        col_index = oracles[i]
        confusion_matrix[int(row_index)][int(col_index)] += 1
    return confusion_matrix

def simple_baseline_system(business_oracles, dataset_name):

    major_label = np.bincount(business_oracles).argmax()
    predictions = np.full(len(business_oracles), major_label)
    sbs = compute_business_metrics(predictions, business_oracles, dataset_name)

    return sbs, major_label


def var_eth_compliance(mlp_eth_compl, enn_eth_compl):
    diff_eth = -(mlp_eth_compl - enn_eth_compl)
    v_rel = diff_eth/mlp_eth_compl #relative variance (from 3 to 6 augment of 100%)
    v_rel_perc = v_rel*100

    return v_rel_perc


def compute_business_metrics(predictions, oracles, dataset_name):
    

    confusion_matrix = compute_confusion_matrix(predictions=predictions, oracles=oracles)

    if dataset_name == "german_credit":
        tp = confusion_matrix[0,0]
        fp = confusion_matrix[0,1]
        fn = confusion_matrix[1,0]
        tn = confusion_matrix[1,1]

    else:
        tp = confusion_matrix[1,1]
        fp = confusion_matrix[1,0]
        fn = confusion_matrix[0,1]
        tn = confusion_matrix[0,0]

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
    business_metrics_dict['Precision C 0'] = precision_c_0
    business_metrics_dict['Recall C 0'] = recall_c_0
    business_metrics_dict['F1 C 0'] = f_1_c_0
    business_metrics_dict['Precision C 1'] = precision_c_1
    business_metrics_dict['Recall C 1'] = recall_c_1
    business_metrics_dict['F1 C 1'] = f_1_c_1
    business_metrics_dict['F1 mean'] = f_1_mean
    business_metrics_dict['Accuracy'] = accuracy
    return business_metrics_dict

########################
#
# ETHICS METRICS
#
########################


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


def get_ethically_optimal_decisions(signatures):

    number_of_decisions = 2
    number_of_values = 5
    ethical_optimum = np.array([0.005,0.015,0.12,0.36,0.5, 0.5, 0.36, 0.12, 0.015, 0.005])

    scores = np.zeros(shape=number_of_decisions)
    ethically_optimal_decisions = list()

    best_score = 100000
    for d in range(number_of_decisions):
        s_i = 2*number_of_values*d
        f_i = 2*number_of_values+s_i
        tmp_signature = signatures[s_i:f_i]
        tmp_dive = compute_symmetric_kl(pdf_1=tmp_signature[:number_of_values], pdf_2=ethical_optimum[:number_of_values])
        tmp_dive += compute_symmetric_kl(pdf_1=tmp_signature[number_of_values:], pdf_2=ethical_optimum[number_of_values:])
        scores[d] = tmp_dive
        if best_score > tmp_dive:
            best_score = tmp_dive

    for d in range(number_of_decisions):
        if scores[d] == best_score:
            ethically_optimal_decisions.append(d)
    return ethically_optimal_decisions


def compute_tuples_for_bubbles(observed_signatures, ethical_values, alfa, beta_b, beta_r):
    number_of_pairs = observed_signatures.shape[0]
    number_of_values = ethical_values.shape[0]
    coordinates  = np.zeros(shape=(number_of_pairs,2))

    for i in range(number_of_pairs):
        signature = observed_signatures[i]
        x = 0
        y = 0
        for v in range(number_of_values):

            x += signature[v]*ethical_values[v]
            y += signature[v+number_of_values]*ethical_values[v]
        coordinates[i,:] = [x, y]

    unique_coordinates, counts = np.unique(coordinates,return_counts=True, axis=0)

    tuples = np.zeros(shape=(unique_coordinates.shape[0],6))
    for u in range(unique_coordinates.shape[0]):
        tuples[u] = np.concatenate([unique_coordinates[u,:],np.array([counts[u],alfa,beta_b,beta_r])])
    return tuples


def compute_ethics_metrics(ethical_signatures, decisions, alfa, beta_b, beta_r, eth_compliance_score_path):
    number_of_decisions = 2
    number_of_values = 5
    number_of_instances = ethical_signatures.shape[0]
    observed_ethical_signatures = np.zeros(shape=(number_of_instances,
                                                  ethical_signatures.shape[1]//number_of_decisions))

    ethical_optimum = np.array([0.005,0.015,0.12,0.36,0.5, 0.5, 0.36, 0.12, 0.015, 0.005])
    ethical_minimum = np.array([0.5, 0.36, 0.12, 0.015, 0.005, 0.005,0.015,0.12,0.36,0.5])
    ethical_values = np.array([0.1,0.25,0.5,0.75,0.9])

    for o in range(number_of_instances):
        s_i = 2*number_of_values*int(decisions[o])
        f_i = 2*number_of_values+s_i
        observed_ethical_signatures[o,:] = ethical_signatures[o, s_i:f_i]

    eth_compliances = np.zeros(shape=number_of_instances)
    divergences_wrt_ontology =np.zeros(shape=number_of_instances)

    for o in range(number_of_instances):

        possible_signatures = ethical_signatures[o]
        observed_signature = observed_ethical_signatures[o]

        div_wrt_opt = compute_symmetric_kl(pdf_1=observed_signature[:number_of_values], pdf_2=ethical_optimum[:number_of_values])
        div_wrt_opt += compute_symmetric_kl(pdf_1=observed_signature[number_of_values:], pdf_2=ethical_optimum[number_of_values:])
        div_wrt_min = compute_symmetric_kl(pdf_1=observed_signature[:number_of_values], pdf_2=ethical_minimum[:number_of_values])
        div_wrt_min += compute_symmetric_kl(pdf_1=observed_signature[number_of_values:], pdf_2=ethical_minimum[number_of_values:])

        eth_opt_decisions = get_ethically_optimal_decisions(signatures=possible_signatures)
        min_divergence = 10000
        for eth_opt_d in eth_opt_decisions:
            s_opt_i = 2*number_of_values*eth_opt_d
            f_opt_i = 2*number_of_values+s_opt_i
            eth_opt_signature = possible_signatures[s_opt_i:f_opt_i]
            tmp_div = compute_symmetric_kl(pdf_1=observed_signature[:number_of_values], pdf_2=eth_opt_signature[:number_of_values])
            tmp_div += compute_symmetric_kl(pdf_1=observed_signature[number_of_values:], pdf_2=eth_opt_signature[number_of_values:])
            if tmp_div < min_divergence:
                min_divergence = tmp_div
        divergences_wrt_ontology[o] = min_divergence
        eth_compliances[o] = (div_wrt_min >= div_wrt_opt)

    ethics_dictionary = dict()
    ethics_dictionary['eth_compliance'] = np.mean(eth_compliances)
    ethics_dictionary['kl div (wrt onto)'] = np.mean(divergences_wrt_ontology)

    with open(eth_compliance_score_path, "w") as f:
        o_counter = 0
        for eth_compliance in eth_compliances:
            f.write(str(o_counter) + ":" + str(eth_compliance) + "\n")
            o_counter += 1
    return ethics_dictionary

def compute_metrics_csv(root_dir, tokens_list, eth_compl_save_path, number_of_instances, y_oracle, sign_path, 
                        dataset_name, index_to_drop, bias_features, eth_compl_mlp):

    """
    It reproduces the paths where it expects the training data to be, reads them, counts the scores and puts them all in the all_metrics csv.
    puts them all in the all_metrics csv.
    """
    print("compute_metrics_csv")

    output_paths = list()
    oracles_paths = list()
    config_params = list()

    for dict in tqdm(tokens_list):
        smoot = "best_smoothing" if "best" in str(dict['alfa']) else str(dict['alfa']).replace(".", "") + "_smoothing"
        smoot_values = str(dict['alfa']) if "best" in str(dict['alfa']) else "" #I have created another value for simplicity. It corresponds to the case where we test more alpha for a fold

        tweak = str(dict['beta_b']).replace(".", "") + "_" + str(dict['beta_r']).replace(".", "") + "_tweaking"
        tokens = [dict['today'], dict['ethics_system'], dict['ethics_mode'], smoot, tweak, 'output', dict['input_type']]

        true_path = ""
        oracle_path = root_dir
        for t in range(len(tokens)):
            true_path = os.path.join(true_path, tokens[t].lstrip().rstrip())
            if t < len(tokens)-4:
                oracle_path = os.path.join(oracle_path,tokens[t].lstrip().rstrip())
            if t == 1:
                ethics_system = tokens[t].lstrip().rstrip()
            if t == 2:
                ethics_mode = tokens[t].lstrip().rstrip()
            if t == 3:
                alfa = parse_float(tokens[t].split("_")[0])
            if t == 4:
                beta_b = parse_float(tokens[t].split("_")[0])
                beta_r = parse_float(tokens[t].split("_")[1])

            if t == 6:
                input_type = tokens[t].rstrip().lstrip()

        # if not os.path.exists(true_path):
        #     print(true_path)
        timestamp = os.listdir(os.path.join(root_dir, true_path, "1000_epochs"))[0]
        true_path = os.path.join(root_dir, true_path, "1000_epochs", timestamp)
        output_paths.append(true_path)
        oracles_paths.append(oracle_path)

        config_params.append([
            alfa, beta_b, beta_r, ethics_system, ethics_mode, input_type, smoot_values, #0, 1, 2, 3, 4, 5, 6
            dict["stats"]
            ])

    ea_DNN_unconstrained_business_dict = [] # TODO NOW
    
    with open(os.path.join(eth_compl_save_path, "all_metrics.csv"),"w+") as f:
        header_csv = "System\tEth_system\tInput_type\tSmoothing\tBeta_b\tBeta_r\tPrec_C_0\tRec_C_0\tF1_C_0\tPrec_C_1\t"+\
                "Rec_C_1\tF1_C_1\tF1_mean\tAccuracy\teth_compliance\teth_compl_rel\t"
        for b_f in bias_features:
            add_header = "Bias_FP_"+b_f+"\tBias_FN_"+b_f+"\t"+ "FPR_"+b_f+"\tFPR_oth\tFNR_"+b_f+"\tFNR_oth\t"+\
                "tot_P\ttot_N\ttot_P_"+b_f+"\ttot_P_oth\ttot_N_"+b_f+"\ttot_N_oth\t" + "ppv_"+b_f+"\tppv_oth\t"
            header_csv += add_header
        header_csv += "\n"


        f.write(header_csv)

        for c in tqdm(range(len(output_paths))):

            ethical_signatures = load_ethical_signature(base_path=sign_path, 
                                                    number_of_instances=number_of_instances,
                                                    number_of_values=5,
                                                    number_of_classes=2,
                                                    index_to_drop=index_to_drop)

            business_oracles = y_oracle

            print("Load business/unconstr/constr predicions from path", output_paths[c])

            business_expert_predictions_path = os.path.join(output_paths[c], "business_expert_predictions.txt")
            business_expert_predictions = load_decisions(path=business_expert_predictions_path,
                                                            number_of_instances=number_of_instances)

            ea_DNN_unconstrained_predictions_path = os.path.join(output_paths[c],
                                                                    "ethics_aware_DNN_unconstrained_business_predictions.txt")
            ea_DNN_unconstrained_predictions = load_decisions(path=ea_DNN_unconstrained_predictions_path,
                                                                number_of_instances=number_of_instances)

            ea_DNN_constrained_predictions_path = os.path.join(output_paths[c],
                                                                "ethics_aware_DNN_eth_constrained_business_predictions.txt")
            ea_DNN_constrained_predictions = load_decisions(path=ea_DNN_constrained_predictions_path,
                                                            number_of_instances=number_of_instances)

            business_expert_business_dict = compute_business_metrics(predictions=business_expert_predictions,
                                                                        oracles=business_oracles,
                                                                        dataset_name=dataset_name)
            business_expert_ethics_dict = compute_ethics_metrics(ethical_signatures=ethical_signatures,
                                                                    alfa=config_params[c][0],
                                                                    beta_b=config_params[c][1],
                                                                    beta_r=config_params[c][2],
                                                                    decisions=business_expert_predictions,
                                                                    eth_compliance_score_path=
                                                                    os.path.join(eth_compl_save_path,"be_"+str(config_params[c][0]).replace(".","")+
                                                                    "_"+str(config_params[c][1]).replace(".","")+"_ethcompliance_scores.txt"))

            ea_DNN_unconstrained_business_dict = compute_business_metrics(predictions=ea_DNN_unconstrained_predictions,
                                                                        oracles=business_oracles,
                                                                        dataset_name=dataset_name)
            ea_DNN_unconstrained_ethics_dict = compute_ethics_metrics(ethical_signatures=ethical_signatures,
                                                                        alfa=config_params[c][0],
                                                                        beta_b=config_params[c][1],
                                                                        beta_r=config_params[c][2],
                                                                        decisions=ea_DNN_unconstrained_predictions,
                                                                        eth_compliance_score_path=
                                                                        os.path.join(eth_compl_save_path, "eau_" + str(
                                                                            config_params[c][0]).replace(".","") +"_"+
                                                                                    str(config_params[c][
                                                                                            1]).replace(".","") + "_ethcompliance_scores.txt")
                                                                        )

            ea_DNN_constrained_business_dict = compute_business_metrics(predictions=ea_DNN_constrained_predictions,
                                                                        oracles=business_oracles,
                                                                        dataset_name=dataset_name)
            ea_DNN_constrained_ethics_dict = compute_ethics_metrics(ethical_signatures=ethical_signatures,
                                                                    alfa=config_params[c][0],
                                                                    beta_b=config_params[c][1],
                                                                    beta_r=config_params[c][2],
                                                                    decisions=ea_DNN_constrained_predictions,
                                                                    eth_compliance_score_path=
                                                                    os.path.join(eth_compl_save_path,
                                                                                    "eac_" + str(config_params[c][0]).replace(".","") +
                                                                                    "_"+str(config_params[c][
                                                                                            1]).replace(".","") + "_ethcompliance_scores.txt")
                                                                    )
            
            if smoot_values != "":
                config_params[c][0] = config_params[c][6]

            be_line_string = "Business_expert\t" + str(config_params[c][3]) + "\t" + str(
                config_params[c][5]) + "\t" + str(config_params[c][0]) + \
                            "\t" + str(config_params[c][1]) + "\t" + str(config_params[c][2]) + "\t" + \
                            str(business_expert_business_dict['Precision C 0']) + "\t" + \
                            str(business_expert_business_dict['Recall C 0']) + "\t" + \
                            str(business_expert_business_dict['F1 C 0']) + "\t" + \
                            str(business_expert_business_dict['Precision C 1']) + "\t" + \
                            str(business_expert_business_dict['Recall C 1']) + "\t" + \
                            str(business_expert_business_dict['F1 C 1']) + "\t" + \
                            str(business_expert_business_dict['F1 mean']) + "\t"+ \
                            str(business_expert_business_dict['Accuracy'])+ "\t" + \
                            str(business_expert_ethics_dict['eth_compliance']) + "\t" + \
                            str(var_eth_compliance(eth_compl_mlp, business_expert_ethics_dict['eth_compliance'])) + \
                            "\n"
            f.write(be_line_string)
            eau_line_string = "EA-DNN (unconstrained)\t" + str(config_params[c][3]) + "\t" + str(
                config_params[c][5]) + "\t" + str(config_params[c][0]) + "\t" + \
                            str(config_params[c][1]) + "\t" + str(config_params[c][2]) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['Precision C 0']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['Recall C 0']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['F1 C 0']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['Precision C 1']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['Recall C 1']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['F1 C 1']) + "\t" + \
                            str(ea_DNN_unconstrained_business_dict['F1 mean']) + "\t"+ \
                            str(ea_DNN_unconstrained_business_dict['Accuracy'])+ "\t" + \
                            str(ea_DNN_unconstrained_ethics_dict['eth_compliance']) + "\t" + \
                            str(var_eth_compliance(eth_compl_mlp, ea_DNN_unconstrained_ethics_dict['eth_compliance'])) + \
                            "\n"
            f.write(eau_line_string)

            eac_line_string = "EA-DNN (constrained)\t" + str(config_params[c][3]) + "\t" + str(
                config_params[c][5]) + "\t" + str(config_params[c][0]) + "\t" + \
                            str(config_params[c][1]) + "\t" + str(config_params[c][2]) + "\t" + \
                            str(ea_DNN_constrained_business_dict['Precision C 0']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['Recall C 0']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['F1 C 0']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['Precision C 1']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['Recall C 1']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['F1 C 1']) + "\t" + \
                            str(ea_DNN_constrained_business_dict['F1 mean']) + "\t"+ \
                            str(ea_DNN_constrained_business_dict['Accuracy'])+"\t" + \
                            str(ea_DNN_constrained_ethics_dict['eth_compliance']) + "\t" + \
                            str(var_eth_compliance(eth_compl_mlp, ea_DNN_constrained_ethics_dict['eth_compliance'])) + "\t"
            
            for val in config_params[c][7]:
                eac_line_string += str(val) + "\t"
            eac_line_string += "\n"
            
            f.write(eac_line_string)

    print("saved to " + str(os.path.join(eth_compl_save_path, "all_metrics.csv")))

    with open(os.path.join(eth_compl_save_path, "short_metrics.csv"),"a+") as f: #Baseline
        #f.write("System\tBeta\tF1mean\tAccuracy\teth_compliance\n")
        #sbs_string = System, Beta, F1mean, Accuracy, etchical compliance
        sbs, major_label = simple_baseline_system(business_oracles, dataset_name)
        simple_baseline_string = "BASELINE (all " + str(major_label) + " predictions)" + "\t" + \
            "-" + "\t" + \
            str(sbs['F1 mean']) + "\t" + \
            str(sbs['Accuracy']) + "\t" + \
            str("-")+ "\n"
            
        f.write(simple_baseline_string)

        # Cacola how many 0's and 1's there are. Take the argmax. Measure on this number.
        # Like 60 out of 100. Then the accuracy will be 60% etc.

    with open(os.path.join(eth_compl_save_path, "short_metrics.csv"),"a+") as f: #Unconstrained
        #ENN_unc_string = System, Beta, F1mean, Accuracy, etchical compliance
        ENN_unc_string = "EA-DNN unconstrained\t" +\
            str(config_params[c][1]) + "\t" + \
            str(ea_DNN_unconstrained_business_dict['F1 mean']) + "\t"+ \
            str(ea_DNN_unconstrained_business_dict['Accuracy'])+ "\t" + \
            str(ea_DNN_unconstrained_ethics_dict['eth_compliance']) + "\n"
        f.write(ENN_unc_string)
    
    with open(os.path.join(eth_compl_save_path, "short_metrics.csv"),"a+") as f: ##Constrained
        #ENN_con_string = System, Beta, F1mean, Accuracy, etchical compliance
        ENN_con_string = "EA-DNN constrained\t" +\
            str(config_params[c][1]) + "\t" + \
            str(ea_DNN_constrained_business_dict['F1 mean']) + "\t"+ \
            str(ea_DNN_constrained_business_dict['Accuracy'])+ "\t" + \
            str(ea_DNN_constrained_ethics_dict['eth_compliance']) + "\n"
        f.write(ENN_con_string)
    
    print("saved to " + str(os.path.join(eth_compl_save_path, "short_metrics.csv")))

    # with open(os.path.join(eth_compl_save_path, "short_FP_FN_rate.csv"),"a+") as f:
    #     #ENN_fp_fn_string = System, Beta, FPR_bf, FPR_not_bf, FNR_bf, FNR_not_bf
    #     ENN_fp_fn_string = "EA-DNN constrained" + "\t" + \
    #         str(config_params[c][1]) + "\t" + \
    #         str(config_params[c][9]) + "\t" + \
    #         str(config_params[c][10]) + "\t" + \
    #         str(config_params[c][11]) + "\t" + \
    #         str(config_params[c][12]) + "\n"
    #     f.write(ENN_fp_fn_string)
    
    # print("saved to " + str(os.path.join(eth_compl_save_path, "short_FP_FN_rate.csv")))

def compute_baselines_metrics(root_dir, save_path, sign_path, number_of_instances, dataset_name, y_oracle, index_to_drop, dir_path_name, mlp_fpr_fnr, bias_features):

    print("compute_baselines_metrics")
    dataset = pd.read_csv(
        os.path.join(root_dir,"resources", dataset_name, "data", "enriched_data", dir_path_name, "final_ethics1_ee_cat_gs_with_predictions.csv"), sep= ";")
    predictions = dataset['basic MLP predictions'].values
    golds = np.array(y_oracle)
    eth_signatures = load_ethical_signature(base_path=sign_path, number_of_classes=2, number_of_values=5,
                                            number_of_instances=number_of_instances, index_to_drop=index_to_drop)
    bus_dict_mlp = compute_business_metrics(predictions=predictions, oracles=golds, dataset_name=dataset_name)
    eth_dict_mlp = compute_ethics_metrics(decisions=predictions, ethical_signatures=eth_signatures, alfa=0.1,
                                          beta_b=1, beta_r=1,eth_compliance_score_path=os.path.join(save_path, "mlp_eth_compliance_scores.txt"))
    bus_dict_gs = compute_business_metrics(predictions=golds, oracles=golds, dataset_name=dataset_name)
    eth_dict_gs = compute_ethics_metrics(decisions=golds, ethical_signatures=eth_signatures, alfa=0.1,
                                         beta_r=0.1, beta_b=0.1, eth_compliance_score_path=os.path.join(save_path,"gs_eth_compliance_scores.txt"))
    
    
    with open(os.path.join(save_path, "baseline_metrics.csv"),"w+") as f:
        f.write("System\tEth_system\tInput_type\tSmoothing\tBeta_b\tBeta_r\tPrec_C_0\tRec_C_0\tF1_C_0\tPrec_C_1\t"+
                "Rec_C_1\tF1_C_1\tF1_mean\tAccuracy\teth_compliance\teth_compl_rel\n")
        gs_string = "Gold Standard\tEthics_2\tBusiness_only\t" + str(0.1) + "\t" + str(0.1) + "\t" \
                    + str(0.1) + "\t" + \
                          str(bus_dict_gs['Precision C 0']) + "\t" + \
                          str(bus_dict_gs['Recall C 0']) + "\t" + \
                          str(bus_dict_gs['F1 C 0']) + "\t" + \
                          str(bus_dict_gs['Precision C 1']) + \
                          "\t" + str(bus_dict_gs['Recall C 1']) + "\t" + \
                          str(bus_dict_gs['F1 C 1']) + "\t" + \
                          str(bus_dict_gs['F1 mean']) + "\t"+str(bus_dict_gs['Accuracy'])+\
                          "\t" + str(eth_dict_gs['eth_compliance']) + "\t" +\
                          str("0") + "\n"
        f.write(gs_string)

        mlp_string = "basic MLP\tEthics_2\tBusiness_only\t" + str(0.1) + "\t" + str(0.1) + "\t" \
                    + str(0.1) + "\t" + \
                          str(bus_dict_mlp['Precision C 0']) + "\t" + \
                          str(bus_dict_mlp['Recall C 0']) + "\t" + \
                          str(bus_dict_mlp['F1 C 0']) + "\t" + \
                          str(bus_dict_mlp['Precision C 1']) + \
                          "\t" + str(bus_dict_mlp['Recall C 1']) + "\t" + \
                          str(bus_dict_mlp['F1 C 1']) + "\t" + \
                          str(bus_dict_mlp['F1 mean']) + "\t"+str(bus_dict_mlp['Accuracy'])+\
                          "\t" + str(eth_dict_mlp['eth_compliance']) +\
                          str("0") + "\n"
        f.write(mlp_string)

    with open(os.path.join(save_path, "short_metrics.csv"),"w+") as f:
        f.write("System\tBeta\tF1mean\tAccuracy\teth_compliance\n")
        gs_string = "MLP" + "\t" + \
            "-" + "\t" + \
            str(bus_dict_gs['F1 mean']) + "\t" + \
            str(bus_dict_gs['Accuracy']) + "\t" + \
            str(eth_dict_gs['eth_compliance']) + "\n"
        f.write(gs_string)

    with open(os.path.join(save_path, "short_FP_FN_rate.csv"),"w+") as f:
        mlp_head_string = "System\tBeta\t"
        for b_f in bias_features:
            mlp_head_string += "FPR_"+b_f+"\tFPR Oth\tFNR_"+b_f+"\tFNR Oth\t"
        mlp_head_string += "\n"
        f.write(mlp_head_string)
        
        mlp_fp_fn_string = "MLP" + "\t" + "-" + "\t"
        for val in mlp_fpr_fnr:
            mlp_fp_fn_string += str(val) + "\t"
        mlp_fp_fn_string += "\n"

            
        f.write(mlp_fp_fn_string)

    return eth_dict_mlp['eth_compliance']

    

def compute_ethical_dilemmas(ethical_signature_path, number_of_instances, number_of_decisions, number_of_values, save_path, index_to_drop):

    print("compute_ethical_dilemmas")
    ethical_signatures = load_ethical_signature(base_path=ethical_signature_path,number_of_instances=number_of_instances,
                                                number_of_values=number_of_values,number_of_classes=number_of_decisions,
                                                index_to_drop=index_to_drop)
    ethical_optimum = np.array([0.005,0.015,0.12,0.36,0.5, 0.5, 0.36, 0.12, 0.015, 0.005])
    ethical_minimum = np.array([0.5, 0.36, 0.12, 0.015, 0.005, 0.005,0.015,0.12,0.36,0.5])

    ethical_dilemmas = np.zeros(shape=number_of_instances)

    for i in tqdm(range(number_of_instances)):
        eth_signature = ethical_signatures[i]
        eth_compliances_per_instance = np.zeros(number_of_decisions)
        for d in range(number_of_decisions):
            s_i = 2*d*number_of_values
            f_i = s_i + 2*number_of_values

            decision_signature = eth_signature[s_i:f_i]
            div_wrt_opt = compute_symmetric_kl(pdf_1=decision_signature[:number_of_values],
                                               pdf_2=ethical_optimum[:number_of_values])
            div_wrt_opt += compute_symmetric_kl(pdf_1=decision_signature[number_of_values:],
                                                pdf_2=ethical_optimum[number_of_values:])
            div_wrt_min = compute_symmetric_kl(pdf_1=decision_signature[:number_of_values],
                                               pdf_2=ethical_minimum[:number_of_values])
            div_wrt_min += compute_symmetric_kl(pdf_1=decision_signature[number_of_values:],
                                                pdf_2=ethical_minimum[number_of_values:])
            if(div_wrt_min >= div_wrt_opt):
                eth_compliances_per_instance[d] = 1
        #print(eth_compliances_per_instance)
        if(np.sum(eth_compliances_per_instance) == 0):
            ethical_dilemmas[i] = 1
            # print(str(i) + " is dilemma")


    with open(save_path,"w+") as f:
        for ed in range(ethical_dilemmas.shape[0]):
            f.write(str(ed)+":"+str(ethical_dilemmas[ed])+"\n")

