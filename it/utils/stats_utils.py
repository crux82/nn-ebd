import numpy as np
import matplotlib.pyplot as plt

def FP_FN_rate(TP_bf, TP_not_bf, FP_bf, FP_not_bf, TN_bf, TN_not_bf, FN_bf, FN_not_bf):
    FPR_bf, FPR_not_bf, FNR_bf, FNR_not_bf = 0, 0, 0, 0
    #FP rate
    if FP_bf != 0:
        FPR_bf = FP_bf / (FP_bf + TN_bf)
    if FP_not_bf != 0:
        FPR_not_bf = FP_not_bf / (FP_not_bf + TN_not_bf)

    #FN rate
    if FN_bf != 0:
        FNR_bf = FN_bf / (FN_bf + TP_bf)
    if FN_not_bf != 0:
        FNR_not_bf = FN_not_bf / (FN_not_bf + TP_not_bf)    

    bias_FPR = abs(FPR_bf - FPR_not_bf)
    bias_FNR = abs(FNR_bf - FNR_not_bf)
    # FPR = (FP_bf + FP_not_bf) / (FP_bf + TN_bf + FP_not_bf + TN_not_bf)
    # FNR = (FN_bf + FN_not_bf) / (FN_bf + TP_bf + FN_not_bf + TP_not_bf)

    return [bias_FPR, bias_FNR, FPR_bf, FPR_not_bf, FNR_bf, FNR_not_bf]


def normalize_bias_feature(bias_feature, X_col):
    for row in X_col:
        if bias_feature == "age": 
            if row <= 0.19230769230769232: #less than 33 years old
                row = 1 #feature bias -> I want to see if under 33 are discriminated against
            else:
                row = 0
        
        if bias_feature == "juv_fel_count": 
            if row == 1 or row == 2: #1 or 2 juvenile offences
                row = 1  #feature bias -> I want to see if people with 1 or 2 juvenile offences are discriminated against.
            else:
                row = 0
    
        if bias_feature == "sex__male":
            if row == 0: row = 1
            else: row = 0 #vI want to see if women are discriminated

    return X_col


def compute_3_confusion_matrix(predictions, oracles, is_bias_feature):
    confusion_matrix = np.zeros(shape=(2,2))
    confusion_matrix_bf = np.zeros(shape=(2,2))
    confusion_matrix_not_bf = np.zeros(shape=(2,2))

    for i in range(len(predictions)):
        row_index = predictions[i]
        col_index = oracles[i]
        is_bf = is_bias_feature[i]

        confusion_matrix[int(row_index)][int(col_index)] += 1
        if is_bf == 1:
            confusion_matrix_bf[int(row_index)][int(col_index)] += 1
        if is_bf == 0:
            confusion_matrix_not_bf[int(row_index)][int(col_index)] += 1

    return confusion_matrix, confusion_matrix_bf, confusion_matrix_not_bf


def count_predictions(y_pred_list, y_test_fold, X_test_fold, bias_feature, dataset_name):
    total_positive, total_negative = 0, 0
    total_bf_positive, total_bf_negative = 0, 0
    total_not_bf_positive, total_not_bf_negative = 0, 0

    is_bias_feature = normalize_bias_feature(bias_feature, X_test_fold[bias_feature].copy())

    confusion_matrix, confusion_matrix_bf, confusion_matrix_not_bf = compute_3_confusion_matrix(y_pred_list, y_test_fold, is_bias_feature.values.tolist())

    if dataset_name == "german_credit":
        tp, tp_bf, tp_not_bf = confusion_matrix[0,0], confusion_matrix_bf[0,0], confusion_matrix_not_bf[0,0]
        fp, fp_bf, fp_not_bf = confusion_matrix[0,1], confusion_matrix_bf[0,1], confusion_matrix_not_bf[0,1]
        fn, fn_bf, fn_not_bf = confusion_matrix[1,0], confusion_matrix_bf[1,0], confusion_matrix_not_bf[1,0]
        tn, tn_bf, tn_not_bf = confusion_matrix[1,1], confusion_matrix_bf[1,1], confusion_matrix_not_bf[1,1]

    else:
        tp, tp_bf, tp_not_bf = confusion_matrix[1,1], confusion_matrix_bf[1,1], confusion_matrix_not_bf[1,1]
        fp, fp_bf, fp_not_bf = confusion_matrix[1,0], confusion_matrix_bf[1,0], confusion_matrix_not_bf[1,0]
        fn, fn_bf, fn_not_bf = confusion_matrix[0,1], confusion_matrix_bf[0,1], confusion_matrix_not_bf[0,1]
        tn, tn_bf, tn_not_bf = confusion_matrix[0,0], confusion_matrix_bf[0,0], confusion_matrix_not_bf[0,0]

    # print("tp, tp_bf, tp_not_bf", tp, tp_bf, tp_not_bf)
    # print("fp, fp_bf, fp_not_bf", fp, fp_bf, fp_not_bf)
    # print("fn, fn_bf, fn_not_bf", fn, fn_bf, fn_not_bf)
    # print("tn, tn_bf, tn_not_bf", tn, tn_bf, tn_not_bf)

    # TP / (TP + FP)
    ppv_bf =  tp_bf / (tp_bf + fp_bf)
    ppv_not_bf = tp_not_bf / (tp_not_bf + fp_not_bf)

    rate_list = FP_FN_rate(tp_bf, tp_not_bf, fp_bf, fp_not_bf, tn_bf, tn_not_bf, fn_bf, fn_not_bf)

    total_positive = tp + fp
    total_negative = tn + fn

    total_bf_positive = tp_bf + fp_bf
    total_not_bf_positive = tp_not_bf + fp_not_bf

    total_bf_negative = tn_bf + fn_bf
    total_not_bf_negative = tn_not_bf + fn_not_bf

    total_pos_neg_list = [
            total_positive, total_negative, 
            total_bf_positive, total_not_bf_positive, 
            total_bf_negative, total_not_bf_negative, 
            ppv_bf, ppv_not_bf]

    return rate_list, total_pos_neg_list

def plot_bias_feature(FPR, FPR_bf, FPR_not_bf, FNR, FNR_bf, FNR_not_bf, bias_feature):

    x1 = ["Total FP", "FP "+bias_feature[:3], "FP NO_"+bias_feature[:3], "Total FN", "FN "+bias_feature[:3], "FN NO_"+bias_feature[:3]]
    y1 = [FPR, FPR_bf, FPR_not_bf, 0, 0, 0]

    x2 = ["Total FP", "FP "+bias_feature[:3], "FP NO_"+bias_feature[:3], "Total FN", "FN "+bias_feature[:3], "FN NO_"+bias_feature[:3]]
    y2 = [0, 0, 0, FNR, FNR_bf, FNR_not_bf]

    plt.bar(x1, y1, label="False poitive rate", color='b')
    plt.bar(x2, y2, label="False negative rate", color='g')
    plt.plot()

    plt.xlabel("")
    plt.ylabel("")
    plt.title("FP & FN rate "+bias_feature)
    plt.legend()
    plt.show()


def statistiche_popolazione(preds_mlp, preds_ethic, X_reduced, bias_feature):
    map_list = X_reduced[bias_feature]  # maps the rows affected by the bias_feature. 
                                        # For example it is = 1 when an individual is black, in the case of race as bias_feature

    mlp_afr_recidivo, mlp_afr_non_recidivo, mlp_nonafr_recidivio, mlp_nonafr_non_recidivio = 0, 0, 0, 0
    eth_afr_recidivo, eth_afr_non_recidivo, eth_nonafr_recidivio, eth_nonafr_non_recidivio = 0, 0, 0, 0
    change_afr, change_not_afr = 0, 0 #how many decisions have changed between blacks and whites (respectively)

    assert len(preds_mlp) == len(preds_ethic), "different lengths"
    assert len(preds_ethic) == len(map_list), "different lengths"
    print("Num pred mlp:", len(preds_mlp)," Num pred eth:", len(preds_ethic), " Num map:", len(map_list))

    for p_mlp, p_ethic, m in zip(preds_mlp, preds_ethic, map_list):
        if m == 1: #e.g. if the individual is black (bias_feature)
            if p_mlp != p_ethic:
                change_afr += 1
        
            if p_mlp == 1:
                mlp_afr_recidivo += 1
            else:
                mlp_afr_non_recidivo += 1

            if p_ethic == 1:
                eth_afr_recidivo += 1
            else:
                eth_afr_non_recidivo += 1

        if m == 0: #e.g. if the individual is white (bias_feature)
            if p_mlp != p_ethic:
                change_not_afr += 1

            if p_mlp == 1:
                mlp_nonafr_recidivio += 1
            else:
                mlp_nonafr_non_recidivio += 1
            
            if p_ethic == 1:
                eth_nonafr_recidivio += 1
            else:
                eth_nonafr_non_recidivio += 1


    print("Population statistics")
    print("Unlike the predictions of the MLP, with the ethical network these decisions have changed:", "Neri->", change_afr, "White->", change_not_afr)
    print("\n MLP -> mlp_afr_recidivo, mlp_afr_non_recidivo, mlp_nonafr_recidivio, mlp_nonafr_non_recidivio", mlp_afr_recidivo, mlp_afr_non_recidivo, mlp_nonafr_recidivio, mlp_nonafr_non_recidivio)
    print("\n Eth -> eth_afr_recidivo, eth_afr_non_recidivo, eth_nonafr_recidivio, eth_nonafr_non_recidivio", eth_afr_recidivo, eth_afr_non_recidivo, eth_nonafr_recidivio, eth_nonafr_non_recidivio)


def calcola_differenza_probabilita(init_list):
    """
    Probs difference calculation:
    input init_list_var_test or reduced_init_list_var_test (for reduced oracle).

    Function used for Debugging
    """
    prob_oracolo_u = np.array(init_list[5])
    prob_oracolo_c = np.array(init_list[6])

    list_diff_u = [abs(elem[0]-elem[1]) for elem in prob_oracolo_u]
    list_diff_c = [abs(elem[0]-elem[1]) for elem in prob_oracolo_c]

    list_somma_u = [abs(elem[0]+elem[1]) for elem in prob_oracolo_u]
    list_somma_c = [abs(elem[0]+elem[1]) for elem in prob_oracolo_c]

    diff_u = sum(list_diff_u)/len(list_diff_u)
    diff_c = sum(list_diff_c)/len(list_diff_c)

    somma_u = sum(list_somma_u)/len(list_somma_u)
    somma_c = sum(list_somma_c)/len(list_somma_c)

    print("mean of the difference in probability for the unconstrained policy", diff_u)
    print("mean of the difference in probability for the constrained policy", diff_c)
    
    print("Average of the sum of probabilities for the unconstrained policy", somma_u)
    print("average of the sum of probabilities for the constrained policy", somma_c)
