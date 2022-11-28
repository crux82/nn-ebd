import os
import numpy as np
import itertools
from sklearn.metrics import accuracy_score

def write_vector(vector,writer):
    for v in range(vector.shape[0]-1):
        writer.write(str(vector[v])+",")
    writer.write(str(vector[-1])+"\n")

def normalize_predictions(l):
    mul_factor= 1/sum(l)
    output_list = [i * mul_factor for i in l]

    return output_list

def get_constrained_scores(decisions, number_of_classes, number_of_values, benefit_indexes, risk_indexes, is_test_cluster = False):

    # Remember:
    # benefit_indexes = [3, 4]
    # risk_indexes = [0, 1]

    v = number_of_values #5
    v_sq = v*v #25
    conditioned_decisions = np.zeros(shape=(len(decisions),number_of_classes)) #100*2

    index_tuple = list(itertools.product(benefit_indexes,risk_indexes)) #[3, 4]*[0, 1] = [(3, 0), (3, 1), (4, 0), (4, 1)]

    # I use the Ethics-Constrained (EC) policy, where I only do the summations of probabilities that correspond to
    # high, very high benefit and low and very low risk (very low = 0, low = 1, etc.)

    for i in range(len(decisions)):
        for d in range(number_of_classes): #2*100 times
            prob_sum = 0
            for tuple in index_tuple: # [(3, 0), (3, 1), (4, 0), (4, 1)]
                j = tuple[0]
                k = tuple[1]
                prob_sum += decisions[i][d*v_sq+v*j+k]
            conditioned_decisions[i][d] = prob_sum #3. also get the difference back
    
    return conditioned_decisions


def get_upsilon_business_scores(upsilon_predictions, number_of_classes, is_test_cluster = False):
        split_length = 50//number_of_classes
        business_predictions = np.zeros(shape=(upsilon_predictions.shape[0],number_of_classes))
        for i in range(upsilon_predictions.shape[0]):
            for j in range(number_of_classes):
                s_j = j*split_length
                f_j = (j+1)*split_length
                tmp_slice = upsilon_predictions[i][s_j:f_j]
                sum_j = np.sum(tmp_slice)
                business_predictions[i][j] = sum_j
        
        return business_predictions


def init_policyEC_EU(X):
    """
    Initialises the variables for the two policies.

    number_of_instances is the number of rows, i.e. the number of instances. For example 1000
    """

    number_of_classes = 2 #number of classes: e.g. whether I grant the loan or not
    number_of_values = 5
    number_of_instances = X.values.shape[0] #number of rows, i.e. the number of instances. For example 1000
    benefits_j = [3, 4]
    risks_k = [0, 1]

    business_expert_business_scores = np.zeros(shape=(number_of_instances, number_of_classes)) #returns an array filled with 0... 1000*2
    ethics_aware_DNN_unconstrained_business_scores = np.zeros(shape=(number_of_instances, number_of_classes)) #unconstrained, 1000*2
    ethics_aware_DNN_eth_constrained_business_scores = np.zeros(shape=(number_of_instances, number_of_classes)) #constrained, 1000*2
    
    return [number_of_classes, number_of_values, benefits_j, risks_k, business_expert_business_scores, 
            ethics_aware_DNN_unconstrained_business_scores, ethics_aware_DNN_eth_constrained_business_scores]

def run_policyEC_EU(predictions_u, predictions_c, test_inds, init_list_var, is_test_cluster = False):
    """
    Policy calculation. Calculates three scores:
    - Business Expert
    - EA unconstrained
    - EA constrained

    """

    #retrieving variables
    number_of_classes = init_list_var[0]
    number_of_values = init_list_var[1]
    benefits_j = init_list_var[2]
    risks_k = init_list_var[3]
    business_expert_business_scores = init_list_var[4]
    ethics_aware_DNN_unconstrained_business_scores = init_list_var[5]
    ethics_aware_DNN_eth_constrained_business_scores = init_list_var[6]
    test_size = len(test_inds)


    predictions_u[2] = np.array([normalize_predictions(r) for r in predictions_u[2]])
    predictions_c[2] = np.array([normalize_predictions(r) for r in predictions_c[2]])


    # I use the Ethics-Constrained (EC) policy, where I only do the summations of probabilities that correspond to
    # high, very high benefit and low and very low risk. Returns array 100 rows and 2 columns
    tmp_ethics_aware_DNN_eth_constrained_business_scores = get_constrained_scores(predictions_c[2],
                                                                                  number_of_classes,
                                                                                  number_of_values, benefits_j,
                                                                                  risks_k,
                                                                                  is_test_cluster)

    # Here we use the Ethics-Unconstrained (EU) policy where we add up all the probabilities.
    # Returns array 100 rows and 2 columns
    tmp_ethics_aware_DNN_unconstrained_business_scores = get_upsilon_business_scores(predictions_u[2], 2, is_test_cluster)

    for i in range(test_size):
        # Populate these three arrays as we go along with the folds by assigning the values of the predictions 
        # to their respective vectors

        business_expert_business_scores[test_inds[i]] = predictions_u[0][i]
        ethics_aware_DNN_unconstrained_business_scores[test_inds[i]] = tmp_ethics_aware_DNN_unconstrained_business_scores[i]
        ethics_aware_DNN_eth_constrained_business_scores[test_inds[i]] = tmp_ethics_aware_DNN_eth_constrained_business_scores[i]


    init_list_var[4] = business_expert_business_scores
    init_list_var[5] = ethics_aware_DNN_unconstrained_business_scores
    init_list_var[6] = ethics_aware_DNN_eth_constrained_business_scores

    return init_list_var

def save_policyEC_EU(output_path, save_var):

    # SAVE STUFFS
    business_expert_business_scores = save_var[4]
    ethics_aware_DNN_unconstrained_business_scores = save_var[5]
    ethics_aware_DNN_eth_constrained_business_scores = save_var[6]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Path didn't exist, created")
    print("Savin to {:}".format(output_path))

    f_o_p = open(os.path.join(output_path, "business_expert_predictions.txt"), "w+")
    f_ub_p = open(os.path.join(output_path, "ethics_aware_DNN_unconstrained_business_predictions.txt"), "w+")
    f_ec_p = open(os.path.join(output_path, "ethics_aware_DNN_eth_constrained_business_predictions.txt"), "w+")

    #vari write
    
    for i in range(business_expert_business_scores.shape[0]):
        f_o_p.write(str(i) + ":" + str(np.argmax(business_expert_business_scores[i])) + "\n")
        f_ub_p.write(str(i) + ":" + str(np.argmax(ethics_aware_DNN_unconstrained_business_scores[i])) + "\n")
        f_ec_p.write(str(i) + ":" + str(np.argmax(ethics_aware_DNN_eth_constrained_business_scores[i])) + "\n")

    f_o_p.close()
    f_ub_p.close()
    f_ec_p.close()


def compute_accuracies(output_BE_list, output_EE_list, output_EA_list, y_BE_one_fold, init_list_var):
    """
    Calculates the accuracy for the train/val phase of the ethical network for both policies
    """
    predictions_train = [np.array(output_BE_list), np.array(output_EE_list), np.array(output_EA_list)]

    train_inds =  y_BE_one_fold.index.tolist()
    init_list_var = run_policyEC_EU(
        # I quote the same line twice below for code compatibility (otherwise I would have to write a slightly different function). 
        # Except in the test, train and val these two are the same, there is no need to calculate a different result. In the test instead
        # I have 2 models (constr. and uncontrained) which were the best as a result of the validation phase. 
        # Here there is only one model in train and val, so the two rows are identical. However, of course, the results are different
        # because of what the function looks like inside

        predictions_u = predictions_train, 
        predictions_c = predictions_train, 
        test_inds = train_inds,
        init_list_var = init_list_var
        )
    
    unconstrained_scores = init_list_var[5]
    constrained_scores = init_list_var[6]

    pred_unconstrained, pred_constrained = [], []
    for i in train_inds:
        pred_unconstrained.append(np.argmax(unconstrained_scores[i]))
        pred_constrained.append(np.argmax(constrained_scores[i]))

    #calculates accuracy taking into account only the index of this fold
    acc_unconstr = accuracy_score(y_BE_one_fold.tolist(), pred_unconstrained)
    acc_constr = accuracy_score(y_BE_one_fold.tolist() , pred_constrained)

    return acc_unconstr, acc_constr

def run_policyEC_EU_Lime(predictions_lime, type_policy):

    number_of_classes = 2 #classes: e.g. whether I grant the loan or not
    number_of_values = 5
    number_of_instances = predictions_lime.shape[0] #number of rows, i.e. the number of instances. For example 5000 for files
    benefits_j = [3, 4]
    risks_k = [0, 1]

    if type_policy == "constrained":
        ethics_aware_DNN_eth_business_scores = get_constrained_scores(predictions_lime,
                                                                                  number_of_classes,
                                                                                  number_of_values, benefits_j,
                                                                                  risks_k)
    elif type_policy == "unconstrained":
        ethics_aware_DNN_eth_business_scores = get_upsilon_business_scores(predictions_lime, 2)


    return ethics_aware_DNN_eth_business_scores
