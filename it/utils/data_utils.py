import numpy as np
import pandas as pd
import os
import scipy.stats as sts
from pathlib import Path
import shutil
from tqdm import tqdm
#from tqdm.notebook import tqdm
import random

def init_results_dir(experiment_path, sign_path):

    """
    Creates sub-folders of the results folder.
    """
    Path(os.path.join(experiment_path, "results")).mkdir(parents=True, exist_ok=True)

    #copy the ethical signature in results
    #sm_str = str(exp_configs[0]['alfa']).replace(".","")+"_smoothing"
    #tw_str = str(exp_configs[0]['beta_b']).replace(".","") + "_" + str(exp_configs[0]['beta_r']).replace(".","") + "_tweaking"
    #oes_path_original = os.path.join(experiment_path, exp_configs[0]['ethics_system'], exp_configs[0]['ethics_mode'],  sm_str, tw_str, "overall_ethical_signature.txt")
    
    sign_path = sign_path.split("tweaking")[0] + "tweaking"
    oes_path_original = os.path.join(sign_path, "overall_ethical_signature.txt")
    oes_path_target = os.path.join(experiment_path, "results", "overall_ethical_signature.txt")
    
    shutil.copyfile(oes_path_original, oes_path_target)


def load_ethical_signature(base_path,number_of_values, number_of_classes,number_of_instances, index_to_drop):
    ethical_signature = np.zeros(shape=(number_of_instances,2*number_of_classes*number_of_values))

    if "result" not in base_path:
        base_path = base_path.split("tweaking")[0] + "tweaking"

    with open(os.path.join(base_path,"overall_ethical_signature.txt"), "r+") as f:
        #print("OK aperto")
        i_reduced = 0 #to iterate on the ethical signature of reduced size compared to the original
        lines = f.readlines()
        for i in range(len(lines)):
            if i not in index_to_drop:
                line = lines[i]
                vector_tokens = line.split(":")[1].lstrip().rstrip().split(",")
                for v in range(len(vector_tokens)):
                    ethical_signature[i_reduced][v] = vector_tokens[v]
                i_reduced += 1
    return ethical_signature

def symmetric_kl(distr_1,distr_2):
    if np.abs(np.sum(distr_1)-1) > 0.001:
        print("Input 1 is not an acceptable approximation of a probability distribution as it sum to "+
              str(np.sum(distr_1)))
        raise ValueError
    if np.abs(np.sum(distr_2)-1) > 0.001:
        print("Input 2 is not an acceptable approximation of a probability distribution as it sum to "+
              str(np.sum(distr_2)))
        raise ValueError

    return np.divide(sts.entropy(distr_1,distr_2)+sts.entropy(distr_2,distr_1),2)


def load_decisions(path, number_of_instances):
    print("Loading decisions from -> "+str(path))
    decisions = np.zeros(shape=number_of_instances)

    with open(path, "r+") as f:
        lines = f.readlines()

        for i in range(number_of_instances):
            line = lines[i]
            value = line.split(":")[1].rstrip().lstrip()
            decisions[i] = int(value)
    return decisions


def check_probability_vector(vect):
    if np.min(vect) < 0:
        return -1
    if np.abs(np.sum(vect)-1)>0.001:
        return -2
    return 0


def parse_float(input_string):
    if "best" in input_string:
        return input_string

    new_string = ""
    if input_string.startswith("0"):
        for c in range(len(input_string)):
            if c == 0:
                new_string += input_string[c]+"."
            else:
                new_string += input_string[c]
            #print(new_string)
    else:
        for c in range(len(input_string)):
            if c < len(input_string)-1:
                new_string += input_string[c]
            else:
                new_string += "."+input_string[c]
    return float(new_string)


def binary_laplace_smoothing(x,alfa):
    number_of_classes = x.shape[1]
    smoothed_vect = np.zeros(shape=x.shape)
    for i in range(x.shape[0]):
        smoothed_vect[i] = np.divide(x[i]+alfa,1+alfa*number_of_classes) #formula as in paper
    return smoothed_vect


def load_dataset(root_path, dataset_name):

    dataset_path = os.path.join(root_path, 'resources', dataset_name, 'standard_' + dataset_name + '.csv')
    dataset_df = pd.read_csv(dataset_path, engine="python", encoding="utf-8", error_bad_lines=False, sep=";")

    if dataset_name == "compas":
        dataset_df = dataset_df.rename(columns={'African-American': 'AfricanAmerican', 'Native American': 'NativeAmerican'})
    
    dataset_size = dataset_df.shape[0]
    
    return dataset_df, dataset_size


def split_features_target(dataset_df, dataset_name):
    if dataset_name == "compas":
        X = dataset_df.iloc[:, 2:]
        y = dataset_df.iloc[:, 1]

    else:
        X = dataset_df.iloc[:, 1:]
        y = dataset_df.iloc[:, 0]
    
    return X, y



def get_alfa_beta_values_list(values):
    alfa_values, beta_values = [], []
    for elem in values:
        alfa_values.append(elem[0])
        beta_values.append(elem[1])
    alfa_values = sorted(list(set(alfa_values)))
    beta_values = sorted(list(set(beta_values)))

    return alfa_values, beta_values




def create_clustered_dataset(X_test_folds):

    pd.options.mode.chained_assignment = None  #default='warn'

    X_test_clustered_fold = X_test_folds.copy()
    #Per ogni fold
    for fold_df in tqdm(range(len(X_test_clustered_fold))):
        
        columns = X_test_clustered_fold[fold_df].columns
        df = pd.DataFrame(columns=columns)
        df = df.fillna(0) #replace NaN with zero
        index_X = X_test_clustered_fold[fold_df].index

        for idx in index_X:

            m_row = X_test_clustered_fold[fold_df].loc[idx]

            m_row['AfricanAmerican'], m_row['Asian'], m_row['Caucasian'] = 1, 0, 0
            m_row['Hispanic'], m_row['NativeAmerican'], m_row['Other'] = 0, 0, 0
            df = df.append(m_row, ignore_index=False)

            m_row['AfricanAmerican'], m_row['Asian'] = 0, 1
            df = df.append(m_row, ignore_index=False)

            m_row['Asian'],  m_row['Caucasian'] = 0, 1
            df = df.append(m_row, ignore_index=False)

            m_row['Caucasian'], m_row['Hispanic'] = 0, 1
            df = df.append(m_row, ignore_index=False)

            m_row['Hispanic'], m_row['NativeAmerican'] = 0, 1
            df = df.append(m_row, ignore_index=False)

            m_row['NativeAmerican'], m_row['Other'] = 0, 1
            df = df.append(m_row, ignore_index=False)

        X_test_clustered_fold[fold_df] = df.copy() #df_temp_base.append(concat_df, ignore_index=True) #with ignore_index=True the original index is lost
    return X_test_clustered_fold



def data_augmentation(dataset_df, train_index, AUG_OPTIONS, verbose = False):
    """
    1) Get an index of all rows that are to be duplicated. This is different for each dataset.
    2) I call this index augment_index.
    3) If there are indices in train_index contained in augment_index, then duplicate them.
    4) Do the same for y_train.
    """
    train_index = train_index.tolist()
    print("Data augmentation...")

    if verbose:
        print("Data augmentation info: Previous dataset size ->", len(train_index))

    for config in AUG_OPTIONS:
        AUG_FEATURE, AUG_MUL_FACTOR, AUG_VALUE_FEATURE = config[0], config[1], config[2]
        AUG_MUL_FACTOR -= 1 #Since I do not multiply but extend the index of this factor, then if I want an x3 multiplication, then I have to extend the list by an x2

        augment_index = dataset_df.index[dataset_df[AUG_FEATURE] == AUG_VALUE_FEATURE].tolist()
        temp_train_index = train_index.copy()

        for i in temp_train_index:
            if i in augment_index:
                add_list = [i]*AUG_MUL_FACTOR
                train_index.extend(add_list)

    if verbose:
        print("Actual fold dataset size ->", len(train_index))
    random.shuffle(train_index)
    train_index = np.asarray(train_index)

    return train_index