import warnings
from datetime import datetime
from it.utils.data_utils import get_alfa_beta_values_list

############################
# Main configuration values
############################
"""
- root_path: string project root path
- FOLDS: (int) number k of folds for k-CrossValidation
- DATA_AUGMENT: Boolean variable to activate augmentation or not
- AUG_OPTIONS: see below in "Dataset configuration" section
- dataset_name: string with dataset name
- active_ef: list of active TMs
- reduced_oracle_actived: Boolean variable used ONLY FOR COMPAS to delete some biased instances concerning African-Americans.
- enable_early_stop: variabile booleana per attivare o meno il meccanismo dell'early stop...
- PATIENCE_VALUE: ...limit value beyond which the early stop must stop if it does not improve the accuracy on the dev set
- ETH_NET_EPOCHS: max epochs of the ethical neural network
- MLP_EPOCHS: max epochs of the MLP
"""
root_path = "C:/Users/luca/Documents/Tesi magistrale/ENN from git/enn"
#SEED =  7763456 #5) 92578432 #4)400423 #3) 7763456 #2) 34204329 #1) 9647566
FOLDS = 10
DATA_AUGMENT, AUG_OPTIONS = True, [] #AUG_MUL_FACTOR, AUG_FEATURE, AUG_VALUE_FEATURE

dataset_name = "compas"
active_ef = ['sexTM']
reduced_oracle_actived = False
enable_early_stop, PATIENCE_VALUE,  = True, 5
ETH_NET_EPOCHS, MLP_EPOCHS = 2, 2 #50, 50



############################
# Ethics configuration values
############################
"""
- values: ethical configuration list. [smoothing (alpha), tweaking benefits (beta_b), tweaking risks (beta_r)]. 
            Beta benefits and beta risks are assumed to be equal. Warning. The more configurations there are, 
            the more ram is needed when calculating ethical signatures.
"""
# values = [
#     [0.1, 0.001, 0.001], [0.1, 0.01, 0.01], [0.1, 0.03, 0.03], [0.1, 0.05, 0.05], [0.1, 0.07, 0.07], [0.1, 0.1, 0.1], [0.1, 0.12, 0.12], [0.1, 0.14, 0.14],
#     [0.3, 0.001, 0.001], [0.3, 0.01, 0.01], [0.3, 0.03, 0.03], [0.3, 0.05, 0.05], [0.3, 0.07, 0.07], [0.3, 0.1, 0.1], [0.3, 0.12, 0.12], [0.3, 0.14, 0.14],
#     [0.5, 0.001, 0.001], [0.5, 0.01, 0.01], [0.5, 0.03, 0.03], [0.5, 0.05, 0.05], [0.5, 0.07, 0.07], [0.5, 0.1, 0.1], [0.5, 0.12, 0.12], [0.5, 0.14, 0.14]
# ]

values = [
    [0.1, 0.001, 0.001]
]

#credit
# values = [
# [0.1, 0.001, 0.001], [0.1, 0.01, 0.01], [0.1, 0.03, 0.03], [0.1, 0.05, 0.05], [0.1, 0.08, 0.08],
# # [0.3, 0.001, 0.001], [0.3, 0.01, 0.01], [0.3, 0.03, 0.03], [0.3, 0.05, 0.05], [0.3, 0.08, 0.08],
# # [0.5, 0.001, 0.001], [0.5, 0.01, 0.01], [0.5, 0.03, 0.03], [0.5, 0.05, 0.05], [0.5, 0.08, 0.08]
# ]


############################
# Dataset configuration: DON'T EDIT
############################
"""
- do_validation: Boolean variable to use or not the validation set. By default, it is always active.

- dataset_y_name: a string with the name of the target feature. WARNING: it must be the first column of the dataset.
- bias_features: a list with the name of features in the dataset affected by bias
- DATA_AUGMENT: Boolean variable to activate augmentation or not
- BATCH_SIZE_MLP: Batch size of Multi Layer Perceptron
- BATCH_SIZE_ENN: Batch size of Ethical Neural Network
- AUG_OPTIONS: a list of list of variables --> [
                                                [AUG_FEATURE (augmentation feature), 
                                                AUG_MUL_FACTOR (multiplication factor), 
                                                AUG_VALUE_FEATURE (augment only features with this value)], 
                                                ...,
                                                ...]
- 
"""


do_validation = True

if dataset_name == "compas":
    dataset_y_name = "is_recid"
    bias_features = ['AfricanAmerican', 'age', 'is_male']
    DATA_AUGMENT = False
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256
    
    if not set(active_ef).issubset(['raceTM', 'ageTM', 'age2TM', 'sexTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)
        
if dataset_name == "german_credit":
    #do_validation = False
    dataset_y_name = "default"
    bias_features = ['sex__male', 'foreign_worker__yes']
    AUG_OPTIONS = [["default", 2, 1]] #AUG_FEATURE, AUG_MUL_FACTOR, AUG_VALUE_FEATURE
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 64, 64
    PATIENCE_VALUE = 10
    ETH_NET_EPOCHS, MLP_EPOCHS = 100, 100

    if not set(active_ef).issubset(['motherhoodFostering', 'culturalInclusiveness', 'sexGermanTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

if dataset_name == "adult" or dataset_name == "adult_ridotto":
    dataset_y_name = "y" # feature target. WARNING: it must be the first column of the dataset
    bias_features = ['is_male', 'White', 'Asian-Pac-Islander', 'Black', 'Amer-Indian-Eskimo', 'Other'] 
    AUG_OPTIONS = [["Amer-Indian-Eskimo", 10, 1], ["Asian-Pac-Islander", 3, 1], ["Other", 10, 1]]
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256

    if not set(active_ef).issubset(['sexTM', 'sex2TM', 'raceAdultTM', 'sex2alternativeTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

if dataset_name == "KDD":
    dataset_y_name = "income_50k" # feature target. WARNING: it must be the first column of the dataset
    bias_features = ['is_male', 'Black'] # Name of features in the dataset affected by bias
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256

    if not set(active_ef).issubset(['sexTM', 'KDD_raceTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

if dataset_name == "credit_card":
    dataset_y_name = "default" # feature target. WARNING: it must be the first column of the dataset
    bias_features = ['is_male'] # Name of features in the dataset affected by bias
    DATA_AUGMENT = False
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256

    if not set(active_ef).issubset(['sexTM', 'sex2TM', 'sex2alternativeTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

if dataset_name == "diabetic":
    dataset_y_name = "readmitted" # feature target. WARNING: it must be the first column of the dataset
    bias_features = ['is_male']
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256

    if not set(active_ef).issubset(['sexTM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

if dataset_name == "law_school":
    dataset_y_name = "pass_bar" # feature target. WARNING: it must be the first column of the dataset
    bias_features = ['is_male', "White"]
    AUG_OPTIONS = [["Black", 2, 1], ["Asian", 3, 1], ["Hisp", 3, 1], ["Other", 6, 1]]
    BATCH_SIZE_MLP, BATCH_SIZE_ENN = 256, 256

    if not set(active_ef).issubset(['sexTM', 'sex2alternativeTM', 'raceLawSchTM', 'raceLawSch2TM', 'mildTM']):
        raise ValueError('TM', active_ef, 'is not valid for', dataset_name)

# consistency check
if reduced_oracle_actived and dataset_name != "compas":
    warnings.warn("The ubniased dataset is only usable for COMPAS, so it will not be considered for this run.")
    reduced_oracle_actived = False

# other var (do not edit them)
today = datetime.today().strftime("%d_%m_%Y__%H-%M-%S")
alfa_values, beta_values = get_alfa_beta_values_list(values)

# ethics var (do not edit them)
number_of_decisions = 2
number_of_eth_values = 5

