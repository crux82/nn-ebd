import imp
from sklearn.model_selection import StratifiedKFold, train_test_split
from it.utils.data_utils import data_augmentation


def KFold_CrossValidate_MLP(dataset_df, X, y, FOLDS, SEED, DATA_AUGMENT, AUG_OPTIONS, test_size=0.1, train_size=0.9, random_state = 0):
    X_train_fold = []
    X_val_fold = []
    X_test_fold = []

    y_train_fold = []
    y_val_fold = []
    y_test_fold = []

    #Split data in "Test | (rest)" folds and then rest folds in "Train | Validation"
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for rest_index, test_index in skf.split(X,y):

        if DATA_AUGMENT:
            rest_index = data_augmentation(dataset_df, rest_index, AUG_OPTIONS)

        #X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        X_rest, y_rest = X.iloc[rest_index], y.iloc[rest_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_rest, y_rest, test_size=test_size, train_size=train_size, stratify=y_rest, random_state = random_state)

        X_train_fold.append(X_train)
        X_val_fold.append(X_val)
        X_test_fold.append(X_test)

        y_train_fold.append(y_train)
        y_val_fold.append(y_val)
        y_test_fold.append(y_test)

    return X_train_fold, X_val_fold, X_test_fold, y_train_fold, y_val_fold, y_test_fold


def KFold_CrossValidate_ENN(dataset_df, X, business_oracle, pd_rec_oracle, pd_revi_oracle_list, FOLDS, SEED, DATA_AUGMENT, 
                            AUG_OPTIONS, test_size=0.1, train_size=0.9, random_state = 0):
    X_train_fold, y_train_BE_fold, y_train_EE_fold = [], [], []
    X_val_fold, y_val_BE_fold, y_val_EE_fold = [], [], []
    X_test_fold, y_test_BE_fold = [], []

    #Split data in "Test | (rest)" folds and then rest folds in "Train | Validation"
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for rest_index, test_index in skf.split(X, business_oracle):

        if DATA_AUGMENT:
            rest_index = data_augmentation(dataset_df, rest_index, AUG_OPTIONS)

        X_rest, y_rest_BE, y_rest_EE = X.iloc[rest_index], business_oracle.iloc[rest_index], pd_rec_oracle.iloc[rest_index]
        X_test, y_test_BE = X.iloc[test_index], business_oracle.iloc[test_index]


        X_train, X_val, y_train_BE, y_val_BE, y_train_EE, y_val_EE = train_test_split(X_rest, y_rest_BE, y_rest_EE, test_size=test_size, train_size=train_size, stratify=y_rest_BE, random_state = random_state)

        X_train_fold.append(X_train)
        y_train_BE_fold.append(y_train_BE)
        y_train_EE_fold.append(y_train_EE)

        X_val_fold.append(X_val)
        y_val_BE_fold.append(y_val_BE)
        y_val_EE_fold.append(y_val_EE)

        X_test_fold.append(X_test)
        y_test_BE_fold.append(y_test_BE)

    
    #Now set folds and configurations Ethics Aware data
    y_train_EA_fold_config, y_val_EA_fold_config = [], []

    for pd_revi_oracle in pd_revi_oracle_list:
        y_train_EA_fold, y_val_EA_fold, y_test_EA_fold = [], [], []
        for iter in range(FOLDS):
            train_I, val_I, test_I = y_train_BE_fold[iter].index, y_val_BE_fold[iter].index, y_test_BE_fold[iter].index
            
            y_train_EA_fold.append(pd_revi_oracle.loc[train_I])
            y_val_EA_fold.append(pd_revi_oracle.loc[val_I])
            y_test_EA_fold.append(pd_revi_oracle.loc[test_I])

        y_train_EA_fold_config.append(y_train_EA_fold) #list of 10 folds for each configuration
        y_val_EA_fold_config.append(y_val_EA_fold)

    return X_train_fold, y_train_BE_fold, y_train_EE_fold, X_val_fold, y_val_BE_fold, y_val_EE_fold, X_test_fold, y_test_BE_fold, y_train_EA_fold_config, y_val_EA_fold_config


def KFold_CrossValidate_ENN_novalidation(dataset_df, X, y_business, pd_rec_oracle, pd_revi_oracle_list, FOLDS, SEED, DATA_AUGMENT, 
                                        AUG_OPTIONS):

    X_train_fold, y_train_BE_fold, y_train_EE_fold = [], [], []
    X_test_fold, y_test_BE_fold = [], []

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in skf.split(X, y_business):

        if DATA_AUGMENT:
            train_index = data_augmentation(dataset_df, train_index, AUG_OPTIONS)

        X_train, y_train_BE, y_train_EE = X.iloc[train_index], y_business.iloc[train_index], pd_rec_oracle.iloc[train_index]
        X_test, y_test_BE = X.iloc[test_index], y_business.iloc[test_index]

        X_train_fold.append(X_train)
        y_train_BE_fold.append(y_train_BE)
        y_train_EE_fold.append(y_train_EE)

        X_test_fold.append(X_test)
        y_test_BE_fold.append(y_test_BE)
    

    #Now set folds and configurations Ethics Aware data
    y_train_EA_fold_config = []

    for pd_revi_oracle in pd_revi_oracle_list:
        y_train_EA_fold, y_test_EA_fold = [], []
        for iter in range(FOLDS):
            train_I, test_I = y_train_BE_fold[iter].index, y_test_BE_fold[iter].index
            
            y_train_EA_fold.append(pd_revi_oracle.loc[train_I])
            y_test_EA_fold.append(pd_revi_oracle.loc[test_I])

        y_train_EA_fold_config.append(y_train_EA_fold) #list of 10 folds for each configuration

    return X_train_fold, y_train_BE_fold, y_train_EE_fold, X_test_fold, y_test_BE_fold, y_train_EA_fold_config



