from torch.utils.data import Dataset, DataLoader
import torch

#Define Custom Dataloaders

class trainData_MLP(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class valData_MLP(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class testData_MLP(Dataset):
    
    def __init__(self, X_data): #, y_data):
        self.X_data = X_data
        # self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index]#, self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class trainData_ENN(Dataset):
    
    def __init__(self, X_data, y_dataBE, y_dataEE, y_dataEA):
        self.X_data = X_data
        self.y_dataBE = y_dataBE
        self.y_dataEE = y_dataEE
        self.y_dataEA = y_dataEA
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_dataBE[index], self.y_dataEE[index], self.y_dataEA[index]
        
    def __len__ (self):
        return len(self.X_data)

class valData_ENN(Dataset):
    
    def __init__(self, X_data, y_dataBE, y_dataEE, y_dataEA):
        self.X_data = X_data
        self.y_dataBE = y_dataBE
        self.y_dataEE = y_dataEE
        self.y_dataEA = y_dataEA
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_dataBE[index], self.y_dataEE[index], self.y_dataEA[index]
        
    def __len__ (self):
        return len(self.X_data)

class testData_ENN(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def init_DataLoaders_MLP(X_train_folds, y_train_folds, X_val_folds, y_val_folds, X_test_folds, X_test_clustered_fold, FOLDS, reduced_oracle_actived, BATCH_SIZE):

    train_data_folds = []
    val_data_folds = []
    test_data_folds = []
    test_cluster_data_folds = []

    for fold in range(FOLDS):
        train_data_folds.append(
            trainData_MLP(torch.FloatTensor(X_train_folds[fold].values), 
                    torch.FloatTensor(y_train_folds[fold].values).to(torch.long)
                    )
            )
        val_data_folds.append(
            valData_MLP(torch.FloatTensor(X_val_folds[fold].values), 
                    torch.FloatTensor(y_val_folds[fold].values).to(torch.long)
                    )
            )
        test_data_folds.append(
            testData_MLP( torch.FloatTensor(X_test_folds[fold].values)
                    )
            )
        if reduced_oracle_actived:
            test_cluster_data_folds.append(
                testData_MLP( torch.FloatTensor(X_test_clustered_fold[fold].values)
                        )
                )


    train_loader_folds = []
    val_loader_folds = []
    test_loader_folds = []
    test_cluster_loader_folds = []

    cluster = 6

    for fold in range(FOLDS):
        train_loader_folds.append(DataLoader(dataset=train_data_folds[fold], batch_size=BATCH_SIZE, shuffle=True))
        val_loader_folds.append(DataLoader(dataset=val_data_folds[fold], batch_size=1))
        test_loader_folds.append(DataLoader(dataset=test_data_folds[fold], batch_size=1))
        if reduced_oracle_actived:
            test_cluster_loader_folds.append(DataLoader(dataset=test_cluster_data_folds[fold], batch_size=cluster))

    return train_loader_folds, val_loader_folds, test_loader_folds, test_cluster_loader_folds


def init_DataLoaders_ENN(X_train_fold, y_train_BE_fold, y_train_EE_fold, y_train_EA_fold_config, \
                        y_val_EA_fold_config, X_val_fold, y_val_BE_fold, y_val_EE_fold, X_test_fold, \
                        dataset_name, FOLDS, BATCH_SIZE, do_validation):

    train_loader_fold_config, val_loader_fold_config, test_loader_fold_config = [], [], []

    if do_validation:
        for y_train_EA_fold, y_val_EA_fold in zip(y_train_EA_fold_config, y_val_EA_fold_config):
            train_loader_fold, val_loader_fold, test_loader_fold = [], [], []


            for fold in range(FOLDS):
                train_data = trainData_ENN(
                        torch.FloatTensor(X_train_fold[fold].values),
                        torch.FloatTensor(y_train_BE_fold[fold].values).to(torch.long), #BE y
                        torch.FloatTensor(y_train_EE_fold[fold].tolist()), #EE y
                        torch.FloatTensor(y_train_EA_fold[fold].tolist())# #EA y
                        )
                
                val_data = valData_ENN(
                        torch.FloatTensor(X_val_fold[fold].values),
                        torch.FloatTensor(y_val_BE_fold[fold].values).to(torch.long), #BE y
                        torch.FloatTensor(y_val_EE_fold[fold].tolist()), #EE y
                        torch.FloatTensor(y_val_EA_fold[fold].tolist()) #EA y
                        )
                        
                test_data = testData_ENN(
                        torch.FloatTensor(X_test_fold[fold].values)
                        )


                train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(dataset=val_data, batch_size=1)
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                train_loader_fold.append(train_loader)
                val_loader_fold.append(val_loader)
                test_loader_fold.append(test_loader)
            
            train_loader_fold_config.append(train_loader_fold)
            val_loader_fold_config.append(val_loader_fold)
            test_loader_fold_config.append(test_loader_fold)
            
    else:
        for y_train_EA_fold in y_train_EA_fold_config:
            train_loader_fold, test_loader_fold = [], []

            for fold in range(FOLDS):
                train_data = trainData_ENN(
                        torch.FloatTensor(X_train_fold[fold].values),
                        torch.FloatTensor(y_train_BE_fold[fold].values).to(torch.long), #BE y
                        torch.FloatTensor(y_train_EE_fold[fold].tolist()), #EE y
                        torch.FloatTensor(y_train_EA_fold[fold].tolist())# #EA y
                        )
                        
                test_data = testData_ENN(
                        torch.FloatTensor(X_test_fold[fold].values)
                        )


                train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
                test_loader = DataLoader(dataset=test_data, batch_size=1)

                train_loader_fold.append(train_loader)
                test_loader_fold.append(test_loader)
            
            train_loader_fold_config.append(train_loader_fold)
            test_loader_fold_config.append(test_loader_fold)
    

    return train_loader_fold_config, val_loader_fold_config, test_loader_fold_config

