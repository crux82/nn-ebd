import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
#from tqdm.notebook import tqdm

def binary_acc(y_pred, y_true):
    """accuracy"""
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    y_pred_tag = torch.argmax(y_pred_tag, axis=1)

    acc = accuracy_score(y_true.cpu(), y_pred_tag.cpu())

    return acc

def new_oracle_debiased(y, predictions_cluster, index_to_drop):
    reduced_predictions = []
    reduced_oracle = []


    for i, y_label, cluster_prediction in zip(range(len(y)), y, predictions_cluster):

        if cluster_prediction != -1: #If the prediction was NOT influenced by race, then I put it in the predictions
            reduced_predictions.append(cluster_prediction)
            reduced_oracle.append(y_label)
        else:
            index_to_drop.append(i)


    return reduced_predictions, reduced_oracle, index_to_drop


def voting_mlp_test(pred_list, num_clusters = 6):
    #I do voting for every 6 values (they are all derived from one case of the test set), there are in fact 6 races

    list_of_decisions = []

    for c in pred_list: #pred_list contains 6 predictions each time, as set in the dataloader
        max_index = c.index(max(c))
        list_of_decisions.append(max_index)
    
    sum_pred = sum(list_of_decisions)

    #if the sum is =0 or =1 it means that the overall decision is 0
    if sum_pred in [0, 1]: #0, 1
        pred = 0
    
    #if the sum is =6 or =5 it means that the overall decision is 1
    elif sum_pred in [6, 5]: #6, 5
        pred = 1
    
    #otherwise the line must be deleted, so I assign a -1
    else:
        pred = -1

    # print(list_of_decisions)
    return pred


def train_mlp(model, data_loader, EPOCHS, device, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        
        loss = criterion(y_pred, y_batch)
        acc = binary_acc(y_pred, y_batch)
        
        loss.backward()

        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return model

def val_mlp(model, data_loader, EPOCHS, device):

    epoch_acc = 0

    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        y_pred = model(X_batch)

        acc = binary_acc(y_pred, y_batch)
        epoch_acc += acc.item()
    
    epoch_acc = epoch_acc/len(data_loader)

    return model, epoch_acc

def test_mlp(model, data_loader, device, is_cluster = False):
    y_pred_list = []
    for data in data_loader:
        # data, target = data.to(device), target.to(device)
        data = data.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        if is_cluster:
            #pred in this case may be 0, 1 or -1 (indicating a row to be deleted because it is biased)
            pred = voting_mlp_test(output.tolist())
            y_pred_list.append(pred)
        else:
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)

            # # compare predictions to true label
            # correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            #save predictions
            y_pred_list.append(pred.item())
    
    return y_pred_list

##################
###TEST cluster###
def test_cluster_MLP(y, FOLDS, device, model_list, test_cluster_loader_folds, y_test_folds, predictions_cluster, index_to_drop):

    for fold, model in zip(range(FOLDS), model_list):
        model.eval()
        y_pred_list = test_mlp(model, test_cluster_loader_folds[fold], device, True)

        # index label of every folds
        index_y = y_test_folds[fold].index

        # put all results in predictions, to calculate metrics
        for ind, label in zip(index_y, y_pred_list):
            predictions_cluster[ind] = label
    reduced_predictions, reduced_oracle, index_to_drop = new_oracle_debiased(y, predictions_cluster, index_to_drop)

    return reduced_predictions, reduced_oracle, index_to_drop
