import pandas as pd
import numpy as np

def load_reconstruction_oracle(business_oracle, oracles_output_dir_list, reduced_oracle_actived, index_to_drop, number_of_decisions, number_of_eth_values):

    rec_oracle = np.zeros(shape=(business_oracle.shape[0],number_of_decisions*number_of_eth_values*number_of_eth_values)) #1000 * 50

    with open(oracles_output_dir_list[-1]+"/reconstruction_oracle.txt","r+") as r:
        for line in r.readlines():
            tokens = line.split(":") #[index:vector]. Vector is for example --> 0.0055,0.0081125, ... 0.0089375,0.0081125,0.0055
            index = int(tokens[0].lstrip().rstrip())
            vector = np.array([tokens[1].lstrip().rstrip().split(",")],dtype='float32') # has now become a list of values
            rec_oracle[index] = vector #build rec_oracle step by step
    rec_oracle = rec_oracle.tolist()
    pd_rec_oracle = pd.Series(rec_oracle)

    return pd_rec_oracle


def load_revision_oracle(business_oracle, oracles_output_dir_list, reduced_oracle_actived, index_to_drop, number_of_decisions, number_of_eth_values):
    revi_oracle_list, revi_reduced_oracle_list = [], []
    pd_revi_oracle_list, pd_revi_reduced_oracle_list = [], []

    # Load revision_oracle.txt in revi_oracle for every configuration
    for config_path in oracles_output_dir_list:
        revi_oracle = np.zeros(shape=(business_oracle.shape[0],number_of_decisions*number_of_eth_values*number_of_eth_values))
        with open(config_path+'/revision_oracle.txt',"r+") as r:
            for line in r.readlines():
                tokens = line.split(":")
                index = int(tokens[0].lstrip().rstrip())
                vector = np.array([tokens[1].lstrip().rstrip().split(",")],dtype='float32')
                revi_oracle[index] = vector
        revi_oracle = revi_oracle.tolist()
        pd_revi_oracle = pd.Series(revi_oracle)
        
        revi_oracle_list.append(revi_oracle)
        pd_revi_oracle_list.append(pd_revi_oracle)


        if reduced_oracle_actived:
            
            pd_revi_reduced_oracle = pd_revi_oracle.drop(pd_revi_oracle.index[index_to_drop])
            revi_reduced_oracle = pd_revi_reduced_oracle.values.tolist()

            revi_reduced_oracle_list.append(revi_reduced_oracle)
            pd_revi_reduced_oracle_list.append(pd_revi_reduced_oracle)

    return pd_revi_oracle_list