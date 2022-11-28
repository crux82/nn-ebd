import pandas as pd
import os
import it.truthMakers as t_makers
import numpy as np

def compute_ethics(smoothing_factor, ethics_mode, tweaking_factor, output_dir, active_ethical_features, dataset_size, root_path, dataset_name, dataset_y_name):
    """
    It is used to calculate oracles. It generates training outputs, so it is a kind of preprocessing and after
    run it you can start training.

    Description of file contents:
    - overall_ethical_signature.txt -> These are the ethical gold standards, the ethical truths. 
                                        The first number indicates the instance id, then there is the separator ":", and then 
                                        and then there is the output which is what the network must reproduce. This output is a 
                                        vector of numbers that for each element indicates the probability, given the instance, 
                                        P(business)*P(ethical benefit)*P(ethical risk). So for each instance 
                                        we have a probability distribution.
    - business_oracle.txt -> gold standard of business only
    - enriched_[...].csv -> starting dataset to which the values given by TMs are added. However, for this 
                            problem, using it does not change the results. In fact, only the business features are needed
    - reconstruction_oracle.txt -> oracle that only considers ethics without business. unlike the revision <audio>
    - revision_oracle.txt -> final oracle

    """
    print("Computing ethics...")

    value_names = ['vlow', 'low', 'mild', 'high', 'vhigh']
    categorical_ds = pd.read_csv(filepath_or_buffer=os.path.join(root_path + '/resources' + '/' + dataset_name + "/splitted_" + dataset_name + ".csv"), sep= ";")
    numerical_ds = pd.read_csv(filepath_or_buffer=os.path.join(root_path + '/resources' + '/' + dataset_name + "/standard_" + dataset_name + ".csv"), sep= ";")
    

    # categorical_ds = categorical_ds.drop(categorical_ds.index[index_to_drop])
    # numerical_ds = numerical_ds.drop(numerical_ds.index[index_to_drop])

    # categorical_ds.reset_index(inplace = True, drop = True)
    # numerical_ds.reset_index(inplace = True, drop = True)

    eth_values, reconstruction_oracle, revision_oracle = t_makers.ethical_data_enrichment(dataframe=categorical_ds,
                                                                                          active_ethical_features=
                                                                                          active_ethical_features,
                                                                                          smoothing_factor=smoothing_factor,
                                                                                          ethics_mode=ethics_mode,
                                                                                          tweaking_factor=tweaking_factor,
                                                                                          dataset_size=dataset_size,
                                                                                          dataset_y_name=dataset_y_name)
    # for each key in the eth_values dictionary, corresponding to the active tm + the overall one containing all of them
    
    for key in eth_values.keys(): #dict_keys(['motherhoodFostering', 'culturalInclusiveness', 'overall'])
        if key == 'overall': #discard the overall
            continue
        
        #{'b_d0':b_d0_distr, 'r_d0':r_d0_distr, 'b_d1':b_d1_distr, 'r_d1':r_d1_distr}
        for key2 in eth_values[key].keys(): #for each key in the eth_values[key] dictionary, i.e. dict_keys(['b_d0', 'r_d0', 'b_d1', 'r_d1'])
            for v in range(len(value_names)): #value_names = ['vlow', 'low', 'mild', 'high', 'vhigh']
                feature_name = key + "_" + key2 + "_" + value_names[v] #any combination of tm, benefit/risk, decision and value (one of the 5)
                categorical_ds[feature_name] =  eth_values[key][key2][:, v] #[:, v]  returns the entire column v of the numpy array, in this case 7000+ values per column.
                numerical_ds[feature_name] =    eth_values[key][key2][:, v]

    #create the folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #write csv files
    categorical_ds.to_csv(os.path.join(output_dir, "enriched_categorical_" + dataset_name + ".csv"), header=True,
                          index=False, sep = ";")
    numerical_ds.to_csv(os.path.join(output_dir, "enriched_numerical_" + dataset_name + ".csv"), header=True,
                        index=False, sep = ";")

    counter_1, counter_2, counter_3, counter_4 = 0, 0, 0, 0

    #concatenates the final ethical values (overall)
    overall_values = np.concatenate([eth_values['overall']['b_d0'], eth_values['overall']['r_d0'],
                                     eth_values['overall']['b_d1'], eth_values['overall']['r_d1']], axis=-1)
    
    business_oracle = numerical_ds[dataset_y_name].values #the gold standard decision column made up of 0 or 1

    #writes the oracles
    #print("output_dir", output_dir)
    with open(os.path.join(output_dir, "overall_ethical_signature.txt"), "w+") as f3:
        for o_line in range(overall_values.shape[0]):
            f3.write(str(counter_3) + ":")
            counter_3 += 1
            line = overall_values[o_line]
            for v in range(line.shape[0] - 1):
                f3.write(str(line[v]) + ",")
            f3.write(str(line[-1]) + "\n")

    with open(os.path.join(output_dir, "reconstruction_oracle.txt"), "w+") as f:
        for rec_line in range(reconstruction_oracle.shape[0]):
            f.write(str(counter_1) + ":")
            counter_1 += 1
            line = reconstruction_oracle[rec_line]
            for v in range(line.shape[0] - 1):
                f.write(str(line[v]) + ",")
            f.write(str(line[-1]) + "\n")

    with open(os.path.join(output_dir, "revision_oracle.txt"), "w+") as f2:
        for rev_line in range(revision_oracle.shape[0]):
            f2.write(str(counter_2) + ":")
            counter_2 += 1
            line = revision_oracle[rev_line]
            for v in range(line.shape[0] - 1):
                f2.write(str(line[v]) + ",")
            f2.write(str(line[-1]) + "\n")

    with open(os.path.join(output_dir,"business_oracle.txt"),"w+") as f4:
        for d in range(business_oracle.shape[0]):
            f4.write(str(counter_4)+":"+str(business_oracle[d])+"\n")
            counter_4 += 1
