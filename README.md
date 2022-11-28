# EthicalNN: Ethical Neural Network


## Introduction

EthicalNN is an architecture written in PyTorch that allows ethics to be integrated into a machine learning context by merging functional neural network reasoning and previously established ethical principles from ontological domains.

This code executes our EthicalNN model by applying it to different datasets:

* [COMPAS](https://github.com/propublica/compas-analysis "COMPAS"): published by the ProPublica organisation on the COMPAS scenario, which includes data from a two-year period of COMPAS scores obtained through a public records request issued by the Broward County Sheriff's Office in Florida. For the full process by which they acquired the data, [we report the source](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm "we report the source"). For our analysis, we used the dataset "*compas-scores-two-years.csv*".

* [Adult](https://archive.ics.uci.edu/ml/datasets/adult "Adult"): derived from US census data in 1994. It is one of the most popular datasets for fairness-aware classification research. It includes continuous and nominal attributes describing social information about registered citizens in terms of age, race, sex or marital status. If is generally used in classification tasks decide whether the annual income of a person exceeds 50,000 US dollars based on demographic characteristics.

* [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) "German Credit"): The German credit dataset is defined to represent bank account holders and it is used in automatic risk assessment prediction, i.e., to determine whether it is risky to grant credit to a person or not. As in the adult dataset, the potential ethical risk is to derive a data-driven model that makes it difficult to lend to women, young people or foreign workers.

* [Default credit cards](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients "Default credit cards"): The Default Credit Card Clients dataset investigated the customers’ default payments and contains payment information, demographics, credit data, payment history, and credit card customer statements in Taiwan from April 2005 to September 2005. The goal is to predict whether a customer will face the default situation in the next month or not.

* [Law school](http://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv "Law school"): The Law School Dataset was defined after the survey conducted by the Law School Admission Council (LSAC) across 163 law schools in the United States. The dataset contains the law school admission records and it is generally used to automatize to predict whether a candidate would pass the bar exam or predict a student’s first-year average grade (FYA).


This repository contains the code that implements the Actionable Ethics through Neural Learning paper published [here](https://ojs.aaai.org//index.php/AAAI/article/view/6005 "here") and extends it to several use cases.


## EthicalNN model: a short overview
The core of the adopted approach is to model ethics via automated reasoning over formal descriptions, e.g., ontologies, but by making it available \emph{during the learning stage}.
We suggest the explicit formulation of ethical principles through truth-makers and let the resulting ethical evidence to guide the model selection of deep learning architecture. 
This network jointly model causal as well as ethical conditions that characterize optimal decision-making. In this way, rules (i.e., truth-makers) are used to estimate risks and benefits connected to training cases; then the discovery of latent ethical knowledge, i.e., hidden information in the data that is meaningful under the ethical perspective, is carried out; finally, the latter evidence is made available when learning the target decision function. 
This framework results in a learning machine able to select the best decisions among those that are also ethically sustainable.

For more details, please refer to the paper published [here](https://ojs.aaai.org//index.php/AAAI/article/view/6005 "here").

## Background
The results were obtained by running the experiments in an environment with python 3.7 and PyTorch 1.9.0+cu111 with gpu on the google colab platform.

This repository was tested with python 3.7 and PyTorch 1.8.1. 


## Requirements

The following libraries are needed to execute the code, with the relevant version used

```
tqdm==4.56.0
scipy==1.6.0
numpy==1.19.5
torch==1.8.1
pandas==1.1.5
matplotlib==3.3.4
scikit_learn==1.0.2
```

Each library can be installed with the command 

```
pip install <lib_name>
```

or by using the requirements.txt file with

```
pip install -r requirements.txt
```


## Setup

1)
	Navigate to the folder where you want to download the code, then

	```
	git clone https://github.com/crux82/nn-ebd.git
	```

	OR

	download from browser and then extract it with the following code or with an appropriate program
	```
	https://github.com/crux82/nn-ebd/archive/refs/heads/main.zip
	```
	```
	unzip nn-ebd-main.zip
	rm nn-ebd-main.zip
	```

2)
	Finally, rename the downloaded folder to EthicalNN.


## Exporting the project root directory to PYTHONPATH

Mac/linux
```
export PYTHONPATH="${PYTHONPATH}:/path/to/nn-ebd-main/
```

Windows
```
set PYTHONPATH=%PYTHONPATH%;C:\path\to\nn-ebd-main\
```


## The configuration parameters

This is the list of configuration parameters, with their explanation, found in the configuration file "config.py", located in the root directory. Remember to modify them appropriately before executing the code:


* `root_path`: Parameter for setting the path to the project's root directory, i.e. the 'nn-ebd-main' directory. It must be written like this `"C:/Users/.../nn-ebd-main"`.
<!--* `SEED`: Parameter used to set the seed of the experiment, represented by an integer. Its purpose is to make reproducibility of results possible.-->
* `dataset_name`: Parameter for setting the dataset to be used during the experiment. The following values are allowed: 'german_credit', 'compas', 'adult', 'credit_card', 'law_school'.
* `active_ef`: Parameter for choosing the truth-makers that make up the ethical ontology. It must be a list consisting of the names of one or more truth-makers, as indicated below:
	 - For Compas you can choose between 'raceTM', 'ageTM', 'sexTM', 'mildTM' (a neutral TM);
	 - For German Credit 'motherhoodFostering' and 'culturalInclusiveness' are available;
	 - For Adult dataset you can choose between 'sexTM', 'raceAdultTM', 'mildTM';
	 - For Default credit cards 'sexTM' and 'mildTM' are available;
	 - For Law school dataset you can choose between 'sexTM', 'raceLawSchTM', 'mildTM'.

* `reduced_oracle_actived`: Parameter enabling the use of the unbiased dataset for the test phase. This configuration parameter only takes effect for experiments on the COMPAS dataset and is ignored in other cases. In other words it is a Boolean variable used ONLY FOR COMPAS to delete some biased instances concerning African-Americans. The default value is False.
* `enable_early_stop`: Parameter enabling early-stop on the ethical network.
* `PATIENCE_VALUE`: Early-stop patience value. Determines the maximum number of epochs before the early-stop mechanism stops if it does not find a better model than the previous ones, evaluated in terms of accuracy on the validation set.
* `ETH_NET_EPOCHS`: Sets the number of maximum epochs of ethical network training (or actual epochs if early-stop is not enabled)
* `MLP_EPOCHS`: Set the number of MLP training epochs (remember that the MLP does NOT have an early stop mechanism)
* `values`: Parameter for defining the list of configurations for the experiment. You can define one configuration or more than one. Each item in the list corresponds to a configuration which must have the following syntax:
    [smoothing, beta, beta]. 

	For example, if you want to run an experiment with the smoothing value alpha = 0.1 and beta = 0.14, you would write [0.1, 0.14, 0.14].   

	If there are several configurations with the same beta value, after running them all, the system automatically selects the one with the best alpha parameter (in the validation phase). We did not proceed in the same way for the german_credit dataset, in order to replicate the results of the [parent paper](https://ojs.aaai.org//index.php/AAAI/article/view/6005 "parent paper")..


<span style="text-decoration:underline;">Parameters for future developments, under experimentation:</span>

* `NUM_SAMPLES_LIME`: Sets the number of points to be evaluated for LIME. By default the value is 5000, but for this dataset we felt that the value 1000 would be a good compromise between speed and accuracy of results.


## How to perform an experiment

After appropriately modifying the parameters in the config.py file, before starting an experiment, we advise you to set the seed.
Seed is represented by an integer and its purpose is to make reproducibility of results possible.

The file to be modified is 'main_run.py' in the root directory.

This is an example of the contents of the file.

```
def main():

    multi_config_exec(9647566)
    multi_config_exec(34204329)
	#...
	#...you can add as many runs as you like
```

In this case we have two executions, the first with seed "9647566" and the second with seed "34204329".

Feel free to edit the SEEDS or add others lines.

Finally, to execute the code, run main_run.py.

The seeds used to produce the results in the paper are "9647566", "34204329", "7763456", "400423", "92578432".

```
cd <main_directory_nn-ebd-main>
python main_run.py
```

The configuration used for the experiment can always be changed in the config.py file before an experiment.


## Examples of configurations used for COMPAS and GERMAN CREDIT

To reproduce our own results on COMPAS datasets, e.g. with the race truth-maker and with SEED = 3274824, the following configuration should be used


```
root_path = SET ROOT PATH <C:/.../nn-ebd-main>

dataset_name = "compas"  
active_ef = ['raceTM']                
reduced_oracle_actived = True
enable_early_stop, PATIENCE_VALUE, ETH_NET_EPOCHS = True, 5, 30
MLP_EPOCHS = 100

FOLDS = 10
DATA_AUGMENT, AUG_OPTIONS = False, []

values = [
 [0.1, 0.01, 0.01], [0.1, 0.1, 0.1], [0.1, 0.11, 0.11], [0.1, 0.12, 0.12], [0.1, 0.13, 0.13], [0.1, 0.14, 0.14], [0.1, 0.15, 0.15], [0.1, 0.16, 0.16], [0.1, 0.17, 0.17], [0.1, 0.2, 0.2],
 [0.3, 0.01, 0.01], [0.3, 0.1, 0.1], [0.3, 0.11, 0.11], [0.3, 0.12, 0.12], [0.3, 0.13, 0.13], [0.3, 0.14, 0.14], [0.3, 0.15, 0.15], [0.3, 0.16, 0.16], [0.3, 0.17, 0.17], [0.3, 0.2, 0.2],
 [0.5, 0.01, 0.01], [0.5, 0.1, 0.1], [0.5, 0.11, 0.11], [0.5, 0.12, 0.12], [0.5, 0.13, 0.13], [0.5, 0.14, 0.14], [0.5, 0.15, 0.15], [0.5, 0.16, 0.16], [0.5, 0.17, 0.17], [0.5, 0.2, 0.2]
]

```


In order to reproduce our same results on the German Credit dataset, with SEED = 1, and the two truth-makers 'motherhoodFostering' and 'culturalInclusiveness' we must use the following configuration.


```
root_path = "C:/.../nn-ebd-main"
SEED = 1

dataset_name = "german_credit"  
active_ef = ['motherhoodFostering', 'culturalInclusiveness']              
reduced_oracle_actived = False
enable_early_stop, PATIENCE_VALUE, ETH_NET_EPOCHS = False, 5, 1000
MLP_EPOCHS = 100

FOLDS = 10
DATA_AUGMENT, AUG_OPTIONS = False, []

values = [
  [0.1, 0.01, 0.01], [0.1, 0.05, 0.05], [0.1, 0.1, 0.1], [0.1, 0.2, 0.2], [0.1, 0.35, 0.35], [0.1, 0.4, 0.4], [0.1, 0.5, 0.5],
  [0.3, 0.01, 0.01], [0.3, 0.05, 0.05], [0.3, 0.1, 0.1], [0.3, 0.2, 0.2], [0.3, 0.35, 0.35], [0.3, 0.4, 0.4], [0.3, 0.5, 0.5],
  [0.6, 0.01, 0.01], [0.6, 0.05, 0.05], [0.6, 0.1, 0.1], [0.6, 0.2, 0.2], [0.6, 0.35, 0.35], [0.6, 0.4, 0.4], [0.6, 0.5, 0.5],
  [1.0, 0.01, 0.01], [1.0, 0.05, 0.05], [1.0, 0.1, 0.1], [1.0, 0.2, 0.2], [1.0, 0.35, 0.35], [1.0, 0.4, 0.4], [1.0, 0.5, 0.5]
]

```


These experiments take quite a long time to run depending on the number of configurations entered. If you want to execute an experiment in a shorter time, you can reduce the number of configurations within the _values_ list. For example


```
values = [  
    [0.1, 0.1, 0.1], [0.1, 0.14, 0.14],
    [0.3, 0.1, 0.1], [0.3, 0.14, 0.14]
    ]
```



## Running on GPU or CPU

The code automatically selects and uses the gpu if present, otherwise it uses the cpu.

The results of a cpu execution may still differ from a gpu execution, even with the same SEED.
All our executions were done on GPUs. We do not recommend execution on CPUs due to longer completion times.


## Displaying the results

The results of the ethics network can be found in


```
C:\...\nn-ebd-main\resources\<DATASET_NAME>\data\enriched_data\<EXPERIMENT_NAME>\results\all_metrics.csv
```


The results of the MLP network can be found in


```
C:\...\nn-ebd-main\resources\<DATASET_NAME>\data\enriched_data\<EXPERIMENT_NAME>\results\baseline_metrics.csv
```

## Results obtained in previous executions
Results that have been obtained from previous executions will be available in the paper "Ethics by Design for Intelligent and Sustainable Adaptive Systems" once it has been published.


# How truth-makers are written
...

# Authors

Luca Squadrone

# Citation

To cite the paper, please use the following:
```
@inproceedings{squadrone-etal-2022-enn,
    title = "Ethics by Design for Intelligent and Sustainable Adaptive Systems",
    author = "Squadrone, Luca  and
      Croce, Danilo  and
      Basili, Roberto",
    booktitle = "",
    month = nov,
    year = "2022",
    address = "Online",
    publisher = "Associazione Italiana per l'Intelligenza Artificiale",
    url = "",
    pages = ""
}
```


# Acknowledgements

Daniele Rossini, Danilo Croce, Roberto Basili ([paper](https://ojs.aaai.org//index.php/AAAI/article/view/6005 "paper"))
