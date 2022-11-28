import numpy as np
from it.utils.data_utils import binary_laplace_smoothing as laplace_smo


def raceTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['race'].lstrip().rstrip() == "African-American": #African-American
            TMarray[index][0] = cat_value_dict['high']  # benefit in classifying as NOT a repeat offender (African-American)
            TMarray[index][1] = cat_value_dict['low']   # risk in classifying as NOT a repeat offender (African-American)
            TMarray[index][2] = cat_value_dict['low']   # recidivist benefit (African-American)
            TMarray[index][3] = cat_value_dict['high']  # recidivist risk (African-American)
        else:
            TMarray[index][0] = cat_value_dict['mild']   # benefit of classifying as NOT a repeat offender (others)
            TMarray[index][1] = cat_value_dict['mild']   # risk in classifying as NOT a repeat offender (others)
            TMarray[index][2] = cat_value_dict['mild']   # recidivist benefit (others)
            TMarray[index][3] = cat_value_dict['mild']   # recidivist risk (others)
    return  TMarray

def raceLawSchTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['race'].lstrip().rstrip() == "White":
            TMarray[index][0] = cat_value_dict['mild']  # benefit classify NOT passed the exam (white)
            TMarray[index][1] = cat_value_dict['mild'] # risk classify NOT passed the exam (white)
            TMarray[index][2] = cat_value_dict['mild'] # benefit passed in the exam (white)
            TMarray[index][3] = cat_value_dict['mild']  # risk passed exam (white)
        else:
            TMarray[index][0] = cat_value_dict['very low']   # benefit classify NOT passed the exam (others)
            TMarray[index][1] = cat_value_dict['very high']   # risk classify NOT passed in the exam (others)
            TMarray[index][2] = cat_value_dict['very high']   # benefit passed in the exam (others)
            TMarray[index][3] = cat_value_dict['very low']   # risk passed (others)
    return  TMarray

def raceLawSch2TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['race'].lstrip().rstrip() == "Asian" or row['race'].lstrip().rstrip() == "Black" or\
            row['race'].lstrip().rstrip() == "Hisp" or row['race'].lstrip().rstrip() == "Other":
            TMarray[index][0] = cat_value_dict['mild']  # benefit classify NOT passed the exam
            TMarray[index][1] = cat_value_dict['mild'] # risk classify NOT passed in the exam
            TMarray[index][2] = cat_value_dict['mild'] # benefit passed in the exam
            TMarray[index][3] = cat_value_dict['mild']  # risk passed exam
        else:
            TMarray[index][0] = cat_value_dict['very high']   # benefit classify NOT passed the exam (others)
            TMarray[index][1] = cat_value_dict['very low']   # risk classify NOT passed in the exam (others)
            TMarray[index][2] = cat_value_dict['very low']   # benefit passed in the exam(others)
            TMarray[index][3] = cat_value_dict['very high']   # risk passed exam  (others)
    return  TMarray

def raceAdultTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['race'].lstrip().rstrip() == "Black" or\
        row['race'].lstrip().rstrip() == "Amer-Indian-Eskimo" or\
        row['race'].lstrip().rstrip() == "Other":

            TMarray[index][0] = cat_value_dict['low']  # benefit classify as <50k (Black, America and Other)
            TMarray[index][1] = cat_value_dict['high'] # risk classify as <50k
            TMarray[index][2] = cat_value_dict['very high'] # benefits >=50k
            TMarray[index][3] = cat_value_dict['very low']  # risk >=50k
        else:
            TMarray[index][0] = cat_value_dict['mild']   # benefit classify as <50k altri
            TMarray[index][1] = cat_value_dict['mild']   # risk classify as <50k altri
            TMarray[index][2] = cat_value_dict['mild']   # benefits >=50k altri
            TMarray[index][3] = cat_value_dict['mild']   # risk >=50k altri
    return  TMarray

# def ageTM(dataframe, cat_value_dict):

#     TMarray = np.zeros(shape=(dataframe.shape[0],4))

#     for index, row in dataframe.iterrows():
#         if row['age_cat'].lstrip().rstrip() == "Less than 25":
#             TMarray[index][0] = cat_value_dict['very high']  # benefit in classifying as NOT a repeat offender (young)
#             TMarray[index][1] = cat_value_dict['very low']   # risk in classifying as NOT a repeat offender (young)
#             TMarray[index][2] = cat_value_dict['very low']   # benefit repeat offender (young)
#             TMarray[index][3] = cat_value_dict['very high']  # risk repeat offender (young)
        
#         elif row['age_cat'].lstrip().rstrip() == "25 - 45":
#             TMarray[index][0] = cat_value_dict['high']  # benefit in classifying as NOT a repeat offender (adult)
#             TMarray[index][1] = cat_value_dict['low']   # risk in classifying as NOT a repeat offender (adult)
#             TMarray[index][2] = cat_value_dict['low']   # benefit repeat offender(adult)
#             TMarray[index][3] = cat_value_dict['high']  # risk repeat offender (adult)

#         else:
#             TMarray[index][0] = cat_value_dict['very low']   # benefit in classifying as NOT a repeat offender (old)
#             TMarray[index][1] = cat_value_dict['very high']  # risk in classifying as NOT a repeat offender (old)
#             TMarray[index][2] = cat_value_dict['very high']  # benefit repeat offender (old)
#             TMarray[index][3] = cat_value_dict['very low']   # risk repeat offender (old)
#     return  TMarray


#TODO transalte italian comments to english
def ageTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['age'] <= 33: #fino a 33
            TMarray[index][0] = cat_value_dict['very high']  # beneficio nel classificare come NON recidivo (giovane)
            TMarray[index][1] = cat_value_dict['very low']   # rischio nel classificare come NON recidivo (giovane)
            TMarray[index][2] = cat_value_dict['very low']   # beneficio recidivo (giovane)
            TMarray[index][3] = cat_value_dict['very high']  # rischio recidivo (giovane)
        else:
            TMarray[index][0] = cat_value_dict['mild']   # beneficio nel classificare come NON recidivo (adulto)
            TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare come NON recidivo (adulto)
            TMarray[index][2] = cat_value_dict['mild']  # beneficio recidivo (adulto)
            TMarray[index][3] = cat_value_dict['mild']   # rischio recidivo (adulto)
    return  TMarray

#cat_value_dict = {'very low': 0.1, 'low': 0.25, 'mild': 0.5, 'high': 0.75, 'very high': 0.9}
# 25 is +0.4 and 34 is 0, so I do medium + this value
# 34-25 and I get the number of segments into which to divide 0.4
# so 0.4/(34-25) and then every time I go forward a year I take away from 0.4 this segment
# that is "mild "+ 0.4 - (age -25)*(0.4/(34-25))
# simplifying 

def age_variable_value(cat_value_dict, v1, v2, age, min_v = 18, max_v = 40): #because it is 18 to 40
    #the closer the person is to 18 years old, the closer the variable a_v will be to 0.9, vice versa the older the person is,
    # the more I will go towards the "mild" value i.e. 0.5
    min_v = min_v-1
    max_v = max_v+1
    if v2 == 'very high':
        a_v = cat_value_dict[v1] + 0.4 - (age - min_v)*(0.4/(max_v - min_v))
    if v2 == 'very low':
        a_v = cat_value_dict[v1] - 0.4 + (age - min_v)*(0.4/(max_v - min_v))

    return a_v

def age2TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['age'] <= 24: #fino a 24
            TMarray[index][0] = cat_value_dict['very high']  # beneficio nel classificare come NON recidivo (giovane)
            TMarray[index][1] = cat_value_dict['very low']   # rischio nel classificare come NON recidivo (giovane)
            TMarray[index][2] = cat_value_dict['very low']   # beneficio recidivo (giovane)
            TMarray[index][3] = cat_value_dict['very high']  # rischio recidivo (giovane)
        elif row['age'] >= 25 and row['age'] <=33: #da 25 a 33
            TMarray[index][0] = age_variable_value(cat_value_dict, 'mild', 'very high', row['age'])  # beneficio nel classificare come NON recidivo (giovane)
            TMarray[index][1] = age_variable_value(cat_value_dict, 'mild', 'very low', row['age'])   # rischio nel classificare come NON recidivo (giovane)
            TMarray[index][2] = age_variable_value(cat_value_dict, 'mild', 'very low', row['age'])    # beneficio recidivo (giovane)
            TMarray[index][3] = age_variable_value(cat_value_dict, 'mild', 'very high', row['age'])  # rischio recidivo (giovane)
        else:
            TMarray[index][0] = cat_value_dict['mild']   # beneficio nel classificare come NON recidivo (adulto)
            TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare come NON recidivo (adulto)
            TMarray[index][2] = cat_value_dict['mild']  # beneficio recidivo (adulto)
            TMarray[index][3] = cat_value_dict['mild']   # rischio recidivo (adulto)
    return  TMarray

# def sexTM(dataframe, cat_value_dict):

#     TMarray = np.zeros(shape=(dataframe.shape[0],4))

#     for index, row in dataframe.iterrows():
#         if row['sex'].lstrip().rstrip() == "Male":
#             TMarray[index][0] = cat_value_dict['high']  # beneficio nel classificare come <50k uomo
#             TMarray[index][1] = cat_value_dict['low']   # rischio nel classificare come <50k uomo
#             TMarray[index][2] = cat_value_dict['low']   # beneficio >=50k uomo
#             TMarray[index][3] = cat_value_dict['high']  # rischio >=50k uomo
#         else:
#             TMarray[index][0] = cat_value_dict['mild']  # beneficio nel classificare come <50k donna
#             TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare come <50k donna
#             TMarray[index][2] = cat_value_dict['mild']  # beneficio >=50k donna
#             TMarray[index][3] = cat_value_dict['mild']  # rischio >=50k donna
#     return  TMarray

def sexTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['sex'].lstrip().rstrip() == "Male":
            TMarray[index][0] = cat_value_dict['almost high']  # beneficio nel classificare come NON default uomo
            TMarray[index][1] = cat_value_dict['almost low']   # rischio nel classificare come NON default uomo
            TMarray[index][2] = cat_value_dict['low']   # beneficio nel classificare come default uomo
            TMarray[index][3] = cat_value_dict['high']  # rischio nel classificare come default uomo
        else:
            TMarray[index][0] = cat_value_dict['mild']  # beneficio nel classificare NON default donna
            TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare NON default donna
            TMarray[index][2] = cat_value_dict['mild']  # beneficio nel classificare default donna
            TMarray[index][3] = cat_value_dict['mild']  # rischio default donna
    return  TMarray

def sex2TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['sex'].lstrip().rstrip() == "Male":
            TMarray[index][0] = cat_value_dict['mild']  # beneficio nel classificare come <50k uomo
            TMarray[index][1] = cat_value_dict['mild']   # rischio nel classificare come <50k uomo
            TMarray[index][2] = cat_value_dict['mild']   # beneficio >=50k uomo
            TMarray[index][3] = cat_value_dict['mild']  # rischio >=50k uomo
        else:
            TMarray[index][0] = cat_value_dict['low']  # beneficio nel classificare come <50k donna
            TMarray[index][1] = cat_value_dict['high']  # rischio nel classificare come <50k donna
            TMarray[index][2] = cat_value_dict['high']  # beneficio >=50k donna
            TMarray[index][3] = cat_value_dict['low']  # rischio >=50k donna
    return  TMarray

def sex2alternativeTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['sex'].lstrip().rstrip() == "Female":
            TMarray[index][0] = cat_value_dict['low']  # beneficio nel classificare come <50k donna
            TMarray[index][1] = cat_value_dict['high']   # rischio nel classificare come <50k donna
            TMarray[index][2] = cat_value_dict['high']   # beneficio >=50k donna
            TMarray[index][3] = cat_value_dict['low']  # rischio >=50k donna
        else:
            TMarray[index][0] = cat_value_dict['mild']  # beneficio nel classificare come <50k uomo
            TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare come <50k uomo
            TMarray[index][2] = cat_value_dict['mild']  # beneficio >=50k uomo
            TMarray[index][3] = cat_value_dict['mild']  # rischio >=50k uomo
    return  TMarray

def sexGermanTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['sex'].lstrip().rstrip() == "female":
            TMarray[index][0] = cat_value_dict['mild']  #beneficio nell'erogare il prestito donna
            TMarray[index][1] = cat_value_dict['mild']  #rischio erogare prestito donna
            TMarray[index][2] = cat_value_dict['mild']  #beneficio non erogare donna
            TMarray[index][3] = cat_value_dict['mild']  #rischio non erogare donna
        else:
            TMarray[index][0] = cat_value_dict['low']  #beneficio nell'erogare il prestito uomo(corrisponde a 0)
            TMarray[index][1] = cat_value_dict['high']   #rischio erogare prestito uomo(corrisponde a 0)
            TMarray[index][2] = cat_value_dict['high']   #beneficio non erogare uomo (corrisponde a 1)
            TMarray[index][3] = cat_value_dict['low']  #rischio non erogare uomo (corrisponde a 1)

    return  TMarray

#inverso
# def sexGermanTM(dataframe, cat_value_dict):

#     TMarray = np.zeros(shape=(dataframe.shape[0],4))

#     for index, row in dataframe.iterrows():
#         if row['sex'].lstrip().rstrip() == "female":
#             TMarray[index][0] = cat_value_dict['low']  #beneficio nell'erogare il prestito donna
#             TMarray[index][1] = cat_value_dict['high']  #rischio erogare prestito donna
#             TMarray[index][2] = cat_value_dict['high']  #beneficio non erogare donna
#             TMarray[index][3] = cat_value_dict['low']  #rischio non erogare donna
#         else:
#             TMarray[index][0] = cat_value_dict['mild']  #beneficio nell'erogare il prestito uomo(corrisponde a 0)
#             TMarray[index][1] = cat_value_dict['mild']   #rischio erogare prestito uomo(corrisponde a 0)
#             TMarray[index][2] = cat_value_dict['mild']   #beneficio non erogare uomo (corrisponde a 1)
#             TMarray[index][3] = cat_value_dict['mild']  #rischio non erogare uomo (corrisponde a 1)

#     return  TMarray

def maritalTM(dataframe, cat_value_dict): #TODO da fare bene

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['marital'].lstrip().rstrip() == "single":
            TMarray[index][0] = cat_value_dict['high']  # beneficio nel classificare come NON recidivo (afro-americano)
            TMarray[index][1] = cat_value_dict['low']   # rischio nel classificare come NON recidivo (afro-americano)
            TMarray[index][2] = cat_value_dict['low']   # beneficio recidivo (afro-americano)
            TMarray[index][3] = cat_value_dict['high']  # rischio recidivo (afro-americano)
        else:
            TMarray[index][0] = cat_value_dict['mild']   # beneficio nel classificare come NON recidivo (altri)
            TMarray[index][1] = cat_value_dict['mild']   # rischio nel classificare come NON recidivo (altri)
            TMarray[index][2] = cat_value_dict['mild']   # beneficio recidivo (altri)
            TMarray[index][3] = cat_value_dict['mild']   # rischio recidivo (altri)
    return  TMarray


def mildTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        TMarray[index][0] = cat_value_dict['mild']  # beneficio nel classificare come NON recidivo (afro-americano)
        TMarray[index][1] = cat_value_dict['mild']   # rischio nel classificare come NON recidivo (afro-americano)
        TMarray[index][2] = cat_value_dict['mild']   # beneficio recidivo (afro-americano)
        TMarray[index][3] = cat_value_dict['mild']  # rischio recidivo (afro-americano)

    return  TMarray


def juv_fel_TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['juv_fel_count'] == 1:
            TMarray[index][0] = cat_value_dict['very high']  # beneficio classificare NON recidivo (baby criminale principiante)
            TMarray[index][1] = cat_value_dict['very low']   # rischio classificare NON recidivo (baby criminale principiante)
            TMarray[index][2] = cat_value_dict['very low']   # beneficio recidivo (baby criminale principiante)
            TMarray[index][3] = cat_value_dict['very high']  # rischio recidivo (baby criminale principiante)

        elif row['juv_fel_count'] == 2:
            TMarray[index][0] = cat_value_dict['high']  # beneficio nel classificare come NON recidivo (baby criminale)
            TMarray[index][1] = cat_value_dict['low']   # rischio nel classificare come NON recidivo (baby criminale)
            TMarray[index][2] = cat_value_dict['low']   # beneficio recidivo (baby criminale)
            TMarray[index][3] = cat_value_dict['high']  # rischio recidivo (baby criminale)
        else:
            TMarray[index][0] = cat_value_dict['mild']   # beneficio classificare come NON recidivo (baby criminale seriale)
            TMarray[index][1] = cat_value_dict['mild']  # rischio nel classificare come NON recidivo (baby criminale seriale)
            TMarray[index][2] = cat_value_dict['mild']  # beneficio recidivo (baby criminale seriale)
            TMarray[index][3] = cat_value_dict['mild']   # rischio recidivo (baby criminale seriale)
    return  TMarray




# def motherhoodTM(dataframe, cat_value_dict):

#     TMarray = np.zeros(shape=(dataframe.shape[0],4))

#     for index, row in dataframe.iterrows():

#         if row['sex'].lstrip().rstrip() == "female":
#             # se sei donna con almeno 2 figli vieni favorita molto
#             if row['people_under_maintenance'] > 1: 
#                 TMarray[index][0] = cat_value_dict['very high'] #beneficio nell'erogare il prestito (corrisponde a 0)
#                 TMarray[index][1] = cat_value_dict['very low'] #rischio erogare prestito (corrisponde a 0)
#                 TMarray[index][2] = cat_value_dict['very low'] #beneficio non erogare (corrisponde a 1)
#                 TMarray[index][3] = cat_value_dict['very high'] #rischio non erogare (corrisponde a 1)
#             else:
#                 # altrimenti (quindi figli <= 1) vieni favorita di meno. Ad esempio anziché 'very high' ricevi 'high'
#                 TMarray[index][0] = cat_value_dict['high']
#                 TMarray[index][1] = cat_value_dict['low']
#                 TMarray[index][2] = cat_value_dict['low']
#                 TMarray[index][3] = cat_value_dict['high']
#         else:   
#             # Se sei uomo e hai almeno due figli, sei favorito quanto una donna con meno di due figli 
#             # (e meno favorito di una donna con almeno 2 figli)
#             if row['people_under_maintenance'] > 1: #uomini con almeno 2 figli
#                 TMarray[index][0] = cat_value_dict['high']
#                 TMarray[index][1] = cat_value_dict['mild']
#                 TMarray[index][2] = cat_value_dict['low']
#                 TMarray[index][3] = cat_value_dict['mild']
#             else:
#                 # altrimenti non sei favorito
#                 TMarray[index][0] = cat_value_dict['mild']
#                 TMarray[index][1] = cat_value_dict['mild']
#                 TMarray[index][2] = cat_value_dict['mild']
#                 TMarray[index][3] = cat_value_dict['mild']
#     return TMarray


def motherhoodTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():

        if row['sex'].lstrip().rstrip() == "female":
            # if you are a woman with at least two children you are greatly favoured
            if row['people_under_maintenance'] > 1: 
                TMarray[index][0] = cat_value_dict['very high'] #beneficio nell'erogare il prestito (corrisponde a 0)
                TMarray[index][1] = cat_value_dict['very low'] #rischio erogare prestito (corrisponde a 0)
                TMarray[index][2] = cat_value_dict['very low'] #beneficio non erogare (corrisponde a 1)
                TMarray[index][3] = cat_value_dict['very high'] #rischio non erogare (corrisponde a 1)
            else:
                # altrimenti (quindi figli <= 1) vieni favorita di meno. Ad esempio anziché 'very high' ricevi 'high'
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['low']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['high']
        else:   
            # Se sei uomo e hai almeno due figli, sei favorito quanto una donna con meno di due figli 
            # (e meno favorito di una donna con almeno 2 figli)
            if row['people_under_maintenance'] > 1: #uomini con almeno 2 figli
                TMarray[index][0] = cat_value_dict['mild']
                TMarray[index][1] = cat_value_dict['mild']
                TMarray[index][2] = cat_value_dict['mild']
                TMarray[index][3] = cat_value_dict['mild']
            else:
                # altrimenti non sei favorito
                TMarray[index][0] = cat_value_dict['mild']
                TMarray[index][1] = cat_value_dict['mild']
                TMarray[index][2] = cat_value_dict['mild']
                TMarray[index][3] = cat_value_dict['mild']
    return TMarray


def culturalTM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if row['foreign_worker'].lstrip().rstrip() == "yes":
            if row['other_debtors'] in ["co-applicant", "guarantor"]:
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['very low']  
                TMarray[index][2] = cat_value_dict['very low']
                TMarray[index][3] = cat_value_dict['high']
            else:
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['low']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['high']
        else:
            TMarray[index][0] = cat_value_dict['mild']
            TMarray[index][1] = cat_value_dict['mild']
            TMarray[index][2] = cat_value_dict['mild']
            TMarray[index][3] = cat_value_dict['mild']
    return  TMarray


def youth1TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0], 4))

    for index, row in dataframe.iterrows():
        if row['age'] <= 35:
            if row['housing'].lstrip().rstrip() in ['for free','own']:
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['low']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['high']
            else:
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['mild']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['mild']
        else:
            TMarray[index][0] = cat_value_dict['mild']
            TMarray[index][1] = cat_value_dict['mild']
            TMarray[index][2] = cat_value_dict['mild']
            TMarray[index][3] = cat_value_dict['mild']
    return  TMarray


def youth2TM(dataframe, cat_value_dict):

    TMarray = np.zeros(shape=(dataframe.shape[0], 4))

    for index, row in dataframe.iterrows():
        if row['age'] <= 35:
            if row['credit_history'].lstrip().rstrip() == "no credits taken/ all credits paid back duly":
                TMarray[index][0] = cat_value_dict['very high']
                TMarray[index][1] = cat_value_dict['low']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['very high']
            else:
                TMarray[index][0] = cat_value_dict['high']
                TMarray[index][1] = cat_value_dict['mild']
                TMarray[index][2] = cat_value_dict['low']
                TMarray[index][3] = cat_value_dict['mild']
        else:
            TMarray[index][0] = cat_value_dict['mild']
            TMarray[index][1] = cat_value_dict['mild']
            TMarray[index][2] = cat_value_dict['mild']
            TMarray[index][3] = cat_value_dict['mild']
    return TMarray

def socially_beneficial_purposeTM(dataframe, cat_value_dict):

    TMArray = np.zeros(shape=(dataframe.shape[0],4))

    for index, row in dataframe.iterrows():
        if (row['job'] in ['unemployed/ unskilled - non-resident',  'unskilled - resident']) and (row['purpose'] in ['retraining', 'business']):
            TMArray[index][0] = cat_value_dict['high']
            TMArray[index][1] = cat_value_dict['low']
            TMArray[index][2] = cat_value_dict['low']
            TMArray[index][3] = cat_value_dict['mild']
        elif (row['job'] in ['unemployed/ unskilled - non-resident',  'unskilled - resident']) and (row['purpose'] == 'radio/television'):
            TMArray[index][0] = cat_value_dict['very low']
            TMArray[index][1] = cat_value_dict['mild']
            TMArray[index][2] = cat_value_dict['mild']
            TMArray[index][3] = cat_value_dict['mild']
        elif (row['purpose'] == 'education' and row['people_under_maintenance']>=1):
            TMArray[index][0] = cat_value_dict['very high']
            TMArray[index][1] = cat_value_dict['very low']
            TMArray[index][2] = cat_value_dict['low']
            TMArray[index][3] = cat_value_dict['very high']
        else:
            TMArray[index][0] = cat_value_dict['mild']
            TMArray[index][1] = cat_value_dict['mild']
            TMArray[index][2] = cat_value_dict['mild']
            TMArray[index][3] = cat_value_dict['mild']

    return TMArray

def normalize(x, bounds):
    return bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower'])

def value_to_pos(v):
    #value_to_pos_dict = {0.1: 0, 0.25: 1, 0.5: 2, 0.75: 3, 0.9: 4} #input [0.1|0.9] -> [0|4]
    
    if v >= 0.1 and v <= 0.25:
        return normalize(v,{'actual': {'lower': 0.1, 'upper': 0.25}, 'desired': {'lower': 0, 'upper': 1}})
    elif v > 0.25 and v <= 0.5:
        return normalize(v,{'actual': {'lower': 0.25, 'upper': 0.5}, 'desired': {'lower': 1, 'upper': 2}})
    elif v > 0.5 and v <= 0.75:
        return normalize(v,{'actual': {'lower': 0.5, 'upper': 0.75}, 'desired': {'lower': 2, 'upper': 3}})
    elif v > 0.75 and v <= 0.9:
        return normalize(v,{'actual': {'lower': 0.75, 'upper': 0.9}, 'desired': {'lower': 3, 'upper': 4}})

def distributions_array(v):
    #distributions_array = np.zeros(shape=(5,5))
    distributions_array = np.array( #input [0|4]
        [[0.5, 0.36, 0.12, 0.015, 0.005], [0.3, 0.39, 0.25, 0.05, 0.01], [0.1, 0.2, 0.4, 0.2, 0.1],
         [0.01, 0.05, 0.25, 0.39, 0.3], [0.005, 0.015, 0.12, 0.36, 0.5]])

    if v == 0 or v==1 or v==2 or v==3 or v==4:
        return distributions_array[int(v)]
    elif v > 0 and v < 1:
        a = -normalize(v,{'actual': {'lower': 0, 'upper': 1}, 'desired': {'lower': -0.5, 'upper': -0.3}})
        b = normalize(v,{'actual': {'lower': 0, 'upper': 1}, 'desired': {'lower': 0.36, 'upper': 0.39}})
        c = normalize(v,{'actual': {'lower': 0, 'upper': 1}, 'desired': {'lower': 0.12, 'upper': 0.25}})
        d = normalize(v,{'actual': {'lower': 0, 'upper': 1}, 'desired': {'lower': 0.015, 'upper': 0.05}})
        e = normalize(v,{'actual': {'lower': 0, 'upper': 1}, 'desired': {'lower': 0.005, 'upper': 0.01}})
        return [a,b,c,d,e]
    elif v > 1 and v < 2:
        a = -normalize(v,{'actual': {'lower': 1, 'upper': 2}, 'desired': {'lower': -0.3, 'upper': -0.1}})
        b = -normalize(v,{'actual': {'lower': 1, 'upper': 2}, 'desired': {'lower': -0.39, 'upper': -0.2}})
        c = normalize(v,{'actual': {'lower': 1, 'upper': 2}, 'desired': {'lower': 0.25, 'upper': 0.4}})
        d = normalize(v,{'actual': {'lower': 1, 'upper': 2}, 'desired': {'lower': 0.05, 'upper': 0.2}})
        e = normalize(v,{'actual': {'lower': 1, 'upper': 2}, 'desired': {'lower': 0.01, 'upper': 0.1}})
        return [a,b,c,d,e]
    elif v > 2 and v < 3:
        a = -normalize(v,{'actual': {'lower': 2, 'upper': 3}, 'desired': {'lower': -0.1, 'upper': -0.01}})
        b = -normalize(v,{'actual': {'lower': 2, 'upper': 3}, 'desired': {'lower': -0.2, 'upper': -0.05}})
        c = -normalize(v,{'actual': {'lower': 2, 'upper': 3}, 'desired': {'lower': -0.4, 'upper': -0.25}})
        d = normalize(v,{'actual': {'lower': 2, 'upper': 3}, 'desired': {'lower': 0.2, 'upper': 0.39}})
        e = normalize(v,{'actual': {'lower': 2, 'upper': 3}, 'desired': {'lower': 0.1, 'upper': 0.3}})
        return [a,b,c,d,e]
    elif v > 3 and v < 4:
        a = -normalize(v,{'actual': {'lower': 3, 'upper': 4}, 'desired': {'lower': -0.01, 'upper': -0.005}})
        b = -normalize(v,{'actual': {'lower': 3, 'upper': 4}, 'desired': {'lower': -0.05, 'upper': -0.015}})
        c = -normalize(v,{'actual': {'lower': 3, 'upper': 4}, 'desired': {'lower': -0.25, 'upper': -0.12}})
        d = -normalize(v,{'actual': {'lower': 3, 'upper': 4}, 'desired': {'lower': -0.39, 'upper': -0.36}})
        e = normalize(v,{'actual': {'lower': 3, 'upper': 4}, 'desired': {'lower': 0.3, 'upper': 0.5}})
        return [a,b,c,d,e]

def compute_trapezoidal_distributions(vect, ethics_mode):

    benefit_d0_distributions = np.zeros(shape=(vect.shape[0], 5))
    risk_d0_distributions = np.zeros(shape=(vect.shape[0], 5))
    benefit_d1_distributions = np.zeros(shape=(vect.shape[0], 5))
    risk_d1_distributions = np.zeros(shape=(vect.shape[0], 5))

    for i in range(vect.shape[0]):

        #the distribution is centred
        benefit_d0_distributions[i] = distributions_array(value_to_pos(vect[i][0]))
        risk_d0_distributions[i] = distributions_array(value_to_pos(vect[i][1]))
        benefit_d1_distributions[i] = distributions_array(value_to_pos(vect[i][2]))
        risk_d1_distributions[i] = distributions_array(value_to_pos(vect[i][3]))

    return benefit_d0_distributions, risk_d0_distributions, benefit_d1_distributions, risk_d1_distributions


def compute_EthicalValue(dataframe, feature_to_TM_dict, active_ethical_features, cat_value_dict, ethics_mode):
    #creates a dictionary
    eth_value_dictionary = dict()

    #We create several numpy matrices, initialised with all values 0 and of (dataframe size)x5
    o_b_d0_distr = np.zeros(shape=(dataframe.shape[0], 5))
    o_r_d0_distr = np.zeros(shape=(dataframe.shape[0], 5))
    o_b_d1_distr = np.zeros(shape=(dataframe.shape[0], 5))
    o_r_d1_distr = np.zeros(shape=(dataframe.shape[0], 5))
    
    #where active_ethical_features are the TMs in use, e.g. ['youthFostering', 'socialBenefit']
    for ef in range(len(active_ethical_features)):
        ethical_feature = active_ethical_features[ef] #ef è l'indice

        ##We create several matrices, initialised with all values 0 and (dataframe size)x5
        b_d0_distr = np.zeros(shape=(dataframe.shape[0], 5))
        r_d0_distr = np.zeros(shape=(dataframe.shape[0], 5))
        b_d1_distr = np.zeros(shape=(dataframe.shape[0], 5))
        r_d1_distr = np.zeros(shape=(dataframe.shape[0], 5))

        #feature_to_TM_dict is the dictionary
        # {
        #  'motherhoodFostering': [lambda x,cat_value_dict: motherhoodTM(x, cat_value_dict)],
        #  ...
        #  'socialBenefit': [lambda x,cat_value_dict: socially_beneficial_purposeTM(x,cat_value_dict)]
        #  }
        if ethical_feature in feature_to_TM_dict:
            truth_makers = feature_to_TM_dict[ethical_feature] #We load the active TM in this for loop, from the dictionary

            # for each TM in the dictionary
            # the for loop is usually only executed once, but there may be more "rules" (?) in a tm and thus be executed several times
            for tm in truth_makers:
                
                #cat_value_dict was {'very low': 0.1, ... 'very high': 0.9}
                ##
                tmp_distribution_centers = tm(dataframe, cat_value_dict)
                
                # calculates the trapezoidal distribution
                b_d0, r_d0, b_d1, r_d1 = compute_trapezoidal_distributions(tmp_distribution_centers, ethics_mode)
                
                b_d0_distr += b_d0
                r_d0_distr += r_d0
                b_d1_distr += b_d1
                r_d1_distr += r_d1

            o_b_d0_distr += b_d0_distr
            o_r_d0_distr += r_d0_distr
            o_b_d1_distr += b_d1_distr
            o_r_d1_distr += r_d1_distr
    #at this point the TMs have completed their work.

        if np.abs(sum(b_d0_distr[0])-1)>0.001:
            b_d0_distr = np.divide(b_d0_distr, np.sum(b_d0_distr))
            r_d0_distr = np.divide(r_d0_distr, np.sum(r_d0_distr))
            b_d1_distr = np.divide(b_d1_distr, np.sum(b_d1_distr))
            r_d1_distr = np.divide(r_d1_distr, np.sum(r_d1_distr))
            # b_d0_distr = softmax(b_d0_distr, axis=-1)
            # r_d0_distr = softmax(r_d0_distr, axis=-1)
            # b_d1_distr = softmax(b_d1_distr, axis=-1)
            # r_d1_distr = softmax(r_d1_distr, axis=-1)
        
        #ethical_feature is the active tm in this for loop
        eth_value_dictionary[ethical_feature] = {'b_d0':b_d0_distr, 'r_d0':r_d0_distr, 'b_d1':b_d1_distr, 'r_d1':r_d1_distr}
        ### ###print("eth_value_dictionary[ethical_feature]", eth_value_dictionary[ethical_feature])



    for i in range(o_b_d0_distr.shape[0]):
        o_b_d0_distr[i] = np.divide(o_b_d0_distr[i], np.sum(o_b_d0_distr[i]))
        o_r_d0_distr[i] = np.divide(o_r_d0_distr[i], np.sum(o_r_d0_distr[i]))
        o_b_d1_distr[i] = np.divide(o_b_d1_distr[i], np.sum(o_b_d1_distr[i]))
        o_r_d1_distr[i] = np.divide(o_r_d1_distr[i], np.sum(o_r_d1_distr[i]))
    # o_b_d0_distr = softmax(o_b_d0_distr, axis=-1)
    # o_r_d0_distr = softmax(o_r_d0_distr, axis=-1)
    # o_b_d1_distr = softmax(o_b_d1_distr, axis=-1)
    # o_r_d1_distr = softmax(o_r_d1_distr, axis=-1)

    ### ###print("eth_value_dictionary['overall']", eth_value_dictionary['overall'])
    eth_value_dictionary['overall'] = {'b_d0':o_b_d0_distr,'r_d0':o_r_d0_distr,'b_d1':o_b_d1_distr,'r_d1':o_r_d1_distr}
    return eth_value_dictionary


def compute_global_reconstruction_oracle(overall_eth_values, number_of_values, number_of_decisions, dataset_size):
    vector_length = number_of_decisions*number_of_values*number_of_values # 2*5*5

    reconstruction_oracle = np.zeros(shape=(dataset_size, vector_length))

    for i in range(dataset_size):
        tmp_row = np.zeros(shape=vector_length)     #2*5*5
        for d in range(number_of_decisions):        #for each decision (there are 2)
            for b in range(number_of_values):       #for each benefit
                for r in range(number_of_values):   #for each risk
                    p_d = 0.5                       #uniform
                    p_b = overall_eth_values['b_d'+str(d)][i][b] #i is the row of the dataset, b the benefit
                    p_r = overall_eth_values['r_d'+str(d)][i][r] #e.g.-> overall_eth_values['r_d0][2][3]
                    #r+number_of_values*b+number_of_values*number_of_values*d is simply an incremental index
                    tmp_row[r+number_of_values*b+number_of_values*number_of_values*d] = p_d*p_b*p_r #calculation as explained in the formula in the paper
        tmp_row = np.divide(tmp_row, np.sum(tmp_row)) #everything between 0 and 1
        reconstruction_oracle[i] = tmp_row #save the line

    return reconstruction_oracle


def compute_global_revisor_oracle(smoothed_decision_vector, overall_eth_values, number_of_values, number_of_decisions,
                                  dataset_size, ethics_mode,tweaking_factor):

    vector_length = number_of_decisions*number_of_values*number_of_values #2*5*5= 50
    revisor_oracle = np.zeros(shape=(dataset_size,vector_length))
    if ethics_mode == 'Flat':
        ethical_optimum = np.array([0.15, 0.15, 0.15, 0.275, 0.275, 0.275, 0.275, 0.15, 0.15, 0.15])
        ethical_minimum = np.array([0.275, 0.275, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.275, 0.275])
    elif ethics_mode in ['Steep', 'Inverted']:
        ethical_optimum = np.array([0.005, 0.015, 0.12, 0.36, 0.5, 0.5, 0.36, 0.12, 0.015, 0.005])
        ethical_minimum = np.array([0.5, 0.36, 0.12, 0.015, 0.005, 0.005, 0.015, 0.12, 0.36, 0.5])
    else:
        print("Allowed value for ethics_mode are \'Flat\' or \'Steep\', not "+ethics_mode)
        raise KeyError

    # for each line
    for i in range(dataset_size): # number of rows dataset
        tmpRow = np.zeros(shape=vector_length) #vector of 50 elements
        for d in range(number_of_decisions):
            #concatenates benefits and risks of the same row i decision
            tmpEthVect = np.concatenate([overall_eth_values['b_d'+str(d)][i],overall_eth_values['r_d'+str(d)][i]],
                                        axis=-1)
            
            #basically calculates the distance from the minimum and ethical optimum of the tmpEthVect
            #divergence w.r.t. ethical OPTIMUM
            posDiv = compute_ethical_divergence(tmpEthVect, ethical_optimum,
                                                number_of_values=number_of_values)
            #divergence w.r.t. ethical MINIMUM
            negDiv = compute_ethical_divergence(tmpEthVect, ethical_minimum,
                                                number_of_values=number_of_values)

            if posDiv < 0 or posDiv > 1:
                print("Error in computing posDiv for instance "+str(i))
                print("output value is "+str(posDiv))
            if negDiv < 0 or negDiv > 1:
                print("Error in computing negDiv for instance "+str(i))
                print("output value is "+str(negDiv))
            for b in range(number_of_values):       #for each benefits
                for r in range(number_of_values):   #for each risk
                    #If the ethical vector is closer to the ethical optimum, it means that the decision is ethically compliant and I must favour it.
                    if negDiv >= posDiv:
                        product_vector_index = r + number_of_values * b + number_of_values * number_of_values * d #in this way an incremental index is formed
                        p_b = ethical_optimum[0:5][b]
                        p_r = ethical_optimum[5:10][r]
                    
                    #If, on the other hand, the ethical vector is closer to the ethical minimum, it means that I have to discourage it and take the minimum
                    else: 
                        product_vector_index = r + number_of_values * b + number_of_values * number_of_values * d 
                        p_b = ethical_minimum[0:5][b]
                        p_r = ethical_minimum[5:10][r]
                    p_d = smoothed_decision_vector[i][d]
                    tweaked_p_b = np.power(p_b,tweaking_factor[0])
                    tweaked_p_r = np.power(p_r, tweaking_factor[1])
                    tmpRow[product_vector_index] = p_d * tweaked_p_b * tweaked_p_r #as paper


        tmpRow = np.divide(tmpRow, np.sum(tmpRow)) # tranform tmpRow into values between 0 and 1, all with sum 1

        if np.abs(sum(tmpRow)-1)>0.001:
            print("sum -> "+str(sum(tmpRow)))
        revisor_oracle[i] = tmpRow #assign the line

    return revisor_oracle


def compute_ethical_divergence(x,ref, number_of_values):
    be_norm = np.linalg.norm(x[0:number_of_values]-ref[0:number_of_values])
    ri_norm = np.linalg.norm(x[number_of_values:2*number_of_values]-ref[number_of_values:2*number_of_values])

    return 1-np.exp(-1*(np.power(be_norm,2)+np.power(ri_norm,2)))

def ethical_data_enrichment(dataframe, active_ethical_features,smoothing_factor, ethics_mode, tweaking_factor, dataset_size, dataset_y_name):
    """
    Some of the values given are examples only, and therefore vary depending on the model input:
    - dataframe = is the csv splitted....csv
    - active_ethical_features = ['youthFostering', 'socialBenefit']
    - smoothing_factor = 0.1
    - ethics_mode = 'Steep'
    - tweaking_factor = [tweaking_factor_benefit, tweaking_factor_risk]. 
        is given as beta in the paper. Values would be e.g. beta_b = 0.75 and beta_r = 0.75.
    """

    # We save in one_hot_decision_vector the values of the dataFrame, without the axis labels. In practice 
    # corresponds to the column of the "default" or "is_recid" field, all loaded into a vector
    one_hot_decision_vector = dataframe[dataset_y_name].values 
    
    # Creates an array initialized with all 0. one_hot_decision_vector.shape[0] returns the dimensionality 
    # of the one_hot_decision_vector, so 1000. At this point we have shape=(1000,2) which creates a 
    # matrix 1000x2 all of 0. i.e. [[0. 0.] [0. 0.] [0. 0.] ... [0. 0.] [0. 0.] [0. 0.]]
    decision_vector = np.zeros(shape=(one_hot_decision_vector.shape[0],2))

    # decision_vector.shape[0] is always 1000. If we have for example: 
    # - one_hot_decision_vector = [0, 1, ..., 0]
    # - decision_vector = [[0. 0.] [0. 0.] ... [0. 0.]]
    # then decision_vector will result in [[1. 0.] [0. 1.] ... [1. 0.]]
    for i in range(decision_vector.shape[0]):
        decision_vector[i][one_hot_decision_vector[i]] = 1

    # 5 value categories
    cat_value_dict = {'very low': 0.1, 'almost low': 0.17, 'low': 0.25, 'mild': 0.5, 'high': 0.75, 'almost high': 0.82, 'very high': 0.9}
    
    # Dictionary. For example, the first line defines a lambda function that takes as input x, cat_value_dict and 
    # return motherhoodTM(x, cat_value_dict)

    feature_to_TM_dict = {  'motherhoodFostering': [lambda x,cat_value_dict: motherhoodTM(x, cat_value_dict)],
                            'raceTM': [lambda x,cat_value_dict: raceTM(x, cat_value_dict)],
                            'raceAdultTM': [lambda x,cat_value_dict: raceAdultTM(x, cat_value_dict)],
                            'raceLawSchTM': [lambda x,cat_value_dict: raceLawSchTM(x, cat_value_dict)],
                            'raceLawSch2TM': [lambda x,cat_value_dict: raceLawSch2TM(x, cat_value_dict)],
                            'maritalTM': [lambda x,cat_value_dict: maritalTM(x, cat_value_dict)],
                            'ageTM': [lambda x,cat_value_dict: ageTM(x, cat_value_dict)],
                            'sexTM': [lambda x,cat_value_dict: sexTM(x, cat_value_dict)],
                            'sex2TM': [lambda x,cat_value_dict: sex2TM(x, cat_value_dict)],
                            'sex2alternativeTM': [lambda x,cat_value_dict: sex2alternativeTM(x, cat_value_dict)],
                            'sexGermanTM': [lambda x,cat_value_dict: sexGermanTM(x, cat_value_dict)],
                            'age2TM': [lambda x,cat_value_dict: age2TM(x, cat_value_dict)],
                            'mildTM': [lambda x,cat_value_dict: mildTM(x, cat_value_dict)],
                            'juv_fel_TM': [lambda x,cat_value_dict: juv_fel_TM(x, cat_value_dict)],
                            'culturalInclusiveness': [lambda x,cat_value_dict: culturalTM(x, cat_value_dict)],
                            'youthFostering': [lambda x,cat_value_dict: youth1TM(x, cat_value_dict),
                                                 lambda x,cat_value_dict: youth2TM(x, cat_value_dict)],
                            'socialBenefit': [lambda x,cat_value_dict: socially_beneficial_purposeTM(x,cat_value_dict)]}

    #calculates benefits and risks for the two decisions, then returns 4 values for each tm and one "overall" (the sum of all)
    eth_values = compute_EthicalValue(dataframe=dataframe, feature_to_TM_dict=feature_to_TM_dict,
                                      active_ethical_features=active_ethical_features,cat_value_dict=cat_value_dict,
                                      ethics_mode=ethics_mode)
    #returns an array of values 1000*50
    reconstruction_oracle = compute_global_reconstruction_oracle(overall_eth_values=eth_values['overall'],
                                                                 number_of_values=5, number_of_decisions=2,
                                                                 dataset_size=dataset_size)
    #applico lo smoothing al vettore decisione (che contiene tutte le 1000 decisioni) per calcolare poi il revision oracle
    smoothed_decision_vector = laplace_smo(decision_vector, smoothing_factor)

    #returns an array of values 1000*50
    revision_oracle = compute_global_revisor_oracle(overall_eth_values=eth_values['overall'], number_of_decisions=2,
                                                    number_of_values=5, smoothed_decision_vector=smoothed_decision_vector,
                                                    dataset_size=dataframe.shape[0], ethics_mode=ethics_mode,
                                                    tweaking_factor=tweaking_factor)

    return eth_values, reconstruction_oracle, revision_oracle
