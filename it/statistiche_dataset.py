import pandas as pd
import os

# file_path = "C:/Users/luca/Documents/Tesi magistrale/scripts_core_unito/RAI/resources/compas/"
# df = pd.read_csv(os.path.join(file_path, "splitted_compas.csv"), sep=';', header=0)

# d_recidivi = {}
# d_n_recidivi = {}

# for index, row in df.iterrows():
#     if row["is_recid"] == 0:
#         try:
#             d_n_recidivi[row["age"]] += 1
#         except:
#             d_n_recidivi[row["age"]] = 1
#     if row["is_recid"] == 1:
#         try:
#             d_recidivi[row["age"]] += 1
#         except:
#             d_recidivi[row["age"]] = 1

# for key in sorted(d_recidivi):
#     print("%s; %s" % (key, d_recidivi[key]))
# print("---------")
# for key in sorted(d_n_recidivi):
#     print("%s; %s" % (key, d_n_recidivi[key]))

# file_path = "C:/Users/luca/Documents/Tesi magistrale/scripts_core_unito/RAI/resources/compas/"
# df = pd.read_csv(os.path.join(file_path, "splitted_compas-eliminare.csv"), sep=';', header=0)

# d_recidivi = {}
# d_n_recidivi = {}

# for index, row in df.iterrows():
#     if row["BE_pred"] == 0:
#         try:
#             d_n_recidivi[row["age"]] += 1
#         except:
#             d_n_recidivi[row["age"]] = 1
#     if row["BE_pred"] == 1:
#         try:
#             d_recidivi[row["age"]] += 1
#         except:
#             d_recidivi[row["age"]] = 1

# for key in sorted(d_recidivi):
#     print("%s; %s" % (key, d_recidivi[key]))
# print("---------")
# for key in sorted(d_n_recidivi):
#     print("%s; %s" % (key, d_n_recidivi[key]))

###################################################################

# file_path = "C:/Users/luca/Documents/Tesi magistrale/scripts_core_unito/RAI/resources/compas/"
# df = pd.read_csv(os.path.join(file_path, "splitted_compas-eliminare.csv"), sep=';', header=0)

# fp = {}
# fn = {}
# tp = {}
# tn = {}

# for index, row in df.iterrows():
#     if row["is_recid"] == 0 and row["basicMLPpredictions"] == 1:
#         try:
#             fp[row["age"]] += 1
#         except:
#             fp[row["age"]] = 1
#     if row["is_recid"] == 1 and row["basicMLPpredictions"] == 0:
#         try:
#             fn[row["age"]] += 1
#         except:
#             fn[row["age"]] = 1
#     if row["is_recid"] == 0 and row["basicMLPpredictions"] == 0:
#         try:
#             tn[row["age"]] += 1
#         except:
#             tn[row["age"]] = 1
#     if row["is_recid"] == 1 and row["basicMLPpredictions"] == 1:
#         try:
#             tp[row["age"]] += 1
#         except:
#             tp[row["age"]] = 1

# print("False positive")
# for key in sorted(fp):
#     print("%s; %s" % (key, fp[key]))
# print("---------")
# print("False negative")
# for key in sorted(fn):
#     print("%s; %s" % (key, fn[key]))
# print("True negative")
# for key in sorted(tn):
#     print("%s; %s" % (key, tn[key]))
# print("True positive")
# for key in sorted(tp):
#     print("%s; %s" % (key, tp[key]))

#########################################################à

file_path = "C:/Users/luca/Documents/Tesi magistrale/Altri Dataset"
df = pd.read_csv(os.path.join(file_path, "AdultDataset.csv"), sep=';', header=0)


df['y'] = df['y'].astype(str)
conteggio = df.y.str.split(expand=True).stack().value_counts()

print(conteggio.to_string())

