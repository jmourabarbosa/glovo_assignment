from matplotlib.pylab import *
from scipy.stats import binned_statistic,spearmanr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scikits import bootstrap
from joblib import Parallel, delayed
import multiprocessing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


num_cores = multiprocessing.cpu_count()
sns.set(style="white")


# courier,feature_1,feature_2
data_lifetime=np.loadtxt("Courier_lifetime_data.csv",skiprows=1,delimiter=",",dtype="str")
data_lifetime[data_lifetime[:,2]=="",2] = "nan"
data_lifetime[data_lifetime[:,1] == "a",1] = 0
data_lifetime[data_lifetime[:,1] == "b",1] = 1
data_lifetime[data_lifetime[:,1] == "c",1] = 2
data_lifetime[data_lifetime[:,1] == "d",1] = 3 
data_lifetime=data_lifetime.astype(float)
df_lifetime = pd.DataFrame(data_lifetime,columns=["id","lf_f1","lf_f2"])

# filling the missing values, not the best method, but..
# m_f2 = df_lifetime["lf_f2"].mean()
# df_lifetime["lf_f2"][isnan(df_lifetime["lf_f2"])] = m_f2

#courier,week,feature_1,..,feature_17
data_week=np.loadtxt("Courier_weekly_data.csv",skiprows=1,delimiter=",")
features = ["f%i" % i for i in range(1,18)]
columns = concatenate([["id","w"],features])
df_week = pd.DataFrame(data_week,columns=columns)


# get couriers that have all 9,10 and 11 week (count == 3)
label_1 = df_week.loc[df_week["w"].isin([9,10,11])].groupby("id")["id"].count() == 3
label_1 = label_1[label_1].index

labels = zeros(len(df_lifetime))
labels[df_lifetime["id"].isin(label_1)] = 1
df_lifetime["label"] = labels

labels = zeros(len(df_week))
labels[df_week["id"].isin(label_1)] = 1
df_week["label"] = labels



# clean extra weeks
df_week = df_week[~df_week["w"].isin([8,9,10,11])]

# shift one value, to make space for week 0 (lifetime data)
df_week["w"]+=1

# there only lf_f1 and lf_f2 for week 0
df_week["lf_f1"] = 0
df_week["lf_f2"] = 0

df_all = pd.concat([df_week,df_lifetime])

# corr = df_week.corr(method="spearman").round(1)

# # # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)

# sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

# show()


# m_fs = df_all.mean()
# for col in df_all.drop("w",axis=1):
# 	df_all[col][isnan(df_all[col])] = m_fs[col]

df_all["w"][isnan(df_all["w"])] = 0

df_majority = df_all[df_all["label"]==0]
df_minority = df_all[df_all["label"]==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
# df_all = pd.concat([df_majority, df_minority_upsampled])




clf = xgb.XGBClassifier(max_depth=20)
param = {'max_depth':20, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 10
X,y = df_all.drop(["label"],axis=1),df_all.label
n_folds = 10

accs=[]
accs2=[]
for _ in range(n_folds):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
	dtrain = xgb.DMatrix(X_train,y_train)
	dtest = xgb.DMatrix(X_test,y_test)
	bst = xgb.train(param, dtrain, num_round)
	preds = bst.predict(dtest)
	accs.append(roc_auc_score(y_test.values, preds))


	# clf.fit(X_train,y_train)
	# prob = clf.predict_proba(X_test)
	# accs2.append(roc_auc_score(y_test.values, prob[:,1]))


print mean(accs)#,mean(accs)
# bar(X_train.columns,clf.feature_importances_)
