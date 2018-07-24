from matplotlib.pylab import *
from scipy.stats import binned_statistic,spearmanr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scikits import bootstrap
from joblib import Parallel, delayed
import multiprocessing
from sklearn import decomposition

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

df_lifetime = pd.DataFrame(data_lifetime,columns=["id","f1","f2"])

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


# feature_power = []
# for c in columns:
# 	cis = df_week.groupby("label")[c].apply(bootstrap.ci).values
# 	means = df_week.groupby("label")[c].mean()
# 	stds = df_week.groupby("label")[c].mean()
# 	if cis[0][0] < cis[1][0]:
# 		a = 0
# 		b = 1
# 	else:
# 		a = 1
# 		b = 0
# 	ss = sqrt((stds[b]**2 + stds[a]**2)/2)
# 	d_prime = (means[b] - means[a])/ss
# 	feature_power.append([c,cis[a][1]-cis[b][0],d_prime])
kk

# Generate correlation matrix
#corr = df_week.corr(method="spearman").round(1)
ica = decomposition.FastICA(whiten=True)
corr = pd.DataFrame(ica.fit_transform(df_week)).corr(method="spearman").round(1)


# # Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, mask=mask, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)


# def one_plot(i,j):
# 	print i,j
# 	f=sns.jointplot(x=columns[i], y=columns[j], data=df_week, kind="kde");
# 	title("joint_grid_%s_vs_%s" %(columns[i],columns[j]))
# 	f.savefig("joint_grid_%s_vs_%s" %(columns[i],columns[j]))

# [Parallel(n_jobs=num_cores)(delayed(one_plot)(i,j) for i in range(j,19)) for j in range(19)]