from matplotlib.pylab import *
from scipy.stats import spearmanr,zscore,linregress
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances as dist
from sklearn.model_selection import GridSearchCV   
from sklearn.metrics import confusion_matrix

SILENT = 0

# for each parameter tuning, eval_model(). Setting this as true will increase computing time a lot (100 training splits per parameters)
track_tuning = False

fill_missing = False
normalize = True


# SET WITH CARE, recomended to use False only when track_tuning is also false
# upsampling done only on training set has slightly better performance, but spliting breaks more often.
upsample_test = True
# final classifier was tested upsampling only the training set (Figure 6, as I belive should be done)
# but other figures, that show effect of parameter tuning were done upsampling also testing set

sns.set(style="white")

## helper functions
def test_model(X,y,clf_params):

	cv_folds  = 100
	ACC = zeros([cv_folds,5])
	ACC[:] = nan
	for nf in range(cv_folds):
		# Not sure why this **** is not working w/ model = xgb.XGBClassifier(clf_params)
		# hate to waste so many lines of code :)
		model = xgb.XGBClassifier(
			learning_rate= 0.1,
			n_estimators= 1000,
			max_depth= 5,
			min_child_weight= 1,
			gamma= 0,
			subsample= 0.8,
			colsample_bytree= 0.8,
			objective='binary:logistic',
			nthread= 4,
			scale_pos_weight= 1,
			seed= 27,
			silent=SILENT)
		model.set_params(n_estimators=clf_params["n_estimators"])
		model.set_params(max_depth= clf_params["max_depth"],min_child_weight=clf_params["min_child_weight"])
		model.set_params(gamma= clf_params["gamma"])
		model.set_params(reg_alpha= clf_params["reg_alpha"])

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
		X_train, X_test, y_train, y_test = resample_d(X_train, X_test, y_train, y_test)
		model.fit(X_train,y_train)
		prob = model.predict_proba(X_test)
		tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
		ACC[nf]=[tn, fp, fn, tp,roc_auc_score(y_test.values, prob[:,1])]

	eval_m=eval_model(ACC)
	return eval_m

def gridsearch(param_test,clf_params,cv_folds=10):
	# Not sure why this **** is not working w/ model = xgb.XGBClassifier(clf_params)
	model = xgb.XGBClassifier(
		learning_rate= 0.1,
		n_estimators= 1000,
		max_depth= 5,
		min_child_weight= 1,
		gamma= 0,
		subsample= 0.8,
		colsample_bytree= 0.8,
		objective='binary:logistic',
		nthread= 4,
		scale_pos_weight= 1,
		seed= 27,
		silent=SILENT)

	model.set_params(n_estimators=clf_params["n_estimators"])
	model.set_params(max_depth= clf_params["max_depth"],min_child_weight=clf_params["min_child_weight"])
	model.set_params(gamma= clf_params["gamma"])

	gsearch = GridSearchCV(estimator =  model, 
	param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=cv_folds,verbose=0 if SILENT else 1)
	gsearch.fit(X,y)

	return gsearch

def long_to_wide(df,key,features):
	df['idx'] = df.groupby(key).cumcount()
	tmp = []
	for var in features:
	    df['tmp_idx'] = var + '_' + df.idx.astype(str)
	    tmp.append(df.pivot(index=key,columns='tmp_idx',values=var))

	reshape = pd.concat(tmp,axis=1)

	return reshape

def stderr(data):
	return nanstd(data)/sqrt(len(data))

def eval_model(metrics):
	m = zeros([len(metrics),6])
	for i,(tn, fp, fn, tp, auc) in enumerate(metrics):
		err_rate = (fp + fn) / (tp + tn + fn + fp)
		acc = (tp + tn) / (tp + tn + fn + fp)
		sens = tp / (tp + fn)
		speci = tn / (tn + fp)
		FPR = fp / (tn + fp)
		m[i]=[err_rate,acc,sens,speci,FPR,auc]
	return m

def resample_d1(data):
	upsampled = resample(X[y==1], replace=True, n_samples=sum(y==0))
	return upsampled

def resample_d(X_train, X_test, y_train, y_test):

	train = X_train
	train.loc[:,"label"] = y_train

	test = X_test
	test.loc[:,"label"] = y_test


	upsampled_train = resample(X_train[y_train==1], replace=True, n_samples=sum(y_train==0))
	upsampled_test = resample(X_test[y_test==1], replace=True, n_samples=sum(y_test==0))

	upsampled_train = pd.concat([upsampled_train,X_train[y_train==0]])
	upsampled_test =  pd.concat([upsampled_test,X_test[y_test==0]])

	X_train,y_train = upsampled_train.drop(["label"],axis=1),upsampled_train.label

	if upsample_test:
		return X_train, X_test.drop("label",axis=1), y_train, y_test

	X_test,y_test = upsampled_test.drop(["label"],axis=1),upsampled_test.label


	return X_train, X_test, y_train, y_test

## 
## TASK 1
##

## IMPORT DATA

data_lifetime=np.loadtxt("Data/Courier_lifetime_data.csv",skiprows=1,delimiter=",",dtype="str")
data_lifetime[data_lifetime[:,2]=="",2] = "nan"
data_lifetime[data_lifetime[:,1] == "a",1] = 0
data_lifetime[data_lifetime[:,1] == "b",1] = 1
data_lifetime[data_lifetime[:,1] == "c",1] = 2
data_lifetime[data_lifetime[:,1] == "d",1] = 3 
data_lifetime=data_lifetime.astype(float)
df_lifetime = pd.DataFrame(data_lifetime,columns=["id","lf_f1","lf_f2"])

data_week=np.loadtxt("Data/Courier_weekly_data.csv",skiprows=1,delimiter=",")
features = ["f%i" % i for i in range(1,18)]
columns = concatenate([["id","w"],features])
df_week = pd.DataFrame(data_week,columns=columns)

## LABEL DATA

# get couriers ID that have all 9,10 and 11 week (count == 3)
label_1 = df_week.loc[df_week["w"].isin([9,10,11])].groupby("id")["id"].count() == 3
label_1 = label_1[label_1].index

# Label them as 1 in both datasets
labels = zeros(len(df_lifetime))
labels[df_lifetime["id"].isin(label_1)] = 1
df_lifetime["label"] = labels

labels = zeros(len(df_week))
labels[df_week["id"].isin(label_1)] = 1
df_week["label"] = labels


# clean weeks data correlated with label (...and 8)
df_week = df_week.loc[~(df_week["w"].isin([8,9,10,11]))]

# plot feature relationship with week
# rel = df_week.drop("id",axis=1).groupby("w").mean().apply(zscore)
# p_v = df_week.drop("id",axis=1).groupby("w").mean().apply(lambda x: spearmanr(x,range(8))[1])
# rel.ix[:,p_v.values<0.05].plot()
# ylabel("z-scored feature value")

# Long to wide: prepare week data to JOIN to lifetime
columns_to_add = df_week.drop(["id","w","label"],axis=1).columns
df_week_w = long_to_wide(df_week.copy(),"id",columns_to_add)
df_week_w["id"]=df_week_w.index

df_lifetime = pd.merge(df_week_w,df_lifetime,how="right",on=['id'])

# Add some univariate and bivariate features
feature = zeros(len(data_lifetime))
feature[:] = nan


for c in df_week.drop(["id","label","w"],axis=1).columns:
	df_week[c] = zscore(df_week[c])
	mean_f = df_week.groupby("id")[c].mean()
	var_f = df_week.groupby("id")[c].var()
	max_f = df_week.groupby("id")[c].max()
	min_f = df_week.groupby("id")[c].min()
	idx = max_f.index

	# fit a regression to add week evolution of each feature
	r=[]
	for _, group in df_week.groupby(["id"]):
		r.append(linregress(group["w"],group[c])[0])
	

	for ei,extra_f in enumerate([r,mean_f,var_f,max_f,min_f]):
		f = feature.copy()
		f[df_lifetime["id"].isin(idx)] = extra_f
		df_lifetime["%i_%s" %(ei, c)]=f



# PREPARE DATA FOR CLASSIFIER

# fill missing values with mean value.
if fill_missing:
	m_fs = df_lifetime.mean()
	for col in df_lifetime:
		df_lifetime[col][isnan(df_lifetime[col])] = m_fs[col]

# z-score data
if normalize:
	for col in df_lifetime.drop(["label","id"],axis=1).columns:
		#df_lifetime[col]=zscore(df_lifetime[col])
		df_lifetime[col]=(df_lifetime[col] - df_lifetime[col].mean())/df_lifetime[col].std(ddof=0)


## 
## TASK 2 
##


predictors = df_lifetime.drop(["label"],axis=1).columns
X,y = df_lifetime.drop(["label"],axis=1),df_lifetime.label
X = X[predictors]

# leave 50% of the data for testing (maybe too much)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.5)

# INITIAL PARAMETERS
cv_folds=10
early_stopping_rounds = 50
clf = xgb.XGBClassifier(
	learning_rate= 0.1,
	n_estimators= 1000,
	max_depth= 5,
	min_child_weight= 1,
	gamma= 0,
	subsample= 0.8,
	colsample_bytree= 0.8,
	objective='binary:logistic',
	nthread= 4,
	scale_pos_weight= 1,
	seed= 27,
	silent=SILENT)

m_tests = []
if track_tuning:
	m_tests.append(test_model(X_test,y_test,clf.get_xgb_params()))

## 
##  TASK 3: Performance is pretty good, but it can be improved
##

# TUNING tree-based parameters: # of estimators (trees)
xgtrain = xgb.DMatrix(X, label=y)
cvresult = xgb.cv(clf.get_xgb_params(), xgtrain, num_boost_round=clf.get_xgb_params()['n_estimators'], nfold=cv_folds,
    metrics='auc', early_stopping_rounds=early_stopping_rounds)

clf.set_params(n_estimators=cvresult.shape[0])
if track_tuning:
	m_tests.append(test_model(X_test,y_test,clf.get_xgb_params()))

# TUNING max_depth and min_child_weight (regularization)
param_test = {'max_depth':range(1,4,1),'min_child_weight':range(1,3,1)}

search = gridsearch(param_test,clf.get_xgb_params())
best_params = search.best_params_

clf.set_params(max_depth= best_params["max_depth"],min_child_weight=best_params["min_child_weight"])
if track_tuning:
	m_tests.append(test_model(X_test,y_test,clf.get_xgb_params()))

# TUNING gamma
param_test = {'gamma':[i/10.0 for i in range(0,5)]}

search = gridsearch(param_test,clf.get_xgb_params())
best_params = search.best_params_

clf.set_params(gamma= best_params["gamma"])
if track_tuning:
	m_tests.append(test_model(X_test,y_test,clf.get_xgb_params()))

# TUNING regularization alpha
param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

search = gridsearch(param_test,clf.get_xgb_params())
best_params = search.best_params_

clf.set_params(reg_alpha= best_params["reg_alpha"])
m_tests.append(test_model(X_test,y_test,clf.get_xgb_params()))

if track_tuning:
	labels = ["initial", "#trees", "depth & \n min_child_weight", "gamma", "reg. alpha"]
	figure()
	title("Performance on left-out dataset")
	test_performance = amap(lambda x: mean(x,0),m_tests)
	test_performance_s = amap(lambda x: std(x,0)/sqrt(len(x)),m_tests)
	errorbar(range(5),test_performance[:,-1],test_performance_s[:,-1],label="AUC")
	errorbar(range(5),test_performance[:,2],test_performance_s[:,2],label="sensitivity")
	errorbar(range(5),test_performance[:,1],test_performance_s[:,1],label="acc")
	xticks(range(5),labels)
	ylim(40,100)
	legend()

figure()
m_test = m_tests[0]
bar(range(4),nanmean(m_test,0)[[1,2,3,5]])
errorbar(range(4),nanmean(m_test,0)[[1,2,3,5]],nanstd(m_test,0)[[1,2,3,5]],fmt=None,color="k")

text(0-0.25,0.75,"%.2f" % nanmean(m_test,0)[1],fontsize=20,color="black")
text(1-0.25,0.75,"%.2f" % nanmean(m_test,0)[2],fontsize=20,color="black")
text(2-0.25,0.75,"%.2f" % nanmean(m_test,0)[3],fontsize=20,color="black")
text(3-0.25,0.75,"%.2f" % nanmean(m_test,0)[5],fontsize=20,color="black")

labels = ["acc","sensitivity", "specificity","AUC"]
ylim(0.5,1)
xticks(range(4),labels)
show()
