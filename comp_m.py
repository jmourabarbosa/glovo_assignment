from pickle import *
from numpy import *
from matplotlib.pylab import *

labels = ["initial", "#trees", "depth & \n min_child_weight", "gamma", "reg. alpha"]


up_f=open("pickles/m_test_upsampling_filling.pickle")
up_nf=open("pickles/m_test_no_filling.pickle")
no_up_f=open("pickles/m_test_no_up_samp.pickle")
no_up_nf=open("pickles/m_test_no_upsampling_no_filling.pickle")
up_f_n=open("pickles/m_test_upsampling_filling_norm.pickle")

no_up_f=load(no_up_f)
up_f=load(up_f)
up_nf=load(up_nf)
no_up_nf=load(no_up_nf)
up_f_n=load(up_f_n)


m_tests = [array(up_f)*100,array(no_up_f)*100,array(up_nf)*100,array(up_f_n)*100]

test_labels = ["reference: w/ up, w/filling", "-up", "-filling", "+normalizatio"]
for i,m_test in enumerate(m_tests):
	figure()
	title(test_labels[i])
	test_performance = amap(lambda x: mean(x,0),m_test)
	test_performance_s = amap(lambda x: std(x,0)/sqrt(len(x)),m_test)
	errorbar(range(5),test_performance[:,-1],test_performance_s[:,-1],label="AUC")
	errorbar(range(5),test_performance[:,2],test_performance_s[:,2],label="sensitivity")
	# errorbar(range(5),test_performance[:,-2],test_performance_s[:,-2],label="FPR")
	errorbar(range(5),test_performance[:,1],test_performance_s[:,1],label="acc")
	xticks(range(5),labels)
	ylim(40,100)
	legend()

figure()
test =array([ amap(lambda x: mean(x,0),m) for m in m_tests])

# up samp
auc_dif = test[0,:,0] - test[1,:,0]
sensi_dif = test[0,:,2] - test[1,:,2]

subplot(1,2,1)
title("AUC")
plot(auc_dif,"k",label="-up samp")
plot(range(5),zeros(5),"k--",alpha=0.5)
xlabel("paremter tuning")
ylabel("% improvement")
ylim(-50,100)
xticks(range(5),labels)

subplot(1,2,2)
title("SENSITIVITY")
plot(sensi_dif,"k",label="-up samp")
plot(range(5),zeros(5),"k--",alpha=0.5)
ylim(-50,100)
yticks([])
xticks(range(5),labels)


# filling
auc_dif = test[0,:,0] - test[2,:,0]
sensi_dif = test[0,:,2] - test[2,:,2]

subplot(1,2,1)
plot(auc_dif,"r",label="-filling")
subplot(1,2,2)
plot(sensi_dif,"r",label="-filling")

# normalizing
auc_dif = test[0,:,0] - test[3,:,0]
sensi_dif = test[0,:,2] - test[3,:,2]

subplot(1,2,1)
plot(auc_dif,"b",label="+normalization")
subplot(1,2,2)
plot(sensi_dif,"b",label="+normalization")




show()