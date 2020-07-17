import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib as mpl

from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn import preprocessing
from sklearn import neighbors

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from sklearn.linear_model import Ridge, RidgeCV

import timeit

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import tree

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
##########################################################################
############################# DATA CLEANING ##############################
##########################################################################

#Importing datasets
resp = pd.read_csv('atusresp_2018.csv') #labor force status and earnings
roster = pd.read_csv('atusrost_2018.csv') #household members info
act = pd.read_csv('atusact_2018.csv') #activity diary
who = pd.read_csv('atuswho_2018.csv') #who was present during each activity

#Merging Datasets
master = pd.merge(who, act, on = ['TUCASEID', 'TUACTIVITY_N'], how = 'right')
master = pd.merge(roster, master, on = ['TUCASEID', 'TULINENO'], how = 'right')
master = pd.merge(resp, master, on = ['TUCASEID', 'TULINENO'], how = 'right') 

master.to_csv('master.csv')

master = pd.read_csv('master.csv')

#Creating dummies for starting hour to account for time of day
master['StartHour'] = master['TUSTARTTIM'].str.split(':')
master['StartHour'] = master['TUSTARTTIM'].str.split(':', expand = True)
master['StartHour'] = master['StartHour'].astype('int') 
master = pd.get_dummies(master, columns=['StartHour'])
print(master.shape)

#categorical labels as seen in the do files
M = pd.get_dummies(master, columns=['TRWHONA','TUWHO_CODE','TERRP','TESEX','TEWHERE','TUCC5', 
	'TUCC5B','TUCC7','TUCC8','TUEC24','TUDURSTOP','TEABSRSN','TEERNHRY','TEERNPER','TEERNRT',
	'TEERNUOT','TEHRFTPT','TEIO1COW','TELAYAVL','TELAYLK','TELFS','TELKAVL','TELKM1','TEMJOT',
	'TERET1','TESCHENR','TESCHFT','TESCHLVL','TESPEMPNOT','TRDPFTPT','TRDTOCC1','TRERNUPD','TRHERNAL',
	'TRHHCHILD','TRHOLIDAY','TRIMIND1','TRLVMODR','TRMJIND1','TRMJOCC1','TRMJOCGR','TRNHHCHILD','TROHHCHILD',
	'TRSPFTPT','TRSPPRES','TRWERNAL','TTHR','TTOT','TTWK','TUABSOT','TUBUS','TUBUS1','TUBUS2OT','TUDIARYDAY',
	'TUDIS','TUDIS1','TUDIS2','TUECYTD','TUELDER','TUELFREQ','TUFWK','TUIO1MFG','TUIODP1','TUIODP2','TUIODP3',
	'TULAY','TULAY6M','TULAYAVR','TULAYDT','TULK','TULKAVR','TULKDK1','TULKDK2','TULKDK3','TULKDK4','TULKM2',
	'TULKM3','TULKM4','TULKPS1','TULKPS2','TULKPS3','TULKPS4','TURETOT','TUSPABS','TUSPUSFT','TUSPWK'])

#dropping null value columns to avoid errors 
print(M.isna().sum(axis=0))
M.dropna(inplace=True)
M = M.drop(['Unnamed: 0'], axis=1)
print(M.shape) #Seeing how many observations were dropped
M.to_csv('M.csv')

M = pd.read_csv('M.csv')
M = M[:20000]
print(M.shape)

all_lexecons = ['TRCODE', 'TRTIER2', 'TUTIER1CODE']
code_variables = ['TUTIER2CODE', 'TUTIER3CODE']
repetitive_variables = ['TUCUMDUR', 'TUCUMDUR24']

#Splits data into 70-20-10 training-validation-test split
train, validate, test = np.split(M.sample(frac=1), [int(.7*len(M)), int(.9*len(M))])

#Dropping Variables
#TUCASEID, TULINENO are person IDs, TUSTARTTIM TUSTOPTIME are start stop times which are strings and are accounted for above
#TUCC2 is Time first household child < 13 woke up
#TUCC4 is Time last household child < 13 went to bed which are also strings and not relevant for the analysis
#TUDIARYDATE is in form of '20180131' which is not relevant for the analysis
#TUFINLWGT final weight of the person representative of their demographich which is also not relevant for the analysis
drop_var = ['TUCASEID', 'TULINENO', 'TUSTARTTIM', 'TUSTOPTIME', 'TUCC2','TUCC4','TUDIARYDATE', 'TUFINLWGT',] + all_lexecons + code_variables + repetitive_variables

#Getting X and Y for each dataset
X_training = train.drop(drop_var, axis=1)
X_validate = validate.drop(drop_var, axis=1)
X_testing = test.drop(drop_var, axis=1)

Y_training = train[all_lexecons]
Y_validate = validate[all_lexecons]
Y_testing = test[all_lexecons]

##########################################################################
############################ MODELS ######################################
##########################################################################

# NAIVE BAYES BERNOULLI 
def Bernoulli(X_train, Y_train, X_test, Y_test):
	clf = BernoulliNB()
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Bernoulli Test Error: ' + str(test_error))
	print('Bernoulli Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Number of samples in each class during fitting: ')
	print(clf.class_count_)
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	print("Each model coefficient is: ")
	print(clf.feature_count_)
	print("Number of features: ")
	print(clf.n_features_)
	return clf, test_error, acc_score


# NAIVE BAYES GAUSSIAN
def Gaussian(X_train, Y_train, X_test, Y_test):
	clf = GaussianNB()
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Gaussian Test Error: ' + str(test_error))
	print('Gaussian Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Number of samples in each class during fitting: ')
	print(clf.class_count_)
	print("Probability of each class: ")
	print(clf.class_prior_)
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	print("Variance of each feature per class: ")
	print(clf.sigma_)
	print("Mean of each feature per class: ")
	print(clf.theta_)
	return clf, test_error, acc_score

# NEAREST CENTROID
def Centroid(X_train, Y_train, X_test, Y_test):
	# Parameter 'shrinkage' is tuned 
	#Cross validation
	shrinkages = np.linspace(0, 10, 100)
	tuned_parameters = [{'shrink_threshold': shrinkages}]
	cv = GridSearchCV(NearestCentroid(), tuned_parameters)
	cv.fit(X_train, Y_train)
	
	#Optimal parameters
	print('Best Params: ')
	print(cv.best_params_)

	#Optimal Model
	clf = NearestCentroid()
	clf.set_params(shrink_threshold=cv.best_params_['shrink_threshold'])
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Nearest Centroid Test Error: ' + str(test_error))
	print('Nearest Centroid Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Centroid of each class: ')
	print(clf.centroids_[0])
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	return clf, test_error, acc_score


# KNN
def KNN(X_train, Y_train, X_test, Y_test):
	# Parameter 'number of neighbors' is tuned 

	#Cross validation
	neighbors = np.linspace(1, 50, 50).astype(int)
	tuned_parameters = [{'n_neighbors': neighbors}]
	cv = GridSearchCV(KNeighborsClassifier(), tuned_parameters)
	cv.fit(X_train, Y_train)
	
	#Optimal parameters
	print('Best Params: ')
	print(cv.best_params_)

	#Optimal Model
	knn = KNeighborsClassifier()
	knn.set_params(n_neighbors=cv.best_params_['n_neighbors'])
	knn.fit(X_train, Y_train)
	pred = knn.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('K Nearest Neighbors Test Error: ' + str(test_error))
	print('K Nearest Neighbors Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Class labels known to the classifier: ')
	print(knn.classes_)
	return knn, test_error, acc_score


# LOGISTIC REGRESSION
def Logistic(X_train, Y_train, X_test, Y_test):
	clf = skl_lm.LogisticRegression(solver='newton-cg', max_iter=10000, multi_class='multinomial')
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Logistic Regression Test Error: ' + str(test_error))
	print('Logistic Regression Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	print('Coefficient of the features in the decision function: ')
	print(clf.coef_)
	return clf, test_error, acc_score


# DECISION TREE CLASSIFIER 
def Tree(X_train, Y_train, X_test, Y_test):
	# Parameter 'number of features' is tuned 

	#Cross validation
	n_features = np.linspace(1, len(X_train.columns), len(X_train.columns))
	tuned_parameters = [{'max_features': n_features}]
	cv = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters)
	cv.fit(X_train, Y_train)
	
	#Optimal parameters
	print('Best Params: ')
	print(cv.best_params_)

	#Optimal Model
	clf = tree.DecisionTreeClassifier()
	clf.set_params(max_features=cv.best_params_['max_features'])
	clf.fit(X_train, Y_train)
	pred = clf.predict(X_test)
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Decision Tree classifier Test Error: ' + str(test_error))
	print('Decision Tree Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	print('Feature importances: ')
	print(clf.feature_importances_)
	print('N features: ')
	print(clf.n_features_)
	print('Max features: ')
	print(clf.max_features_)
	print('N outputs: ')
	print(clf.n_outputs_)
	
	importance = pd.Series(abs(clf.feature_importances_) * 100, index = X_train.columns)
	importance = importance.sort_values(axis=0, ascending = False)[:10]
	print('Top 10 Most Important Features: ')
	print(importance)

	return clf, test_error, acc_score


# NEURAL NETWORK 
def Neural(X_train, Y_train, X_test, Y_test):

	#Cross validation
	#tuned_parameters = {
	#'max_iter': np.arange(1000, 10000, 10), 
	#'learning_rate_init': np.linspace(0.1, 1, 10), 
	#'alpha': 10.0 ** -np.arange(1, 10), 
	#'hidden_layer_sizes':np.arange(10, 50, 5),}
	#cv = GridSearchCV(MLPClassifier(), tuned_parameters)
	#cv.fit(X_train, Y_train)

	#Optimal parameters
	#print('Best Params: ')
	#print(cv.best_params_)
	
	#Standardizing Variables
	scaler = StandardScaler()

	#Optimal Model
	clf = MLPClassifier()
	#clf.set_params(activation='logistic', max_iter=cv.best_params_['max_iter'], alpha=cv.best_params_['alpha'], hidden_layer_sizes=cv.best_params_['hidden_layer_sizes'], learning_rate_init=cv.best_params_['learning_rate_init'])
	clf.set_params(activation='logistic', max_iter=1000, alpha=0.05, hidden_layer_sizes=380, learning_rate_init=0.09)
	clf.fit(scaler.fit_transform(X_train), Y_train)
	pred = clf.predict(scaler.fit_transform(X_test))
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred)
	print('Neural Network Test Error: ' + str(test_error))
	print('Neural Network Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	print('Class labels known to the classifier: ')
	print(clf.classes_)
	print('Loss: ')
	print(clf.loss_) 
	print('Coefficient: ')
	print(clf.coefs_)
	print('N outputs: ')
	print(clf.n_outputs_)
	return clf, test_error, acc_score


# RIDGE CLASSIFIER 
def Ridge(X_train, Y_train, X_test, Y_test):
	# Parameter 'alpha' is tuned 
	#Standardizing Variables
	scaler = StandardScaler()

	#Cross validation
	alphas = 10**np.linspace(10,-2,100)
	ridgecv = skl_lm.RidgeCV(alphas=alphas, scoring='neg_mean_squared_error')
	ridgecv.fit(scaler.fit_transform(X_train), Y_train)
	
	#Optimal parameters
	print('Best Params: ')
	print(ridgecv.alpha_)

	#Optimal Model
	optimal_ridge = skl_lm.Ridge()
	optimal_ridge.set_params(alpha=ridgecv.alpha_)
	optimal_ridge.fit(scaler.fit_transform(X_train), Y_train)
	pred = optimal_ridge.predict(scaler.fit_transform(X_test))
	test_error = mean_squared_error(Y_test, pred)
	acc_score = accuracy_score(Y_test, pred.round())
	print('Ridge Regression Test Error: ' + str(test_error))
	print('Ridge Regression Accuracy Score: ' + str(acc_score))
	print('First 10 predictions: ')
	print(pred[:10])
	print('First 10 actual: ')
	print(Y_test[:10])
	#print('Class labels known to the classifier: ')
	#print(optimal_ridge.classes_)
	print(pd.Series(optimal_ridge.coef_.flatten(), index=X_train.columns))
	return optimal_ridge, test_error, acc_score

def FindTestMSE(model, X_test, Y_test, ridge=False):

	if ridge == True:
		#Standardizing Variables
		scaler = StandardScaler()
		pred = model.predict(scaler.fit_transform(X_test))
		test_error = mean_squared_error(Y_test, pred)
		acc_score = accuracy_score(Y_test, pred.round())
		print('Test Error: ' + str(test_error))
		print('Accuracy Score: ' + str(acc_score))
		return test_error, acc_score
	else:
		pred = model.predict(X_test)
		test_error = mean_squared_error(Y_test, pred)
		acc_score = accuracy_score(Y_test, pred)
		print('Test Error: ' + str(test_error))
		print('Accuracy Score: ' + str(acc_score))
		return test_error, acc_score


##########################################################################
############################ TRAINING ####################################
##########################################################################

start = timeit.default_timer()

#Split training and test data within 'Train' dataset 
X_train_R, X_test_R, Y_train_R, Y_test_R = train_test_split(X_training, Y_training, test_size=0.3, random_state=22)

#Testing Models 
summary_R = {}
summary_R_acc_score = {}

warnings.filterwarnings("ignore") 
tree_R, tree_MSE, tree_acc_score = Tree(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Tree': tree_MSE})
summary_R_acc_score.update({'Tree': tree_acc_score})

knn_R, knn_MSE, knn_acc_score = KNN(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'KNN': knn_MSE})
summary_R_acc_score.update({'KNN': knn_acc_score})

neural_R, neural_MSE, neural_acc_score = Neural(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Neural': neural_MSE})
summary_R_acc_score.update({'Neural': neural_acc_score})

bernoulli_R, bernoulli_MSE, bernoulli_acc_score = Bernoulli(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Bernoulli': bernoulli_MSE})
summary_R_acc_score.update({'Bernoulli': bernoulli_acc_score})

gaussian_R, gaussian_MSE, gaussian_acc_score = Gaussian(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Gaussian': gaussian_MSE})
summary_R_acc_score.update({'Gaussian': gaussian_acc_score})

centroid_R, centroid_MSE, centroid_acc_score = Centroid(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Centroid': centroid_MSE})
summary_R_acc_score.update({'Centroid': centroid_acc_score})

ridge_R, ridge_MSE, ridge_acc_score = Ridge(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
summary_R.update({'Ridge': ridge_MSE})
summary_R_acc_score.update({'Ridge': ridge_acc_score})

#should run but takes a while
#logistic_R, logistic_MSE, logistic_acc_score = Logistic(X_train_R, Y_train_R['TUTIER1CODE'], X_test_R, Y_test_R['TUTIER1CODE'])
#summary_R.update({'Logistic': logistic_MSE})
#summary_R_acc_score.update({'Logistic': logistic_acc_score})

print('Training Summary')
print(summary_R)
print(summary_R_acc_score)

#Find optimal model
min_MSE = min(summary_R.values()) 
for model, MSE in summary_R.items():  
    if MSE == min_MSE:
        print(model) 

##########################################################################
########################## Validation ####################################
##########################################################################

summary_V = {}
summary_V_acc_score = {}

bernoulli_V_MSE, bernoulli_V_acc_score = FindTestMSE(bernoulli_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'Bernoulli': bernoulli_V_MSE})
summary_V_acc_score.update({'Bernoulli': bernoulli_V_acc_score})

gaussian_V_MSE, gaussian_V_acc_score = FindTestMSE(gaussian_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'Gaussian': gaussian_V_MSE})
summary_V_acc_score.update({'Gaussian': gaussian_V_acc_score})

centroid_V_MSE, centroid_V_acc_score = FindTestMSE(centroid_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'Centroid': centroid_V_MSE})
summary_V_acc_score.update({'Centroid': centroid_V_acc_score})

#logistic_V_MSE, logistic_V_acc_score = FindTestMSE(logistic_R, X_validate, Y_validate['TUTIER1CODE'])
#summary_V.update({'Logistic': logistic_V_MSE})
#summary_V_acc_score.update({'Logistic': logistic_V_acc_score})

ridge_V_MSE, ridge_V_acc_score = FindTestMSE(ridge_R, X_validate, Y_validate['TUTIER1CODE'], ridge=True)
summary_V.update({'Ridge': ridge_V_MSE})
summary_V_acc_score.update({'Ridge': ridge_V_acc_score})

tree_V_MSE, tree_V_acc_score = FindTestMSE(tree_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'Tree': tree_V_MSE})
summary_V_acc_score.update({'Tree': tree_V_acc_score})

knn_V_MSE, knn_V_acc_score = FindTestMSE(knn_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'KNN': knn_V_MSE})
summary_V_acc_score.update({'KNN': knn_V_acc_score})

neural_V_MSE, neural_V_acc_score = FindTestMSE(neural_R, X_validate, Y_validate['TUTIER1CODE'])
summary_V.update({'Neural': neural_V_MSE})
summary_V_acc_score.update({'Neural': neural_V_acc_score})

print('Validation Summary')
print(summary_V)

#Find optimal model
min_MSE = min(summary_V.values()) 
for model, MSE in summary_V.items():
    if MSE == min_MSE:
        print(model)

##########################################################################
############################# Test #######################################
##########################################################################

frames = [train, validate]
final_train = pd.concat(frames)

X_final_train = final_train.drop(drop_var, axis=1)
Y_final_train = final_train[all_lexecons]

tree_F, tree_MSE_F, tree_acc_score_F = Tree(X_final_train, Y_final_train['TUTIER1CODE'], X_testing, Y_testing['TUTIER1CODE'])

stop = timeit.default_timer()
print('Time: ', stop - start)