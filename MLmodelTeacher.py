import os
import csv
import random
import metrics
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB

from sklearn.naive_bayes import GaussianNB


pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\all_features.csv"
#columns = ["FE1W", "FE1P", "FE2W", "FE2P", "FE3W", "FE3P", "FE4W", "FE4P", "FE7XW", "FE7XP", "FE7YW", "FE7YP", "FE7ZW", "FE7ZP", "FE8XW", "FE8XP", "FE8YW", "FE8YP", "FE8ZW", "FE8ZP", "FE9XW", "FE9XP", "FE9YW", "FE9YP", "FE9ZW", "FE9ZP",
 #                    "FE10XW", "FE10XP", "FE10YW", "FE10YP", "FE10ZW", "FE10ZP", "FE11XW", "FE11XP", "FE11YW", "FE11YP", "FE11ZW", "FE11ZP", "F12XW", "F12XP", "F12YW", "F12YP", "F12ZW", "F12ZP", "F5W", "F5P", "F6W", "F6P"]

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\wearable_all_features.csv"
columns = ["FE1W", "FE2W", "FE3W", "FE4W", "FE7XW", "FE7YW", "FE7ZW", "FE8XW", "FE8YW", "FE8ZW", "FE9XW", "FE9YW", "FE9ZW",
                     "FE10XW", "FE10YW", "FE10ZW", "FE11XW", "FE11YW", "FE11ZW", "F12XW", "F12YW", "F12ZW", "F5W", "F6W"]
"""

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\phone_all_features.csv"
columns = ["FE1P", "FE2P", "FE3P", "FE4P", "FE7XP", "FE7YP", "FE7ZP", "FE8XP", "FE8YP", "FE8ZP", "FE9XP", "FE9YP", "FE9ZP",
                     "FE10XP", "FE10YP", "FE10ZP", "FE11XP", "FE11YP", "FE11ZP", "F12XP", "F12YP", "F12ZP", "F5P", "F6P"]
"""

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\single_features.csv"
columns = ["FE7XW", "FE7XP", "FE7YW", "FE7YP", "FE7ZW", "FE7ZP", "FE8XW", "FE8XP", "FE8YW", "FE8YP", "FE8ZW", "FE8ZP", "FE9XW", "FE9XP", "FE9YW", "FE9YP", "FE9ZW", "FE9ZP",
                     "FE10XW", "FE10XP", "FE10YW", "FE10YP", "FE10ZW", "FE10ZP", "FE11XW", "FE11XP", "FE11YW", "FE11YP", "FE11ZW", "FE11ZP", "F12XW", "F12XP", "F12YW", "F12YP", "F12ZW", "F12ZP"]
"""


#pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\single_w_features.csv"
columns = ["FE7XW", "FE7YW", "FE7ZW", "FE8XW", "FE8YW", "FE8ZW", "FE9XW", "FE9YW", "FE9ZW",
                     "FE10XW", "FE10YW", "FE10ZW", "FE11XW", "FE11YW", "FE11ZW", "F12XW", "F12YW", "F12ZW"]


"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\single_p_features.csv"
columns = ["FE7XP", "FE7YP", "FE7ZP", "FE8XP", "FE8YP", "FE8ZP", "FE9XP", "FE9YP", "FE9ZP",
                     "FE10XP", "FE10YP", "FE10ZP", "FE11XP", "FE11YP", "FE11ZP", "F12XP", "F12YP", "F12ZP"]
"""

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\tied_features.csv"
columns = ["FE1W", "FE1P", "FE2W", "FE2P", "FE3W", "FE3P", "FE4W", "FE4P", "F5W", "F5P", "F6W", "F6P"]
"""

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\tied_w_features.csv"
columns = ["FE1W", "FE2W", "FE3W", "FE4W", "F5W", "F6W"]
"""

"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\tied_p_features.csv"
columns = ["FE1P", "FE2P", "FE3P", "FE4P", "F5P", "F6P"]
"""

#BEST TRIES
"""
pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\all_features.csv"
columns = ["FE1W", "FE1P", "FE11XW", "FE12XP", "FE12YW", "FE12YP", "FE12ZW", "FE12ZP"]
"""

#TEST DATA

#1
#test = [1429.0486345817626,31.861135729310885,0.3116256967832852,0.583426254416407,-0.21093666341885828,-0.08669783787823351,-0.22949349641185707,0.14572589453311985,228.39455782312925,-3.8643824259440103,50.48979591836735,4.321858574362362,59.80952380952381,-0.3058515062519148,16251.158326344235,39.45431746932541,13087.48448420464,28.52746823531106,16695.63470319635,9.155608651285316,127.48003108857573,6.281267186589456,114.4005440730272,5.341111142385174,129.21158888890866,3.0258236318869143,34142,606.9876861572266,11162,521.0641326904297,15100,213.16746520996094,1162,28.77471923828125,1130,31.0089111328125,916,20.104522705078125,4312.0,147.20372009277344,882.0000000000001,131.89993286132812,2058.0,142.94334411621094,181.5125148014334,6.972081907193198,214.55600088029516,8.782789668204618]
#1
#test = [1300.4337737847322,27.044514274563774,-0.2638580063634889,0.436723479068676,-0.23242586246479618,0.4449488002679045,0.11934415928954678,0.7171040750881457,25.561643835616437,-1.753698216806544,116.61643835616438,-1.002740992177831,107.43835616438356,-4.974029994246983,41276.82720831365,3.922525722359879,14779.231176192725,49.345908991641934,18714.35824279641,20.101036990929654,203.16699340275144,1.9805367258296118,121.56986129873113,7.0246643330227485,136.80043217328083,4.483418003145553,24564,219.7259979248047,17702,688.2848968505859,17338,570.1888885498047,1654,15.207183837890625,772,21.92291259765625,1176,31.473785400390625,924.6666666666667,155.15763346354169,438.0,131.4331029256185,292.0,132.09116872151694,244.9309809948714,4.901383754950181,273.44179751329676,8.565598152197632]
#0
#test = [892.8560914279524,26.938365094500668,0.3239080586226549,0.009646180376996938,0.047183069214647815,0.07319149312056916,0.6984585256998892,-0.2279065937204612,237.18032786885246,1.4569146184637995,170.75409836065575,8.978604156191986,30.42622950819672,-1.9458622696376082,29303.650273224044,7.502692750359812,31404.4218579235,37.98153300460903,2958.9486338797815,15.968954052224083,171.18308991610135,2.7391043701107507,177.2129280214158,6.162915949825133,54.39621892999349,3.99611737217816,15748,232.99649047851562,11204,942.3242797851562,2800,348.13929748535156,872,14.646133422851562,826,32.64094543457031,262,24.674636840820312,2033.3333333333333,353.49229431152344,1037.0,301.43497467041016,1057.3333333333333,274.4594141642253,179.61792479344544,4.844754565773575,252.3232465807051,7.83920785584825]
#0
#test = [492.3616556962981,23.472657633558764,0.29551114857362115,0.6365463330193232,0.03184405096671016,-0.635887639293032,0.4761760408236129,-0.8245655038789783,243.3793103448276,1.1572723388671875,88.48275862068965,7.6902349312317195,33.42068965517242,1.9560070319715979,3649.3065134099616,4.805288922214548,4824.404214559387,7.647836832752242,1871.550957854406,12.064933422316548,60.409490259477955,2.1920969235447934,69.45793125741211,2.7654722621556416,43.261425749209955,3.4734613028385026,35290,421.2255096435547,13138,1561.117691040039,5886,485.3819274902344,434,13.3668212890625,450,22.868621826171875,348,17.297317504882812,5123.333333333333,232.6160405476888,2368.3333333333335,220.06070200602215,966.6666666666666,227.5670598347982,74.3024728475733,4.107337622418091,101.71165953726128,4.95157138464986]

#ACCURACY
accuracy = []

#LOADING DATA
data = []
with open(pathDatasetFile, 'r', newline='') as datasetfile:
    reader = csv.reader(datasetfile, delimiter=';')
    next(reader, None)
    for row in reader:
        data.append(row)


data = np.array(data)
df = pd.DataFrame(data[:, np.arange(8, 44, 2)], columns=columns) #change nr of columns in file
y = data[:,48] #last column in file
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

#MODELS
#svm
""""
clf = svm.SVC(kernel='rbf', probability=0, C=1).fit(X_train, y_train)
#FE48W+P linear: 87-97(long fitting), poly:70-85, rbf: 79-94, sigmoid: 50-67, precomputed: (needs square matrix)
#FE24W linear: 86-97(long fitting)
#FE24P 83-94
#FESWP 90-97
#FESW 94-98
#FESP 81-93
#FETWP 89-98
#FETW 90-97
#FETP 72-85
#pickle.dump(clf, open("SVM-18P", 'wb'))
"""
#clf = pickle.load(open("SVM-18W", 'rb'))

#logistic regression
"""
clf = LogisticRegression(solver='liblinear', C=10).fit(X_train, y_train)
#liblinear: 87-100(no error), sag/saga: 80-93(with error), lbfgs:90-97(with errors), newton-cg: 89-98 (error/no error)
#FE24W liblinear: 87-98
#FE24P 81-94
#FESWP 87-98
#FESW 90-100
#FESP 79-91
#FETWP 86-97
#FETW 87-97
#FETP 70-85
#pickle.dump(clf, open("LR-24W", 'wb'))
"""
#clf = pickle.load(open("LR-24W", 'rb'))

#random forest

#clf = RandomForestClassifier(criterion='gini', n_estimators=10, min_samples_leaf=2).fit(X_train, y_train)
#gini: 93-100, entropy: 93-100
#FE24W 93-100
#FE24P 87-97
#FESWP 93-100
#FESW 90-100
#FESP 85-95
#FETWP 89-100
#FETW 89-98
#FETP 78-95
#n_estimators: (3) 87-97, (10) 90-98, (100) 93-100, (1000) 93-100, (10000) 93-100
#pickle.dump(clf, open("WEAR", 'wb'))

clf = pickle.load(open("WEAR", 'rb'))

#decision tree
"""
clf = DecisionTreeClassifier(criterion='entropy', max_features='auto', min_samples_leaf=2).fit(X_train, y_train)
#gini: 87-100, entropy: 86-100
#best: better ranodm: worse
#FE24W 87-100
#FE24P 79-94
#FESWP 87-98
#FESW 87-100
#FESP 81-93
#FETWP 83-97
#FETW 85-97
#FETP 72-89
#pickle.dump(clf, open("DT-BST", 'wb'))
"""
#clf = pickle.load(open("DT-BST", 'rb'))

#gaussian naive Bayes
"""
clf = GaussianNB().fit(X_train, y_train)
# 87-97
#FE24W 89-98
#FE24P 71-87
#FESWP 85-97
#FESW 89-98
#FESP 68-87
#FETWP 85-98
#FETW 87-97
#FETP 71-86
#pickle.dump(clf, open("GNB-24W", 'wb'))
"""
#clf = pickle.load(open("GNB-24W", 'rb'))

#SCORING
print(clf.score(X_test, y_test))
#PREDICTING
#print(clf.predict([test]))
