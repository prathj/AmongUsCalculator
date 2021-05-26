import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("C:/Users/prath/Desktop/amongproj/crewData.csv")
data.head()

# print(data.shape)
# print(data.columns)
# print(data.describe())

data.replace({'No': 0, 'Yes': 1}, inplace=True)
data.replace({'Loss': 0, 'Win': 1}, inplace=True)

X = data.drop(['Outcome'], 1)
X.head()

y = data['Outcome']
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
scaler = MinMaxScaler()

X_train[['Task Completed', 'All Tasks Completed', 'Murdered', 'Game Length', 'Ejected', 'Sabotages Fixed']] = \
    scaler.fit_transform(X_train[['Task Completed', 'All Tasks Completed', 'Murdered', 'Game Length', 'Ejected',
                                  'Sabotages Fixed']])
X_train.head()

logreg = LogisticRegression()
rfe = RFE(logreg, 15)
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train = X_train[col]

X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
res = logm2.fit()
res.summary()

logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
res = logm1.fit()

res.summary()
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by="VIF", ascending=False)

y_train_pred = res.predict(sm.add_constant(X_train))

y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'Conversion_Prob': y_train_pred})
y_train_pred_final.head()

y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted)
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs,
                                             drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return None


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob,
                                         drop_intermediate=False)

numbers = [float(x) / 10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

cutoff_df = pd.DataFrame(columns=['prob', 'accuracy', 'sensi', 'speci'])

num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i])
    total1 = sum(sum(cm1))
    accuracy = (cm1[0, 0] + cm1[1, 1]) / total1

    speci = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    sensi = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    cutoff_df.loc[i] = [i, accuracy, sensi, speci]
# print(cutoff_df)

cutoff_df.plot.line(x='prob', y=['accuracy', 'sensi', 'speci'])

y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.565 else 0)
y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

TP = confusion2[1, 1]
TN = confusion2[0, 0]
FP = confusion2[0, 1]
FN = confusion2[1, 0]

y_test_pred = res.predict(sm.add_constant(X_test))
# print(y_test_pred[:10])

y_pred_1 = pd.DataFrame(y_test_pred)
y_test_df = pd.DataFrame(y_test)

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)


# print(y_pred_final.head())


def calculateCrew(tasks, completed, murdered, game_length, ejected, sabotages):
    example = pd.DataFrame(np.array([[1, tasks, completed, murdered, game_length, ejected, sabotages]]),
                           columns=['Outcome', 'Task Completed', 'All Tasks Completed', 'Murdered', 'Game Length',
                                    'Ejected', 'Sabotages Fixed'])
    y_test_pred = res.predict(sm.add_constant(example))
    return y_test_pred


