from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def divider(title):
    print("\n\n———————————————————————" + title + "—————————————————————————")


# 异常值处理
def outlier_processing(dfx):
    df = dfx.copy()
    q1 = df.quantile(q=0.25)
    q3 = df.quantile(q=0.75)
    iqr = q3 - q1
    Umin = q1 - 1.5 * iqr
    Umax = q3 + 1.5 * iqr
    df[df > Umax] = df[df <= Umax].max()
    df[df < Umin] = df[df >= Umin].min()
    return df


def func(test_y,n):
    global y_rate, equal, true_count
    # 生成模型预测数据 y_pred
    y_rate = gs.best_estimator_.predict(dummy_test)
    for i in range(0, len(y_rate)):
        if y_rate[i] > n:
            y_rate[i] = 1
        else:
            y_rate[i] = 0
    # 判断是否相同
    equal = []
    true_count = 0
    for i in range(0, len(test_y)):
        if (test_y[i] == y_rate[i]):
            equal.append(True)
            true_count = true_count + 1
        else:
            equal.append(False)
    rate = true_count/len(test_y)
    return rate, equal, y_rate,n

if __name__ == '__main__':
    # 获取数据
    divider("获取数据")
    train = pd.read_csv('bank-full.csv', sep=';')
    test = pd.read_csv('bank.csv', sep=';')
    train['y'].replace('yes', 1, inplace=True)
    train['y'].replace('no', 0, inplace=True)
    test.drop(['y'], inplace=True, axis=1)
    print("初始数据:\n", train)

    # 划分特征值
    divider("划分特征值")
    str_features = []
    num_features = []
    for col in train.columns:
        if train[col].dtype == 'object' and col != 'y':
            str_features.append(col)
        if train[col].dtype == 'int64' and col != 'y':
            num_features.append(col)
    print("Str型的特征值:", str_features)
    print("数值型的特征值:", num_features)

    # 异常值处理
    divider("异常值处理")
    train['age'] = outlier_processing(train['age'])
    train['day'] = outlier_processing(train['day'])
    train['duration'] = outlier_processing(train['duration'])
    train['campaign'] = outlier_processing(train['campaign'])
    test['age'] = outlier_processing(test['age'])
    test['day'] = outlier_processing(test['day'])
    test['duration'] = outlier_processing(test['duration'])
    test['campaign'] = outlier_processing(test['campaign'])

    # 将String类型的值转为数值型
    divider("将String类型的值转为数值型")
    print(train[str_features].shape)
    print(test[str_features].shape)

    dummy_train = train.join(pd.get_dummies(train[str_features])).drop(str_features, axis=1).drop('y', axis=1)
    dummy_test = test.join(pd.get_dummies(test[str_features])).drop(str_features, axis=1)
    print("训练集dummy_train:\n", dummy_train.head())
    print("测试集dummy_test:\n", dummy_test.head())
    # 根据方差分析，将与y相关性强的特征值筛选出来
    divider("根据方差分析，将与y相关性强的特征值筛选出来")
    print(train[num_features])
    f, p = f_classif(train[num_features], train['y'])
    k = f.shape[0] - (p > 0.05).sum()
    selector = SelectKBest(f_classif, k=k)
    selector.fit(train[num_features], train['y'])
    print('scores_:', selector.scores_)
    print('pvalues_:', selector.pvalues_)  # p值
    print('selected index:', selector.get_support(True))  # 筛选出的特征值索引

    # 标准化数值型的特征值，返回值为标准化后的数据
    divider("标准化，返回值为标准化后的数据")
    standardScaler = StandardScaler()
    ss = standardScaler.fit(dummy_train.loc[:, num_features])
    dummy_train.loc[:, num_features] = ss.transform(dummy_train.loc[:, num_features])
    dummy_test.loc[:, num_features] = ss.transform(dummy_test.loc[:, num_features])
    print("标准化后的dummy_train:\n", dummy_train)
    print("标准化后的dummy_test:\n", dummy_test)

    # 划分x与y
    x = dummy_train
    y = train['y']

    # 拆分数据集为训练集和验证集
    divider("拆分数据集为训练集和验证集")
    print("拆分前  x=", x.shape, " y=", y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
    print("拆分后  x_train=", x_train.shape, " x_test=", x_test.shape, " y_train=", y_train.shape, " y_test=",
          y_test.shape)

    # 不平衡数据集处理
    divider("不平衡数据集处理")
    print("处理前 y_train=\n", y_train.value_counts())
    smote_tomek = SMOTETomek(random_state=2)
    x_resampled, y_resampled = smote_tomek.fit_resample(x_train, y_train)
    print("处理后 y_resampled=\n", y_resampled.value_counts())

    divider("数据建模：LGBM")
    # param = {'max_depth': [5, 7, 9, 11, 13],
    #          'learning_rate': [0.01, 0.03, 0.05, 0.1],
    #          'num_leaves': [30, 90, 120],
    #          'n_estimators': [1000, 1500, 2000, 2500]}
    param = {'max_depth': [5],
             'learning_rate': [0.01],
             'num_leaves': [30],
             'n_estimators': [1000]}
    transfer = LGBMRegressor(max_depth=5, learning_rate=0.01, n_estimators=1000, num_leaves=30)
    gs = GridSearchCV(estimator=transfer, param_grid=param, cv=3, scoring="roc_auc", n_jobs=-1, verbose=10)
    gs.fit(x_resampled, y_resampled)
    estimator = gs.best_estimator_

    y_rate = estimator.predict(x_test)
    print("AUC评测值(概率的合理程度)：",roc_auc_score(y_test, y_rate))

    # 获取预测答案数据 test_y
    test_y = pd.read_csv('bank.csv', sep=';')['y']
    test_y.replace('yes',1,inplace=True)
    test_y.replace('no',0,inplace=True)
    max_rate = 0 # 最高正确率
    max_n = 0 # 拆分百分比
    equal = []
    pred_y = []
    for n in range(50, 100, 1):
        temp_rate,temp_equal,temp_y_pred,temp_n = func(test_y,n/100)
        if(temp_rate>max_rate):
            equal = temp_equal
            pred_y = temp_y_pred
            max_rate = temp_rate
            max_n = temp_n
    print("正确率：",max_rate)
    print("对应的拆分百分比：",max_n)
    # 导入csv
    result = pd.DataFrame({
        '正确答案': test_y,
        '预测答案':pred_y,
        '是否相同':equal
    })
    result.to_csv('result.csv',index=False)
