import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from datetime import datetime
from tensorflow.python import keras
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow.python.keras.backend as K
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import time

Train_data = pd.read_csv("C:\\Users\\will\\Desktop\\used_car-master\\data1\\used_car_train_20200313.csv", sep=' ')
Test_data = pd.read_csv("C:\\Users\\will\\Desktop\\used_car-master\\data1\\used_car_testB_20200421.csv", sep=' ')
start = time.perf_counter()
df = pd.concat([Train_data, Test_data], ignore_index=True)


feature = ['model', 'brand', 'bodyType', 'fuelType', 'kilometer', 'notRepairedDamage', 'power', 'regDate_month',
           'creatDate_year', 'creatDate_month'
    , 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
           'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14', 'car_age_day', 'car_age_year', 'regDate_year',
           'name_count']


df.drop(df[df['seller'] == 1].index, inplace=True)
df_copy = df
df['power'][df['power'] > 600] = 600

df.replace(to_replace='-', value=0.5, inplace=True)
le = LabelEncoder()
df['notRepairedDamage'] = le.fit_transform(df['notRepairedDamage'].astype(str))


from datetime import datetime


def date_process(x):
    year = int(str(x)[:4])
    month = int(str(x)[4:6])
    day = int(str(x)[6:8])

    if month < 1:
        month = 1

    date = datetime(year, month, day)
    return date


df['regDates'] = df['regDate'].apply(date_process)
df['creatDates'] = df['creatDate'].apply(date_process)
df['regDate_year'] = df['regDates'].dt.year
df['regDate_month'] = df['regDates'].dt.month
df['regDate_day'] = df['regDates'].dt.day
df['creatDate_year'] = df['creatDates'].dt.year
df['creatDate_month'] = df['creatDates'].dt.month
df['creatDate_day'] = df['creatDates'].dt.day
df['car_age_day'] = (df['creatDates'] - df['regDates']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)
df['name_count'] = df.groupby(['name'])['SaleID'].transform('count')


df.fillna(df.median(), inplace=True)


scaler = MinMaxScaler()
scaler.fit(df[feature].values)
df = scaler.transform(df[feature].values)



nn_data = pd.DataFrame(df, columns=feature)
nn_data['price'] = np.array(df_copy['price'])
nn_data['SaleID'] = np.array(df_copy['SaleID'])
print(nn_data.shape)
train_num = df.shape[0] - 50000
nn_data[0:int(train_num)].to_csv('C:/Users/will/Desktop/used_car-master/data1/' + 'train_nn.csv', index=0, sep=' ')
nn_data[train_num:train_num + 50000].to_csv('C:/Users/will/Desktop/used_car-master/data1/' + 'test_nn.csv', index=0,
                                            sep=' ')


path = 'C:/Users/will/Desktop/used_car-master/data1/'

Train_data = pd.read_csv(path + 'train_tree.csv', sep=' ')
TestA_data = pd.read_csv(path + 'text_tree.csv', sep=' ')

# print(Train_data.info())
# print(TestA_data.info())


a = [Train_data, TestA_data]
for j in a:
    for i in j.columns:
        dtype = j[i].dtype
        if dtype == 'float64':
            j[i] = j[i].astype(np.float32)
        elif dtype == 'int64':
            j[i] = j[i].astype(np.int32)
numerical_cols = Train_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price', 'SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_data[feature_cols]
X_test = TestA_data[feature_cols]
# print(X_data.shape)
# print(X_test.shape)

X_data = np.array(X_data)
X_test = np.array(X_test)
Y_data = np.array(Train_data['price'])
# #缩减运算开销
# X_data = X_data[0:5000]
# X_test = X_test[0:5000]
# Y_data = Y_data[0:5000]
# print(X_data[0:1000].shape)
# print(Y_data.shape)
star = time.perf_counter()
"""
lightgbm
"""


# 自定义损失函数
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_absolute_error(np.expm1(label), np.expm1(preds))
    return 'myFeval', score, False


param = {'boosting_type': 'gbdt',
         'num_leaves': 31,
         'max_depth': -1,
         "lambda_l2": 2,  # 防止过拟合
         'min_data_in_leaf': 20,  # 防止过拟合，好像都不用怎么调
         'objective': 'regression_l1',
         'learning_rate': 0.01,
         "min_child_samples": 20,

         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8,
         "bagging_seed": 11,
         "metric": 'mae',
         }
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(X_data))
predictions_lgb = np.zeros(len(X_test))
predictions_train_lgb = np.zeros(len(X_data))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_data, Y_data)):  # trn_idx训练集idx，val_idx测试集idx
    print("fold n°{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_data[trn_idx], Y_data[trn_idx])
    val_data = lgb.Dataset(X_data[val_idx], Y_data[val_idx])

    num_round = 1
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], feval=myFeval)
    oof_lgb[val_idx] = clf.predict(X_data[val_idx], num_iteration=clf.best_iteration)  # 测试集预测
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits  # 验证集预测
    predictions_train_lgb += clf.predict(X_data, num_iteration=clf.best_iteration) / folds.n_splits


output_path = 'C:/Users/will/Desktop/used_car-master/data1/'
# 测试集输出
predictions = predictions_lgb
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = TestA_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path + 'lgb_test.csv', index=False)

# 验证集输出
oof_lgb[oof_lgb < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_data.SaleID
sub['price'] = oof_lgb
sub.to_csv(output_path + 'lgb_train.csv', index=False)
print('lgb完成训练')
end = time.perf_counter()
print(star - end)

# ## 读取神经网络模型数据
tree_data_path = 'C:/Users/will/Desktop/used_car-master/data1/'
Train_NN_data = pd.read_csv(tree_data_path + 'train_nn.csv', sep=' ')
Test_NN_data = pd.read_csv(tree_data_path + 'test_nn.csv', sep=' ')
# 转换精度
a = [Train_NN_data, Test_NN_data]
for j in a:
    for i in j.columns:
        dtype = j[i].dtype
        if dtype == 'float64':
            j[i] = j[i].astype(np.float32)
        elif dtype == 'int64':
            j[i] = j[i].astype(np.int32)

per = []
for i in Train_NN_data:
    per.append([i, Train_NN_data['price'].corr(Train_NN_data[i])])
per1 = []
per2 = []
for i in per:
    per1.append(i[0])
    per2.append(i[1])

per3 = []
for i in per:
    if -0.15 < i[1] < 0.15:
        per3.append(i[0])

per3.remove('SaleID')
per3.remove('regDate_month')
per3.remove('creatDate_year')
per3.remove('creatDate_month')
per3.remove('model')

Train_NN_data.drop(per3, axis=1, inplace=True)
Test_NN_data.drop(per3, axis=1, inplace=True)

numerical_cols = Train_NN_data.columns
feature_cols = [col for col in numerical_cols if col not in ['price', 'SaleID']]
## 提前特征列，标签列构造训练样本和测试样本
X_data = Train_NN_data[feature_cols]
X_test = Test_NN_data[feature_cols]
x = np.array(X_data)
y = np.array(Train_NN_data['price'])
x_test = np.array(X_test)


def scheduler(epoch):
    # 到规定的epoch，学习率减小为原来的1/10

    if epoch == 1400:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        # print("lr changed to {}".format(lr * 0.1))
    if epoch == 1700:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        # print("lr changed to {}".format(lr * 0.1))
    if epoch == 1900:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        # print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


reduce_lr = LearningRateScheduler(scheduler)

kfolder = KFold(n_splits=2, shuffle=True, random_state=2018)
oof_nn = np.zeros(len(x))
predictions_nn = np.zeros(len(x_test))
predictions_train_nn = np.zeros(len(x))
kfold = kfolder.split(x, y)
fold_ = 0
print(kfolder.n_splits)
c = 0
for train_index, vali_index in kfold:
    c = c + 1
    print(c)
    k_x_train = x[train_index]
    k_y_train = y[train_index]
    k_x_vali = x[vali_index]
    k_y_vali = y[vali_index]

    model = tf.keras.Sequential()
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
    model.add(layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.02)))

    model.compile(
        loss='mean_absolute_error',
        #   # optimizer=tensorflow.python.keras.optimizers.Adam(),
        # optimizer=keras.optimizers.Adam(),
        optimizer='adam',

        metrics=['mae']
    )

    model.fit(k_x_train, k_y_train, batch_size=512, epochs=2000, validation_data=(k_x_vali, k_y_vali),
              callbacks=[reduce_lr])  # callbacks=callbacks,
    oof_nn[vali_index] = model.predict(k_x_vali).reshape((model.predict(k_x_vali).shape[0],))
    predictions_nn += model.predict(x_test).reshape((model.predict(x_test).shape[0],)) / kfolder.n_splits
    predictions_train_nn += model.predict(x).reshape((model.predict(x).shape[0],)) / kfolder.n_splits

    print("NN score: {:<8.8f}".format(mean_absolute_error(oof_nn, y)))

# 测试集输出
predictions = predictions_nn
predictions[predictions < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Test_NN_data.SaleID
sub['price'] = predictions
sub.to_csv(output_path + 'nn_test.csv', index=False)

# 验证集输出
oof_nn[oof_nn < 0] = 0
sub = pd.DataFrame()
sub['SaleID'] = Train_NN_data.SaleID
sub['price'] = oof_nn
sub.to_csv(output_path + 'nn_train.csv', index=False)

print('nn完成训练')
end = time.perf_counter()
print(start - end)

# 模型融合与最终结果输出
# 导入树模型lgb预测数据
predictions_lgb = np.array(pd.read_csv(tree_data_path + 'lgb_test.csv')['price'])
oof_lgb = np.array(pd.read_csv(tree_data_path + 'lgb_train.csv')['price'])


# #读取price，对验证集进行评估
Train_data = pd.read_csv(tree_data_path + 'train_tree.csv', sep=' ')
TestA_data = pd.read_csv(tree_data_path + 'text_tree.csv', sep=' ')

# 导入神经网络模型预测训练集数据，进行三层融合
predictions_nn = np.array(pd.read_csv(tree_data_path + 'nn_test.csv')['price'])
oof_nn = np.array(pd.read_csv(tree_data_path + 'nn_train.csv')['price'])
nn_point = mean_absolute_error(oof_nn, np.expm1(Y_data))
print("神经网络: {:<8.8f}".format(nn_point))

# 之前决策树模型改变了预测值的长尾分布状况，现在还原回去
predictions_lgb = np.expm1(predictions_lgb)

predictions = (predictions_lgb + predictions_nn) / 2

# 测试集输出
sub = pd.DataFrame()
sub['SaleID'] = Test_data.SaleID
predictions[predictions < 0] = 0
sub['price'] = predictions
sub.to_csv(output_path + 'predictions.csv', index=False)

# 结果对照
oof_lgb = np.expm1(oof_lgb)
oof = (oof_nn + oof_lgb) / 2

all_point = mean_absolute_error(oof, np.expm1(Y_data))  # mean_absolute_error平均绝对误差
print("总输出：三层融合: {:<8.8f}".format(all_point))
