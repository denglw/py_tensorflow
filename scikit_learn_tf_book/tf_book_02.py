#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2020/4/4 14:23
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_02.py
# @desc: 一个完整的机器学习项目
# tensorflow 1.14.0

'''可执行
完整地学习一个案例项目。下面是主要步骤：
1、项目概述。
2、获取数据,创建测试集。
3、数据探索和可视化,发现规律。
4、为机器学习算法准备数据。
5、选择模型,进行训练和评估。
6、微调模型。
7、给出解决方案。
8、部署、监控、维护系统。
'''
# 1、项目概述 （略过）--可以查看另外一个文件 tf_book_02_demo

# 构建保存图片的方法
# Common imports
import tarfile
import urllib

import numpy as np
import os

# to make this notebook's output stable across runs
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)
# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.isdir(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# 2、获取数据,创建测试集。
# 2.1下载数据解压
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets\housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
# 获取数据 运行一次
# fetch_housing_data()

# 2.2 查看数据结构
def load_housing_data(housing_path=HOUSING_PATH):
    current_dir = os.path.dirname(os.getcwd()) #当前工程路径
    data_dir = os.path.join(current_dir,housing_path)
    csv_path = os.path.join(data_dir, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print('head():''\n',housing.head()) # 前5行
print('info():')
print(housing.info()) #总行数，属性类型，非空值数量
# 发现字段ocean_proximity的属性类型是object
print('object类型：''\n',housing['ocean_proximity'].value_counts())
print('describe():''\n',housing.describe() )# 数值属性概况（count max min std mean）
# 绘制每个属性的直方图
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
# plt.show()

# 2.3创建测试集（随机挑选20%）
# sklearn分层采样
housing["median_income"].hist() #中位收入分布图
save_fig("median_income_1")
# plt.show()
# 分层处理median_income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5,5.0,inplace=True) #注意此语法
housing["income_cat"].hist() #中位收入分布图
save_fig("median_income_cat")
# plt.show()
print('income_cat分层处理：''\n',housing["income_cat"].value_counts())
#  Scikit-Learn的StratifiedShuffleSplit类
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 对比采样效果
#分层采样的测试集的收入分类比例
print('分层采样占比：''\n',strat_test_set["income_cat"].value_counts()/len(strat_test_set))
#总数据集的收入分类比例
print('总样本占比''\n',housing["income_cat"].value_counts() / len(housing))
# 结论：分层采样测试集的收入分类比例与总数据集几乎相同

# 2.4删除income_cat属性，使数据回到初始状态
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# 3、数据探索、可视化、发现规律。
housing = strat_train_set.copy() # 创建一个副本
# 3.1地理数据可视化（经纬度） --所有街区的散点图
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
# alpha设为 0.1,数据点的密度
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")

'''看房价
每个圈的半径表示街区的人口（选项s），颜色代表价格（选项c）。
我们用预先定义的名为jet的颜色图（选项cmap），它的范围是从蓝色（低价）到红色（高价）
'''
# 加州房价图1
# 图说明房价和位置（比如，靠海）和人口密度联系密切
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
                  s=housing["population"]/100,label="population",
                  c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()
save_fig("housing_prices_scatterplot")
'''房价和位置（比如，靠海）和人口密度联系密切-->聚类算法'''
# 加州房价图2
import matplotlib.image as mpimg
california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
# plt.show()

# 3.2查找关联 标准相关系数（standard correlation coefficient，也称作皮尔逊相关系数）
corr_matrix = housing.corr() #pandas
corr = corr_matrix["median_house_value"].sort_values(ascending=False) #房价中位数关联度
print('房价中位数corr:''\n',corr)
'''# 1强正相关，-1强负相关'''
# Pandas的scatter_matrix函数    -->标准相关系数
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# 房价中位数与收入中位数
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.axis([0,16,0,550000])
save_fig("income_vs_house_value_scatterplot")

# 3.3属性组合
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] #平均每户房间数
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] #平均每房间卧室数
housing["population_per_household"]=housing["population"]/housing["households"] #平均每户人口数
# 标准相关系数corr()
corr_matrix = housing.corr()
corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print('房价中位数corr组合属性:''\n',corr)
'''某些组合属性更有信息
bedrooms_per_room卧室数/总房间数的比例越低，房价就越高
rooms_per_household每户的房间数,房屋越大，房价就越高
'''
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
save_fig("rooms_per_household_vs_median_house_value") #房间数与房价
# plt.show()

housing.plot(kind="scatter", x="bedrooms_per_room", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
save_fig("bedrooms_per_room_vs_median_house_value") #卧室数与房价
plt.show()
print('组合属性，describe():''\n',housing.describe())

# 4、为机器学习算法准备数据
# 标签不需要转换
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
# 4.1空值处理
# 特征缺失--不完整数据
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head() # 查看为null的数据 head前5记录
print('不完整数据：''\n',sample_incomplete_rows)
# 空值处理--删除空值 具体某一个属性空值
# sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
# 空值处理--删除属性
# sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
# 空值处理--进行赋值（0，平均值，中位数）
median = housing["total_bedrooms"].median()
# sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
# 数值（Imputer）
imputer = SimpleImputer(strategy="median")
# 只有数值属性才能算出中位数
housing_num = housing.drop('ocean_proximity', axis=1)
# 拟合数据
imputer.fit(housing_num)
# imputer计算出了每个属性的中位数，并将结果保存在了实例变量statistics_
print('statistics_''\n',imputer.statistics_)
print('median().values''\n',housing_num.median().values)
# 超参数都可以通过实例的public变量直接访问
print('imputer超参数strategy''\n',imputer.strategy)
# Transform the training set
X = imputer.transform(housing_num)

# Numpy数组转成Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing.index)
housing_tr.loc[sample_incomplete_rows.index.values]

# 4.2文本和类别属性（独热编码OneHotEncoder）
housing_cat = housing[['ocean_proximity']]
print('housing_cat head():''\n',housing_cat.head(10))
cat_encoder = OneHotEncoder()
#cat_encoder = OneHotEncoder(sparse=False) # False不返回稀疏矩阵
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # 返回稀疏矩阵
housing_cat_1hot_array = housing_cat_1hot.toarray() #转成数组
print('housing_cat_1hot独热编码:''\n',housing_cat_1hot_array)
cat_encoder.categories_ #查看映射表
housing.columns #查看列名


# 4.3自定义转换器 --基于FunctionTransformer函数创建属性组合转换器
# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
                     bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                 kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = attr_adder.fit_transform(housing.values)
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
print('housing_extra_attribs head():''\n',housing_extra_attribs.head())

# 4.4特征缩放
# 让所有的属性有相同的量度：线性函数归一化（Min-Max scaling）和标准化（standardization）
# MinMaxScaler  StandardScaler
# 缩放器只能向训练集拟合

# 4.5 转换流水线 Pipeline
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
print('housing_prepared:''\n',housing_prepared)

# 5、选择模型,进行训练和评估。
# 5.1 线性回归模型 LinearRegression
# 训练模型
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared)) #预测值
print("Labels:", list(some_labels)) #标签值
print("some_data_prepared:",some_data_prepared) #数据预处理结果
# 模型评估
# 均方误差（mean-square error, MSE）
# 均方根误差（Root Mean Square Error，RMSE）
# 平均绝对误差 Mean Absolute Error(MAE)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions) #标签值，预测值
lin_rmse = np.sqrt(lin_mse)
print("lin_rmse:",lin_rmse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("lin_mae:",lin_mae)

# 方法---显示评估分数
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# 使用交叉验证做更佳的评估 # K 折交叉验证（K-fold cross-validation）
# 交叉验证--线性回归模型
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
# 警告：Scikit-Learn 交叉验证功能期望的是效用函数（越大越好）而不是损失函数（越低越好），
# 因此得分函数实际上与 MSE 相反（即负值），这就是为什么前面的代码在计算平方根之前先计算-lin_scores。


# 5.2决策树 DecisionTreeRegressor
# 模型训练
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
# 模型评估
# RMSE
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("tree_rmse:",tree_rmse)
# 交叉验证--决策树模型
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


# 5.3 随机森林 RandomForestRegressor
# 模型训练
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
# 模型评估
# RMSE
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("forest_rmse:",forest_rmse)
# 交叉验证--随机森林模型
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# 5.4 支持向量机回归 Support Vector Regression(SVR)
# 模型训练
svm_reg = SVR(kernel="linear") # linear，poly，rbf默认值，sigmoid，recomputer
svm_reg.fit(housing_prepared, housing_labels)
# 模型评估 RMSE
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
print("svm_rmse:",svm_rmse)
'''支持向量机（support vector machines, SVM）是一种二分类模型'''
'''支持向量机回归 Support Vector Regression(SVR)是SVM的一个分支'''

# 6、模型微调
# 6.1 网格搜索 GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
# 超参数的最佳组合
print("grid_search.best_params_ :" ,grid_search.best_params_)
# 提示：因为 30 是n_estimators的最大值，你也应该估计更高的值，因为评估的分数可能会随n_estimators的增大而持续提升
# 最佳的估计器
print("grid_search.best_estimator_ :" ,grid_search.best_estimator_)
# 评估得分
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_) # pandas查看


# 6.2 随机搜索 RandomizedSearchCV
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
# 分析最佳模型和它们的误差
feature_importances = grid_search.best_estimator_.feature_importances_
print("feature_importances :",feature_importances)
# 将重要性分数和属性名放到一起：
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# 7、解决方案
# 7.1用测试集评估系统
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final_rmse :" ,final_rmse)


# 7.2模型保存
# A full pipeline with both preparation and prediction
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])
full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

# Model persistence using joblib
my_model = full_pipeline_with_predictor
joblib.dump(my_model, "my_model.pkl") # DIFF
# 模型加载
my_model_loaded = joblib.load("my_model.pkl") # DIFF



