#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# @Time    : 2020/2/29 14:34
# @Author  : Denglw
# @Email   : 892253193@qq.com
# @File    : tf_book_02_demo.py
# @desc:  一个完整的机器学习项目
# https://www.jianshu.com/p/aec40aa580e0

'''详细分析过程
完整地学习一个案例项目。下面是主要步骤：
1、项目概述。(规划问题,商业目标，选择什么样算法，评估模型性能的指标)
2、获取数据，查看数据结构，创建测试集。（分层采样）
3、数据探索和可视化，发现规律。（标准相关系数，属性组合，合理原型）
4、为机器学习算法准备数据。(空值处理，文本处理，自定义转换器，特征缩放，数据流水线)
5、选择模型，进行训练和评估。（线性回归，决策树，随机森林，支持向量机）
6、微调模型。（网格搜索，随机搜索,测试集评估）
7、给出解决方案。
8、部署、监控、维护系统。
'''
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
import tarfile
from six.moves import urllib

# 一系列的数据处理组件被称为数据流水线。
# 1、项目概述（目标，算法，模型评估）
'''
1.1规划问题
监督或非监督，还是强化学习？
分类任务、回归任务，还是其它的？(分类、回归、聚类、关联)
批量学习还是线上学习？
如果数据量很大，你可以要么在多个服务器上对批量学习做拆分（使用MapReduce技术）或是使用线上学习。
1.2性能指标
均方误差（mean-square error, MSE）
均方根误差（Root Mean Square Error，RMSE）
平均绝对误差 Mean Absolute Error(MAE)
回归问题的典型指标是均方根误差（RMSE）。均方根误差测量的是系统预测误差的标准差。
“68-95-99.7”规则 1σ，2σ，3σ
1.3核实假设
上下游系统对接，核实输入输出
'''
# 构建保存图片的方法

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
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

# 2、下载数据解压
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

# 3、快速查看数据结构
import pandas as pd
# 加载数据
def load_housing_data(housing_path=HOUSING_PATH):
    current_dir = os.path.dirname(os.getcwd()) #当前工程路径
    data_dir = os.path.join(current_dir,housing_path)
    csv_path = os.path.join(data_dir, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head() # 前5行
housing.info() #总行数，属性类型，非空值数量
# 发现字段ocean_proximity的属性类型是object
housing['ocean_proximity'].value_counts()
housing.describe() # 数值属性概况（count max min std mean）
# 绘制每个属性的直方图（直方图能够表示出每个数值变量在给定范围的数据个数）
# %matplotlib inline  # only in a Jupyter notebook
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
'''
median_income不像是用美元表示的，实际上是被缩放过了，被限定在0.5-15之间。
housing_median_age 和 median_house_value 都被覆盖了顶部（大于某个值的数据被该值覆盖）。为了处理这种情况，可以考虑：
-给这些被覆盖的数据找到合适的标签
-把这些数据从训练数据中删除
不同的属性有不一样的尺度，后面我们会谈到特征放缩
大部分特征都是重尾的：他们更偏向于中位数的右边而不是左边。我们会尝试进行特征转换将这些数据转换成更像正态分布。
'''


# 4、创建测试集（随机挑选20%）
# 4.1自定义随机采样--numpy
import numpy as np
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
'''
存在问题：如果再次运行程序，就会产生一个不同的测试集！多次运行之后，你（或你的机器学习算法）就会得到整个数据集
解决办法：
1、保存第一次运行得到的测试集，并在随后的过程加载。
2、在调用np.random.permutation()之前，设置随机数生成器的种子（比如np.random.seed(42)），
以产生总是相同的洗牌指数（shuffled indices）
3、如果数据集更新，这两个方法都会失效
一个通常的解决办法是使用每个实例的ID来判定这个实例是否应该放入测试集（假设每个实例都有唯一并且不变的ID）。
例如，你可以计算出每个实例ID的哈希值，只保留其最后一个字节，
如果该值小于等于 51（约为 256 的 20%），就将其放入测试集。
这样可以保证在多次运行中，测试集保持不变，即使更新了数据集。
新的测试集会包含新实例中的 20%，但不会有之前位于训练集的实例。
'''
# 4.2自定义随机采样--id的hash值
import hashlib
def test_set_check(indentifier,test_ratio,hash):
    return hash(np.int64(indentifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]
# 索引作为id （要求：新增尾部追加，无删除）
housing_with_id = housing.reset_index()   # adds an `index` column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# housing经纬度联合id
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")
print(len(train_set),"train + " , len(test_set), "test")

# 4.3 sklearn随机采样
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set),"train + " , len(test_set), "test")

# 纯随机采样
# 问题：采样偏差
# 解决：分层采样
# 分层采样（stratified sampling）：将人群分成均匀的子分组，称为分层，从每个分层去取合适数量的实例，以保证测试集对总人数有代表性。
'''
数据集中的每个分层都要有足够的实例位于你的数据中，这点很重要。
否则，对分层重要性的评估就会有偏差。
这意味着，你不能有过多的分层，且每个分层都要足够大。
后面的代码通过将收入中位数除以 1.5（以限制收入分类的数量），
创建了一个收入类别属性，用ceil对值舍入（以产生离散的分类），
然后将所有大于 5的分类归入到分类 5
'''

# 4.4 sklearn分层采样
# %matplotlib inline
housing["median_income"].hist() #中位收入分布图
# 分层处理median_income
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5,5.0,inplace=True)
# % matplotlib inline
housing["income_cat"].hist() # 分层处理结果图
# 语法解释 https://www.cnblogs.com/xiashiwendao/p/9112355.html
#  Scikit-Learn的StratifiedShuffleSplit类
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# 对比采样效果
#分层采样的测试集的收入分类比例
strat_test_set["income_cat"].value_counts()/len(strat_test_set)
#随机采样的测试集的收入分类比例
test_set["income_cat"].value_counts() / len(test_set)
#总数据集的收入分类比例
housing["income_cat"].value_counts() / len(housing)
# 结论：分层采样测试集的收入分类比例与总数据集几乎相同

# 分层采样和纯随机采样的样本偏差比较
from sklearn.model_selection import train_test_split
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props

# 4.5删除income_cat属性，使数据回到初始状态
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# 5、数据探索和可视化，发现规律
housing = strat_train_set.copy() # 创建一个副本
# 5.1地理数据可视化（经纬度） --所有街区的散点图
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
# alpha设为 0.1,数据点的密度
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
# 看房价
'''
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
plt.show()
'''人类的大脑非常善于发现图片中的规律，但是需要调整可视化参数使规律显现出来。'''
# 5.2查找关联 标准相关系数（standard correlation coefficient，也称作皮尔逊相关系数）
'''标准相关系数--corr()'''
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False) #房价中位数关联度
'''相关系数的范围是-1到1。当接近1时，意味强正相关;接近-1时,意味强负相关'''

'''标准相关系数--Pandas的scatter_matrix函数'''
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# 房价中位数与收入中位数
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.1)
plt.axis([0,16,0,550000])
save_fig("income_vs_house_value_scatterplot")

# 属性组合
# 长尾分布（计算log转换数据）
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] #平均每户房间数
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] #平均每房间卧室数
housing["population_per_household"]=housing["population"]/housing["households"] #平均每户人口数

# 标准相关系数corr()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
'''某些组合属性更有信息
bedrooms_per_room卧室数/总房间数的比例越低，房价就越高
rooms_per_household每户的房间数,房屋越大，房价就越高
'''
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
# describe()
housing.describe()

# 6、为机器学习算法准备数据 （数据清洗，文本分类处理 ，特征缩放--归一化or标准化，转换流水线）
# 6.1标签不需要转换label
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy
# 6.2特征缺失--空值处理(null)
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head() # 查看为null的数据 head前5记录
sample_incomplete_rows
# 空值处理--删除空值 具体某一个属性空值
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
# 空值处理--删除属性
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
# 空值处理(数值属性)--进行赋值（0，平均值，中位数）
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows
# 空值处理(数值属性)--imputer处理每个数值属性
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
imputer = SimpleImputer(strategy="median")
# 只有数值属性才能算出中位数
# Remove the text attribute because median can only be calculated on numerical attributes
housing_num = housing.drop('ocean_proximity', axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num) #拟合数据
# imputer计算出了每个属性的中位数，并将结果保存在了实例变量statistics_
imputer.statistics_ #查看拟合结果1
# Check that this is the same as manually computing the median of each attribute
housing_num.median().values #查看拟合结果2
# Transform the training set
X = imputer.transform(housing_num) #转换处理
# 超参数都可以通过实例的public变量直接访问
imputer.strategy #结果median
# Numpy数组转成Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing.index)
housing_tr.loc[sample_incomplete_rows.index.values]
housing_tr.head()

# 6.3 文本和分类属性
# Now let's preprocess the categorical input feature, ocean_proximity:
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)

# 顺序编码 OrdinalEncoder  （文本转成数字映射）
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
# 查看映射表
ordinal_encoder.categories_

# 独热编码 OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
#cat_encoder = OneHotEncoder(sparse=False) # False不返回稀疏矩阵
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) # 返回稀疏矩阵
housing_cat_1hot
housing_cat_1hot.toarray() #转成数组
#查看映射表
cat_encoder.categories_

# 6.4 自定义转换器
# Let's create a custom transformer to add extra attributes:
housing.columns #查看列名
# 自定义转换器 --属性组合转换器
from sklearn.base import BaseEstimator, TransformerMixin
# get the right column indices: safer than hard-coding indices 3, 4, 5, 6
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(housing.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix] # 获取所有行的rooms_ix数据X[:, rooms_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()


# 自定义转换器 --基于FunctionTransformer函数创建属性组合转换器
from sklearn.preprocessing import FunctionTransformer

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
housing_extra_attribs.head()

# 6.5特征缩放
'''
# 让所有的属性有相同的量度：线性函数归一化（Min-Max scaling）和标准化（standardization）
# MinMaxScaler  StandardScaler
# 缩放器只能向训练集拟合
'''

# 6.6转换流水线 Pipeline
# let's build a pipeline for preprocessing the numerical attributes
# 方案一 new
# 数值属性流水线
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr

# 完整的处理数值和类别属性的流水线
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared

#方案二 old
# 转换流水线 老版本处理方案
# For reference, here is the old solution based on a DataFrameSelector transformer (to just select a subset of the Pandas DataFrame columns), and a FeatureUnion:
from sklearn.base import BaseEstimator, TransformerMixin
# Create a class to select numerical or categorical columns
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features
# (again, we could use CombinedAttributesAdder() instead of FunctionTransformer(...) if we preferred):
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])

# 整合多个流水线--处理数值和类别属性的流水线
from sklearn.pipeline import FeatureUnion
old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])

old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared

# 对比两个方案
# The result is the same as with the ColumnTransformer:
np.allclose(housing_prepared, old_housing_prepared)


# 7、选择并训练模型
# 7.1 线性回归模型 LinearRegression
# 模型训练和评估
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data) #数据预处理（训练集pipeline）
# some_data_prepared #查看数据预处理
print("Predictions:", lin_reg.predict(some_data_prepared)) #预测值
print("Labels:", list(some_labels)) #标签值

# 模型评估
# 均方误差（mean-square error, MSE）
# 均方根误差（Root Mean Square Error，RMSE）
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# 模型评估
# 平均绝对误差 Mean Absolute Error(MAE)
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

# 显示评分函数
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# K 折交叉验证（K-fold cross-validation）
# 交叉验证--线性回归模型
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)



# 7.2决策树 DecisionTreeRegressor
# 模型训练和评估
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# 模型评估
# MAE
# RMSE
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# 使用交叉验证做更佳的评估
# K 折交叉验证（K-fold cross-validation）
# 交叉验证--决策树模型
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)
'''
# 警告：Scikit-Learn 交叉验证功能期望的是效用函数（越大越好）而不是损失函数（越低越好），
因此得分函数实际上与 MSE 相反（即负值），
这就是为什么前面的代码在计算平方根之前先计算-scores。
'''


# 7.3 随机森林 RandomForestRegressor
# 模型训练
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# 模型评估
# MAE
# RMSE
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# 交叉验证--随机森林模型
from sklearn.model_selection import cross_val_score
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# 7.4 不同核心的支持向量机 Support Vector Regression(SVR)
# 模型训练和评估
from sklearn.svm import SVR
svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


# 8、Fine-tune your model
# 模型微调

# 8.1 网格搜索 GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]
forest_reg = RandomForestRegressor(random_state=42) #随机森林
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

# The best hyperparameter combination found:
# 超参数的最佳组合
grid_search.best_params_
# 提示：因为 30 是n_estimators的最大值，你也应该估计更高的值，因为评估的分数可能会随n_estimators的增大而持续提升

# 最佳的估计器
grid_search.best_estimator_

# Let's look at the score of each hyperparameter combination tested during the grid search:
# 评估得分
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

pd.DataFrame(grid_search.cv_results_) #dataframe展示结果


# 8.2 随机搜索 RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
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
feature_importances
# 将重要性分数和属性名放到一起：
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# 8.3用测试集评估系统
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test) #transform()
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# 计算测试集的 RMSE 95%的置信区间
# We can compute a 95% confidence interval for the test RMSE:
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
mean = squared_errors.mean()
m = len(squared_errors)
np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))

# We could compute the interval manually like this:
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

# Alternatively, we could use a z-scores rather than t-scores:
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)



# 9、模型保存
# A full pipeline with both preparation and prediction
# 整合数据准备和预测的Pipeline
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])
full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

# Model persistence using joblib
my_model = full_pipeline_with_predictor
from sklearn.externals import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF


# 10 上线、监督、维护你的系统
'''
需要编写程序监督你的系统运行，当性能出问题时应及时预警。
评估系统性能时需要对系统的预测进行抽样，评估是否准确，可能需要人为的分析。
时常评估系统输入的数据质量。
定期使用新数据重新训练模型。
'''

