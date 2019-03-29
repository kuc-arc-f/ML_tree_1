# encoding: utf-8


# 以下のモジュールを使うので、あらかじめ読み込んでおいてください
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
from pandas import Series, DataFrame

# 可視化モジュール
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#%matplotlib inline

# 小数第３まで表示
#%precision 3
# きのこデータの取得
import requests, zipfile
from io import StringIO
import io
import pandas as pd

mush_data = pd.read_csv("mush_data.csv")

mush_data.columns =["classes","cap_shape","cap_surface","cap_color","odor","bruises",
                    "gill_attachment","gill_spacing","gill_size","gill_color","stalk_shape",
                   "stalk_root","stalk_surface_above_ring","stalk_surface_below_ring",
                    "stalk_color_above_ring","stalk_color_below_ring","veil_type","veil_color",
                    "ring_number","ring_type","spore_print_color","population","habitat"]
#
#print(mush_data.info() )

# 参考（カテゴリー変数をダミー特徴量として変換する方法）
mush_data_dummy = pd.get_dummies(mush_data[['gill_color','gill_attachment','odor','cap_color']])
#df.groupby(['city', 'food']).mean()
#df_1 = mush_data.groupby(["classes"]).size()
#print("len=" ,len(df_1) )
#print( d)
#quit()
#
# 目的変数：flg立てをする
mush_data_dummy["flg"] = mush_data["classes"].map(lambda x: 1 if x =='p' else 0)

a4 =mush_data_dummy.groupby(["cap_color_c", "flg"])["flg"].count().unstack()

print(a4)
a5 =mush_data_dummy.groupby(["gill_color_b", "flg"])["flg"].count().unstack()
print(a5 )

#エントロピー
a6 = - (0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
print(a6 )
	
a7= - (0.001 * np.log2(0.001) + 0.999 * np.log2(0.999))
print(a7 )
#
def calc_entropy(p):
    return - (p * np.log2(p) + (1 - p) *  np.log2(1 - p) )
    
#
# pの範囲を0~1とするとエラーが出るため、少しずらしている
# pの値を0.001から0.999まで0.01刻みで動かす
p = np.arange(0.001, 0.999, 0.01)

# グラフ化
plt.plot(p, calc_entropy(p)) 
plt.xlabel("prob")
plt.ylabel("entropy")
plt.grid(True)
plt.title('entoropi')
#plt.show()
#quit()

#
a8 =mush_data_dummy.groupby("flg")["flg"].count()
print(a8 )
print("mush_data_dummy:" , len(mush_data_dummy))

a9= - (0.518 * np.log2(0.518) + 0.482 * np.log2(0.482))
print(a9 )
#
a10 =mush_data_dummy.groupby(["cap_color_c", "flg"])["flg"].count().unstack()
#print(a10 )

#
# データの分類
from sklearn.model_selection import train_test_split
# 決定木
from sklearn import tree
from sklearn.tree import  DecisionTreeClassifier

# 説明変数と目的変数
X = mush_data_dummy.drop("flg", axis=1)

Y = mush_data_dummy['flg']
#print(Y.head())
#quit()


# 学習データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=50)

# 決定木インスタンス（エントロピー、深さ5）
tree_model = DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=50)

tree_model.fit(X_train,y_train)

print("train:",tree_model.__class__.__name__ ,tree_model.score(X_train,y_train))
print("test:",tree_model.__class__.__name__ , tree_model.score(X_test,y_test))



