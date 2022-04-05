import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def predict_price( LandSlope, OverallQual, GarageCars, OverallCond, Fireplaces ):
    houses_data = pd.read_csv("train.csv")
    houses_data.drop(['Id','Alley', 'PoolQC', 'Fence', 'MiscFeature','FireplaceQu'], axis=1, inplace=True)
    columns_with_null =  houses_data.isnull().sum()[lambda x : x>0].index.values.tolist()

    for col in columns_with_null :
      houses_data[col].fillna(houses_data[col].mode()[0], inplace=True)

    Non_Numerical_Columns = houses_data.select_dtypes(include=["object"]).columns.tolist()
    
    for col in Non_Numerical_Columns:
      lb=LabelEncoder()
      houses_data[col]=lb.fit_transform(houses_data[col])

    Y = houses_data[['SalePrice']].values
    X = houses_data.loc[:, houses_data.columns != 'SalePrice'].values
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state= 20)
    X_train = houses_data.loc[:,['LandSlope', 'OverallQual', 'GarageCars', 'OverallCond', 'Fireplaces']].values
    Y_train =  houses_data.loc[:,['SalePrice']].values
    lasso_model=Lasso(alpha=10)
    lasso_model.fit(X_train,Y_train)

    if ( LandSlope  == "Gtl"):
      LandSlope = 0
    elif ( LandSlope  == "Mod"):
      LandSlope = 1
    elif ( LandSlope  == "Sev"):
      LandSlope = 2
    else:
       LandSlope = 3

    print(LandSlope)

    my_test = np.array([ LandSlope, OverallQual, GarageCars, OverallCond, Fireplaces] )
    my_test = my_test.reshape(1,5)

    Y_predicted = lasso_model.predict(my_test)

    return Y_predicted
