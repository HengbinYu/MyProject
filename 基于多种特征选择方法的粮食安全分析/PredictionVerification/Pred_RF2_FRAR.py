import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
import os
import numpy as np
from timeit import default_timer as timer
CornData=r'H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\Data_Corn.xlsx'
LassoData=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedData_Lasso.xlsx'
RFData=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedData_RF.xlsx'
FRARData=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedData_FRAR.xlsx'

FileDirectory=os.getcwd()
WriteAddress=FileDirectory + r'\ForecastEvaluation_RF.xlsx'
WriteAddress2=FileDirectory + r'\ForecastEvaluation_RF_Total.xlsx'
writer = pd.ExcelWriter(WriteAddress)
writer2 = pd.ExcelWriter(WriteAddress2)

fm=pd.read_excel(CornData,sheet_name=None,index_col=0)
fm_lasso=pd.read_excel(LassoData,sheet_name=None,index_col=0)
fm_rf=pd.read_excel(RFData,sheet_name=None,index_col=0)
fm_frar=pd.read_excel(FRARData,sheet_name=None,index_col=0)
AllProvince=fm.keys()
AllResults = pd.DataFrame()
with pd.ExcelWriter(WriteAddress) as writer:
    for province in AllProvince:
        print(province)
        df=dict()
        df['Original']=fm[province]
        df['Lasso']=fm_lasso[province]
        df['RF']=fm_rf[province]
        df['FRAR']=fm_frar[province]
        results=pd.DataFrame()
        for data in df:
            # 1. 取出X和y
            print(data)
            temp=df[data]
            X=temp.drop(labels='每亩主产品产量',axis=1)
            y=temp['每亩主产品产量']
            # 2. 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(X.drop(index=2021), y.drop(index=2021), test_size=0.2,
                                                                random_state=42)

            # 3. 训练模型
            start_time = timer()
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            end_time=timer()
            train_time = end_time - start_time
            # 4. 计算训练集精度
            y_train_pred = model.predict(X_train)
            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            print('train_rmse:',train_rmse)
            # 5. 计算测试集精度
            y_test_pred = model.predict(X_test)
            test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
            print('test_rmse:',test_rmse)

            # 6. 预测2021年玉米产量
            # X_2021 = X.loc[2021,:].values.reshape(1, -1)
            X_2021 = X.loc[2021,:].to_frame().T
            y_2021_pred = model.predict(X_2021)[0]
            y_2021_true = y.loc[2021]
            pred_error = abs(y_2021_pred - y_2021_true) / y_2021_true
            print('pred_error:',pred_error)
            # 输出结果
            print(f'Training time: {train_time:.2f} seconds')
            evaluation=[y_2021_true,y_2021_pred,train_rmse,test_rmse,pred_error,train_time]
            eva_index=['y_2021_true','y_2021_pred','train_rmse','test_rmse','pred_error','train_time']
            result=pd.Series(evaluation, index=eva_index)
            result.rename(data,inplace=True)
            results=pd.concat([results,result],axis=1)
        results.to_excel(writer,sheet_name=province)
        results3=results.reset_index(drop=False)
        Insert=[province,province,province,province,province,province]
        InsertCol=pd.Series(Insert)
        InsertCol.rename('Province',inplace=True)
        results2=pd.concat([results3,InsertCol],axis=1)
        results2.rename(columns={"index":"Result"},inplace=True)
        AllResults = pd.concat([AllResults,results2])
# writer.save()
with pd.ExcelWriter(WriteAddress2) as writer2:
    AllResults.to_excel(writer2,index=False)
# writer2.save()
