from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import os
import time
from timeit import default_timer as timer
FileDirectory=os.getcwd()
ReadAddress=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData_Normalized.xlsx"
ReadAddress2=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData.xlsx"
WriteAddress=FileDirectory + r'\ReducedFeatures_Lasso.xlsx'
WriteAddress2=FileDirectory + r'\ReducedData_Lasso.xlsx'
WriteAddress3=FileDirectory + r'\RunningTime_Lasso.xlsx'
fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
fm2=pd.read_excel(ReadAddress2,sheet_name=None,index_col=0)
writer = pd.ExcelWriter(WriteAddress)
writer2 = pd.ExcelWriter(WriteAddress2)
writer3 = pd.ExcelWriter(WriteAddress3)
AllProvince=fm.keys()
ReducedFeatures=pd.DataFrame()
ConsumingTime=pd.Series()
with pd.ExcelWriter(WriteAddress2,engine='openpyxl') as writer2:
    for province in AllProvince:
        df=fm[province]
        print(province)
        X=df.iloc[1:,1:-1]
        y=df['每亩主产品产量'][1:]
        # 选择正则化参数
        params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        # 训练Lasso模型
        lasso = Lasso(max_iter=1000000)
        grid_search = GridSearchCV(lasso, params, cv=5)
        start_time = timer()
        grid_search.fit(X, y)
        end_time = timer()
        run_time=end_time-start_time
        ConsumingTime[province]=run_time
        # 特征选择
        best_model = grid_search.best_estimator_
        coefs = best_model.coef_
        # 获取前10个影响最大的特征
        num_features = 10
        top_features_idx = coefs.argsort()[-num_features:][::-1]
        top_features=X.columns[top_features_idx]
        Temp=pd.Series(top_features)
        Temp.rename(province,inplace=True)
        ReducedFeatures=pd.concat([ReducedFeatures,Temp],axis=1)
        df2=fm2[province]
        selected_data=df2.iloc[:,top_features_idx]
        selected_data=pd.concat([selected_data,df2['每亩主产品产量']],axis=1)
        selected_data=selected_data.iloc[1:,:]
        selected_data.to_excel(writer2,sheet_name=province)
        print("Top {} features: {}".format(num_features, X.columns[top_features_idx]))
# writer2.save()
ColumnName = [f'feature {i}' for i in range(1, num_features+1)]
ReducedFeatures=ReducedFeatures.T
ReducedFeatures = ReducedFeatures.rename(columns=dict(zip(ReducedFeatures.columns, ColumnName)))
with pd.ExcelWriter(WriteAddress,engine='openpyxl') as writer:
    ReducedFeatures.to_excel(writer)
# writer.save()
print(ConsumingTime)
ConsumingTime.rename('特征选择耗时',inplace=True)
with pd.ExcelWriter(WriteAddress3,engine='openpyxl') as writer3:
    ConsumingTime.to_excel(writer3,sheet_name='Lasso')
# writer3.save()
