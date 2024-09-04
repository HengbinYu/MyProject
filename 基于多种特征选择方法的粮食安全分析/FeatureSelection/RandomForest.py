from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time
from timeit import default_timer as timer
FileDirectory=os.getcwd()
ReadAddress=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData_Normalized.xlsx"
ReadAddress2=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData.xlsx"
WriteAddress=FileDirectory + r'\ReducedFeatures_RF.xlsx'
WriteAddress2=FileDirectory + r'\ReducedData_RF.xlsx'
WriteAddress3=FileDirectory + r'\RunningTime_RF.xlsx'
fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
fm2=pd.read_excel(ReadAddress2,sheet_name=None,index_col=0)
writer = pd.ExcelWriter(WriteAddress,engine='openpyxl')
writer2 = pd.ExcelWriter(WriteAddress2,engine='openpyxl')
writer3 = pd.ExcelWriter(WriteAddress3,engine='openpyxl')
AllProvince=fm.keys()
ReducedFeatures=pd.DataFrame()
ConsumingTime=pd.Series()
with pd.ExcelWriter(WriteAddress2,engine='openpyxl') as writer2:
    for province in AllProvince:
        df=fm[province]
        print(province)
        X=df.iloc[1:,1:-1]
        y=df['每亩主产品产量'][1:]
        # 训练随机森林模型
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        start_time = timer()
        rf.fit(X, y)
        # 特征选择
        importances = rf.feature_importances_
        end_time = timer()
        run_time=end_time-start_time
        ConsumingTime[province]=run_time
        # 获取前10个影响最大的特征
        num_features = 10
        top_features_idx = importances.argsort()[-num_features:][::-1]
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
    ConsumingTime.to_excel(writer3,sheet_name='RandomForest')
# writer3.save()


