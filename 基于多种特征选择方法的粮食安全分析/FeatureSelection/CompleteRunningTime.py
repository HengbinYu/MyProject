import pandas as pd
import os
import time

FileDirectory=os.getcwd()
ReadAddress1=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\RunningTime_Lasso.xlsx'
ReadAddress2=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\RunningTime_RF.xlsx'
# ReadAddress3=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\RunningTime_FQR.xlsx'
ReadAddress3=r'H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\RunningTime_FRAR.xlsx'
WriteAddress=FileDirectory + r'\RunningTime.xlsx'
fm1=pd.read_excel(ReadAddress1,sheet_name=0,index_col=0)
fm1.rename(columns={'特征选择耗时': 'Lasso'},inplace=True)
fm2=pd.read_excel(ReadAddress2,sheet_name=0,index_col=0)
fm2.rename(columns={'特征选择耗时': 'RF'},inplace=True)
fm3=pd.read_excel(ReadAddress3,sheet_name=0,index_col=0)
fm3.rename(columns={'特征选择耗时': 'FRAR'},inplace=True)

df=pd.concat([fm1,fm2,fm3],axis=1)
avg=df.mean()
df.loc['平均耗时'] = avg
# print(df)
# # df.applymap(lambda x: '{:.2e}'.format(x))
print(df)
writer = pd.ExcelWriter(WriteAddress)
with pd.ExcelWriter(WriteAddress) as writer:
    df.to_excel(writer)
# writer.save()
