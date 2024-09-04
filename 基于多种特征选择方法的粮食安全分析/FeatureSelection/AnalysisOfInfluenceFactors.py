import pandas as pd
import numpy as np
import os

FileDirectory=os.getcwd()
ReadAddress1=r"H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedFeatures_Lasso.xlsx"
ReadAddress2=r"H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedFeatures_RF.xlsx"
ReadAddress3=r"H:\PythonProject\MachineLearning\FoodSecurity\FeatureSelection\ReducedFeatures_FRAR.xlsx"
WriteAddress=FileDirectory + r'\TotalFeatures.xlsx'
fm_lasso=pd.read_excel(ReadAddress1,sheet_name='Sheet1',index_col=0)
fm_rf=pd.read_excel(ReadAddress2,sheet_name='Sheet1',index_col=0)
fm_fqr=pd.read_excel(ReadAddress3,sheet_name='Sheet1',index_col=0)
writer = pd.ExcelWriter(WriteAddress)

dict_feature={'Lasso':fm_lasso, 'RF':fm_rf,'FRAR':fm_fqr}
# print(dict_feature)
with pd.ExcelWriter(WriteAddress) as writer:
    for method in dict_feature:
        result_dict = {}
        df=dict_feature[method]
        for i, col in enumerate(df.columns):
            col_count = df[col].value_counts().to_dict()
            for k, v in col_count.items():
                if k not in result_dict:
                    result_dict[k] = 0
                result_dict[k] += v * (df.shape[1]-i)
        features_count=pd.Series(result_dict)
        features_count = features_count.sort_values(ascending=False)
        features_count.rename(method,inplace=True)
        features_count.to_excel(writer,sheet_name=method)

# writer.save()
