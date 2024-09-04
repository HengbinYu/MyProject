import pandas as pd
import numpy as np
import os
import time
from timeit import default_timer as timer
import math

def fuzzy_membership_function_1(x):
    if x < 0.23:
        fuzzy_membership = 1
    elif x > 0.37:
        fuzzy_membership = 0
    else:
        fuzzy_membership = (x-0.37)/(0.23-0.37)
    return fuzzy_membership


def fuzzy_membership_function_2(x):
    if x < 0.23 or x > 0.57:
        fuzzy_membership = 0
    elif 0.37<x<0.43:
        fuzzy_membership = 1
    elif 0.23 < x < 0.37:
        fuzzy_membership = (x-0.23)/(0.37-0.23)
    else:
        fuzzy_membership = (x-0.57)/(0.43-0.57)
    return fuzzy_membership


def fuzzy_membership_function_3(x):
    if x < 0.43:
        fuzzy_membership = 0
    elif x > 0.57:
        fuzzy_membership = 1
    else:
        fuzzy_membership = (x-0.43)/(0.57-0.43)
    return fuzzy_membership

def FRAR(data):
    num_class = 3
    num_features = data.shape[1]-1
    num_instance = data.shape[0]
    fuzzy_table1 = data.applymap(fuzzy_membership_function_1)
    fuzzy_table2 = data.applymap(fuzzy_membership_function_2)
    fuzzy_table3 = data.applymap(fuzzy_membership_function_3)
    fuzzy_table = [fuzzy_table1, fuzzy_table2, fuzzy_table3]

    # importance=dict()
    importance = list()
    for index_attribute in range(num_features):
        # 计算每个属性的重要度，即计算决策属性Q对各个条件属性的模糊依赖度
        fuzzy_lower_approximation = list()  # 存储该属性在不同决策内里的模糊下近似
        for index in range(num_class):
            # 计算该属性在不同决策类里的模糊下近似
            attr_c = fuzzy_table[index].iloc[:, index_attribute]
            attr_d = fuzzy_table[index].iloc[:, -1]
            df_attr = pd.concat([1 - attr_c, attr_d], axis=1)
            inf_sup_df = df_attr.max(axis=1).min(axis=0)
            fuzzy_lower_approximation.append(inf_sup_df)


        POS_membership = list()  # 存储每个对象对于模糊正域的隶属度
        for index_instance in range(num_instance):
            # 计算每个对象对该条件属性的模糊正域的隶属度
            class_d_membership = list()  # 存储该对象在不同决策类下的隶属度
            for index_class in range(num_class):
                # 计算该对象对每个决策类的隶属度
                ind_membership = list()
                for index in range(num_class):
                    # 计算每个对象在各个等价类下对每个决策类的隶属度
                    temp = min(fuzzy_lower_approximation[index_class],
                               fuzzy_table[index].iloc[index_instance, index_attribute])
                    ind_membership.append(temp)
                temp_class = max(ind_membership)
                class_d_membership.append(temp_class)
            instance_POS_membership = max(class_d_membership)
            POS_membership.append(instance_POS_membership)
        fuzzy_dependence = sum(POS_membership) / num_instance
        # importance[feathers[index_attribute]]=fuzzy_dependence
        importance.append(fuzzy_dependence)
    # sort_importance=dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance



FileDirectory=os.getcwd()
ReadAddress=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData_Normalized.xlsx"
ReadAddress2=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ProcessingData\InitialData\InitialData.xlsx"
WriteAddress=FileDirectory + r'\ReducedFeatures_FRAR.xlsx'
WriteAddress2=FileDirectory + r'\ReducedData_FRAR.xlsx'
WriteAddress3=FileDirectory + r'\RunningTime_FRAR.xlsx'
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
        max_val = y.max()
        min_val = y.min()
        y_norm = (y - min_val) / (max_val - min_val)
        data = pd.concat([X, y_norm], axis=1)
        start_time = timer()
        result=FRAR(data)
        end_time = timer()
        run_time=end_time-start_time
        ConsumingTime[province]=run_time
        # 获取前10个影响最大的特征
        num_features = 10
        importance=np.array(result)
        top_features_idx = importance.argsort()[-num_features:][::-1]
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
    ConsumingTime.to_excel(writer3,sheet_name='FuzzyRoughAttributeReduction')
# writer3.save()
