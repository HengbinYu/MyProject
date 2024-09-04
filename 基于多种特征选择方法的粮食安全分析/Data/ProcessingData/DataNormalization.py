import pandas as pd
import os
import numpy as np
def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize(df):
    """
    对DataFrame进行归一化
    """
    for col in df.columns:
        if df[col].sum() == 0: # 检查某一列是否全为零
            # df[col] = 0.0000000001
            df[col] = 0
        else:
            # 计算最大值和最小值
            max_val = df[col].max()
            min_val = df[col].min()
            # 归一化非零列
            if min_val != max_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0 # 如果最大值和最小值相等，则该列全部归一化为0
    return df
# 对 DataFrame 中的每一列数据应用最小-最大归一化函数
# df_normalized = df.apply(min_max_scaler)

FileDirectory=os.getcwd()
FileDirectory=FileDirectory+r"\InitialData"
File=r"\InitialData.xlsx"
ReadAddress=FileDirectory+File
WriteName=r"\InitialData_Normalized.xlsx"
WriteAddress=FileDirectory+WriteName
fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
writer = pd.ExcelWriter(WriteAddress)
AllProvince=fm.keys()
with pd.ExcelWriter(WriteAddress) as writer:
    for province in AllProvince:
        print(province)
        temp=fm[province]
        temp2=temp.iloc[1:,1:-1]
        normalize(temp2)
        # temp3=temp2.apply(min_max_scaler)
        temp_normalized=temp
        temp_normalized.iloc[1:,1:-1]=temp2
        temp_normalized.to_excel(writer,sheet_name=province)

# writer.save()
