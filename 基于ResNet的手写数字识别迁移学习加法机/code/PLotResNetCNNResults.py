# 该文件为读取results目录下所有跑下来的结果并将绘制出来的图表保存到figs目录下

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import openpyxl

ResultPath='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\results\\'
WritePath=ResultPath+'AllResults.xlsx'
AllResults=os.listdir(ResultPath)
print(AllResults)

AllResults.remove('AllResults.xlsx')

df_TrainRight=pd.DataFrame()
df_ValRight=pd.DataFrame()
df_TrainLoss=pd.DataFrame()
df_ValLoss=pd.DataFrame()
df_TestRight=pd.DataFrame(columns=['CNN_NoTransfer', 'CNN_TransferFixed', 'CNN_TransferPretrained',
       'NoTransfer', 'TransferFixed', 'TransferFixed_L1', 'TransferFixed_L2',
       'TransferFixed_L3', 'TransferPretrained'],index=['Fraction=20', 'Fraction=10', 'Fraction=8',
       'Fraction=6', 'Fraction=5', 'Fraction=4', 'Fraction=3',
       'Fraction=2', 'Fraction=1'])
df_TimeCost=pd.DataFrame(columns=['CNN_NoTransfer', 'CNN_TransferFixed', 'CNN_TransferPretrained',
       'NoTransfer', 'TransferFixed', 'TransferFixed_L1', 'TransferFixed_L2',
       'TransferFixed_L3', 'TransferPretrained'],index=['Fraction=20', 'Fraction=10', 'Fraction=8',
       'Fraction=6', 'Fraction=5', 'Fraction=4', 'Fraction=3',
       'Fraction=2', 'Fraction=1'])

for index,File in enumerate(AllResults):
    FilePath=ResultPath+File
    FileName=File.strip('.xlsx').strip('Result_')
    print(FileName)
    df=pd.read_excel(FilePath,sheet_name='Fraction=1')
    Temp_TR=df['Train_right']
    Temp_VR=df['Val_right']
    Temp_TL=df['Train_loss']
    Temp_VL=df['Val_loss']
    df_all=pd.read_excel(FilePath,sheet_name=None)
    for fraction in df_all:
        df_temp=df_all[fraction]
        df_TestRight.loc[fraction,FileName]=df_temp.loc[0,'Test_right']
        df_TimeCost.loc[fraction,FileName]=df_temp.loc[0,'Time_cost']
    df_TrainRight[FileName]=Temp_TR
    df_ValRight[FileName]=Temp_VR
    df_TrainLoss[FileName]=Temp_TL
    df_ValLoss[FileName]=Temp_VL
    print('Finished')
df_TrainLoss=df_TrainLoss.drop(columns=['CNN_NoTransfer', 'CNN_TransferFixed', 'CNN_TransferPretrained'],axis=1)
df_ValLoss=df_ValLoss.drop(columns=['CNN_NoTransfer', 'CNN_TransferFixed', 'CNN_TransferPretrained'],axis=1)
df_TestRight.rename(index={'Fraction=20': '5%','Fraction=10': '10%','Fraction=8': '12.5%','Fraction=6': '16.7%','Fraction=5': '20%','Fraction=4': '25%','Fraction=3': '33.3%','Fraction=2': '50%','Fraction=1': '100%'}, inplace=True)
df_TimeCost.rename(index={'Fraction=20': '5%','Fraction=10': '10%','Fraction=8': '12.5%','Fraction=6': '16.7%','Fraction=5': '20%','Fraction=4': '25%','Fraction=3': '33.3%','Fraction=2': '50%','Fraction=1': '100%'}, inplace=True)
with pd.ExcelWriter(WritePath) as writer:
    df_TrainRight.to_excel(writer,sheet_name='TrainRight',index=False)
    df_ValRight.to_excel(writer,sheet_name='ValRight',index=False)
    df_TrainLoss.to_excel(writer,sheet_name='TrainLoss',index=False)
    df_ValLoss.to_excel(writer,sheet_name='ValLoss',index=False)
    df_TestRight.to_excel(writer,sheet_name='TestRight',index=False)
    df_TimeCost.to_excel(writer,sheet_name='TimeCost',index=False)
f=pd.read_excel(WritePath,sheet_name=None)
Keys=f.keys()
# print(Keys)
for key in Keys:
    # if key in ['TestRight','TimeCost']:
    #     continue
    df_result=f[key]
    plt.figure(figsize=(12, 10))
    for column in df_result.columns:
        plt.plot(df_result.index, df_result[column], label=column,linewidth=1)

    if key in ['TestRight','TimeCost']:
        plt.xlabel('Fraction')
        plt.title('{} over Fractions'.format(key))
    else:
        plt.xlabel('Epoch')
        plt.title('{} over Epochs'.format(key))
    plt.ylabel(key)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    FigName='D:\\work\\PythonProject\\DeepLearning\\Study\\手写数字识别\\06_Transfer Learning\\'+'AllResults{}.png'.format(key)
    plt.savefig(FigName,dpi=1200)
    plt.show()




