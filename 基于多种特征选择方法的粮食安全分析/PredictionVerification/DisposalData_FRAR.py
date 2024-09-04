import pandas as pd
import numpy as np
import os

FileDirectory=os.getcwd()
ReadAddress=r"H:\PythonProject\MachineLearning\FoodSecurity\PredictionVerification\ForecastEvaluation_RF_Total.xlsx"
WriteAddress=FileDirectory + r'\DisposaledData.xlsx'
WriteAddress2=FileDirectory + r'\DataRank.xlsx'
WriteAddress3=FileDirectory + r'\CompareRF.xlsx'
fm=pd.read_excel(ReadAddress,sheet_name='Sheet1')
writer = pd.ExcelWriter(WriteAddress)
writer2 = pd.ExcelWriter(WriteAddress2)
writer3 = pd.ExcelWriter(WriteAddress3)
EvaluationIndex=set(fm['Result'].tolist())
with pd.ExcelWriter(WriteAddress) as writer:
    with pd.ExcelWriter(WriteAddress2) as writer2:
        with pd.ExcelWriter(WriteAddress3) as writer3:
            for index in EvaluationIndex:
                temp=fm[fm['Result'] == index]
                # temp.('Result',axis=1,inplace=True)
                del temp['Result']
                temp2=temp.loc[:,['Province','Original','Lasso','RF','FRAR']]
                temp3=temp2.set_index('Province')
                # print(temp3)
                temp3_rank=temp3.rank(axis=1)
                # print(temp3_rank)
                avg_rank=temp3_rank.mean()
                temp3_rank.loc['平均排名'] = avg_rank
                compareRF=temp3.apply(lambda x: x / x.iloc[0], axis=1)
                avg_compareRF=compareRF.mean()
                compareRF.loc['平均'] = avg_compareRF
                # print(compareRF)
                avg=temp3.mean()
                temp3.loc['平均值'] = avg
                # temp2.loc['平均值','Province']='平均值'
                temp3.to_excel(writer,sheet_name=index)
                temp3_rank.to_excel(writer2,sheet_name=index)
                compareRF.to_excel(writer3,sheet_name=index)

# writer.save()
# writer2.save()
# writer3.save()
