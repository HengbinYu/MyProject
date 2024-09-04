import pandas as pd
import os

FileDirectory=os.getcwd()
FileDirectory=FileDirectory+r"\InitialData"
File=r"\InitialData_MergedData.xlsx"
ReadAddress=FileDirectory+File
WriteName=r"\InitialData.xlsx"
CornName=r"\Data_Corn.xlsx"
WriteAddress=FileDirectory+WriteName
CornAddress=FileDirectory+CornName
fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
fm.pop('全国')
writer = pd.ExcelWriter(WriteAddress)
writer2=pd.ExcelWriter(CornAddress)
AllProvince=fm.keys()
with pd.ExcelWriter(WriteAddress) as writer:
    with pd.ExcelWriter(CornAddress) as writer2:
        for province in AllProvince:
            temp=fm[province]
            temp_filled = temp.fillna(0)
            temp_filled.drop(labels=['大豆播种面积','大豆单位面积产量'],axis=1,inplace=True)
            corn_yield=temp_filled['每亩主产品产量']
            corn=temp_filled.drop(labels=['玉米单位面积产量','每亩主产品产量'],axis=1)
            corn=pd.concat([corn,corn_yield],axis=1)
            corn_data=corn.iloc[1:,:]
            temp_filled.to_excel(writer,sheet_name=province)
            corn_data.to_excel(writer2,sheet_name=province)
# writer.save()
# writer2.save()
