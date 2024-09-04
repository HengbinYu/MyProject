import pandas as pd
import os
FileDirectory=os.getcwd()
FileDirectory=FileDirectory+r"\InitialData"
File1=r"\InitialData_Compilation.xlsx"
File2=r"\InitialData_YearBook.xlsx"
WriteName=r"\InitialData_MergedData.xlsx"
ReadAddress1=FileDirectory+File1
ReadAddress2=FileDirectory+File2
WriteAddress=FileDirectory+WriteName
fm1=pd.read_excel(ReadAddress1,sheet_name=None,index_col=0)
fm2=pd.read_excel(ReadAddress2,sheet_name=None,index_col=0)
writer = pd.ExcelWriter(WriteAddress)
Years=fm1.keys()
SkipYear=['2001','2002','2003']
temp=fm1['2021']
AllProvince=temp.iloc[1:,:].index.values
print(AllProvince)
SkipProvince=['广西']
Head1=temp.iloc[0:1,:]
del temp
temp=fm2['2021']
Head2=temp.iloc[0:1,:]
del temp

with pd.ExcelWriter(WriteAddress) as writer:
    for province in AllProvince:
        if province not in SkipProvince:
            print(province)
            # 分为两部分分别从两个文件里取出数据,然后再合并到一个dataframe里,然后再写入一个sheet
            # 先从第一个文件里提取出一个dataframe
            Compilation0=pd.DataFrame()
            for year in Years:
                if year not in SkipYear:
                    df1=fm1[year]
                    temp1=df1.loc[province,:]
                    temp1.rename(int(year), inplace=True)
                    temp1=temp1.to_frame().T
                    Compilation0=pd.concat([Compilation0,temp1],axis=0)
            Compilation1=Compilation0.sort_index()
            Compilation=pd.concat([Head1,Compilation1],axis=0)
            # print(Compilation)
            del Compilation1,Compilation0,df1,temp1
            # 开始统计年鉴
            YearBook0=pd.DataFrame()
            for year in Years:
                if year not in SkipYear:
                    df2=fm2[year]
                    temp2=df2.loc[province,:]
                    temp2.rename(int(year), inplace=True)
                    temp2=temp2.to_frame().T
                    YearBook0=pd.concat([YearBook0,temp2],axis=0)
            YearBook1=YearBook0.sort_index()
            YearBook=pd.concat([Head2,YearBook1],axis=0)
            # print(YearBook)
            del YearBook1,YearBook0,df2,temp2
            MergedData=pd.concat([Compilation,YearBook],axis=1)
            # print(MergedData.shape)
            # print(MergedData)
            MergedData.to_excel(writer,sheet_name=province)
# writer.save()

