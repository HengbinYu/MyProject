import pandas as pd
import xlrd
from openpyxl import load_workbook
import os

def ImportCompilation():
    # FileDirectory=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ImportingData\InitialData"
    FileDirectory=os.getcwd()
    FileDirectory=FileDirectory+r"\InitialData"
    ReadName=r"\InitialData_Compilation0.xlsx"
    WriteName=r"\InitialData_Compilation.xlsx"
    ReadAddress=FileDirectory+ReadName
    WriteAddress=FileDirectory+WriteName
    fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
    writer = pd.ExcelWriter(WriteAddress)

    # print(fm)
    Years=fm.keys()
    print(Years)
    with pd.ExcelWriter(WriteAddress) as writer:
        for year in Years:
            print(year,type(year))
            example=fm[year]
            example1=example
            example1=example1.replace('\t','', regex=True).replace('\n','', regex=True)
            example2=example1.iloc[:,1:]
            print(example2.shape)
            example2= example2.apply(pd.to_numeric)
            example1.iloc[:,1:]=example2
            # print(example1)
            example1=example1.T
            # print(example1)
            example1.to_excel(writer,sheet_name=year)

        # writer.save()

def ImportYearBook():
    # 读取中国农村统计年鉴数据
    # FileDirectory=r"H:\PythonProject\MachineLearning\FoodSecurity\Data\ImportingData\InitialData"
    FileDirectory=os.getcwd()
    FileDirectory=FileDirectory+r"\InitialData"
    ReadName=r"\InitialData_YearBook0.xlsx"
    WriteName=r"\InitialData_YearBook.xlsx"
    ReadAddress=FileDirectory+ReadName
    WriteAddress=FileDirectory+WriteName
    fm=pd.read_excel(ReadAddress,sheet_name=None,index_col=0)
    writer = pd.ExcelWriter(WriteAddress)

    # print(fm)
    Years=fm.keys()
    print(Years)
    with pd.ExcelWriter(WriteAddress) as writer:
        for year in Years:
            print(year,type(year))
            example=fm[year]
            example1=example
            # print('删除前\n',example1)
            example1=example1.replace('\t','', regex=True).replace('\n','', regex=True)
            # print('删除后\n',example1)
            example2=example1.iloc[1:,:]
            print(example2.shape)
            example2= example2.apply(pd.to_numeric)
            example1.iloc[1:,:]=example2
            # print(example1)
            example1.to_excel(writer,sheet_name=year)

        # writer.save()

ImportCompilation()
ImportYearBook()
