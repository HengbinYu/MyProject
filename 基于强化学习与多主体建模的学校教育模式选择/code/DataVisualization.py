import numpy as np
import pandas as pd
import random
import math
import time
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']

def DrawSchool(df,batch,school_num):

    colors = {'OverallExam': 'orange', 'EliteExam': 'red', 'EliteQuality': 'blue', 'OverallQuality': 'green'}
    df['Color'] = df['Decision_EduMode'].map(colors)
    # 分组绘制折线图
    previous_color = None
    for i, (index, row) in enumerate(df.iterrows()):
        current_color = row['Color']
        if i > 0:
            plt.plot([i-1, i], [df['社会评价'].iloc[i-1], row['社会评价']], marker='o', color=previous_color, label=row['Decision_EduMode'])
        previous_color = current_color
    # 创建自定义图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label) for label, color in colors.items()]
    legend = plt.legend(handles=legend_elements, title='教育模式')
    FigName=batch+str(school_num)
    # 设置图例的外观
    frame = legend.get_frame()
    frame.set_facecolor('lightgray')
    frame.set_edgecolor('gray')
    frame.set_linewidth(1.5)
    FileName='学校社会评价随学年变化情况_'+FigName
    # 显示图形
    plt.title(FileName)
    plt.xlabel('学年')
    plt.ylabel('社会评价')
    FileName2='H:\\PythonProject\\MachineLearning\\Research\\SchoolEdu\\Figures\\'+FileName+'.png'
    plt.savefig(FileName2)
    plt.show()


num_school = [4, 6, 4, 6]
num_AllSchool = sum(num_school)
num_EachSchoStud = 500
num_stdu = sum(num_school) * num_EachSchoStud
AllBatch = ['First', 'Second', 'Third', 'Forth']
num_school_dict = dict(zip(AllBatch, num_school))
EducationModes = ['EliteQuality', 'EliteExam', 'OverallQuality', 'OverallExam']

SchoolResult = pd.read_excel('SchoolInformation.xlsx', sheet_name=None)

SchoolInformation={}

for index_batch in num_school_dict:
    batch_num=num_school_dict[index_batch]
    SchoolInformation[index_batch]=[None]*batch_num
    for index_school in range(batch_num):
        SheetName=index_batch+str(index_school)
        SchoolInformation[index_batch][index_school]=SchoolResult[SheetName]


# 画教育模式变化图
PlotData_EduMode=pd.DataFrame()
for index_batch in num_school_dict:
    batch_num=num_school_dict[index_batch]
    for index_school in range(batch_num):
        df=SchoolInformation[index_batch][index_school]
        Temp_Series=df['Decision_EduMode']
        Temp_Series.loc['SchoolBatch']=index_batch
        Temp_Series.loc['SchoolNum']=index_school
        PlotData_EduMode=pd.concat([PlotData_EduMode,Temp_Series],axis=1)
PlotData_EduMode=PlotData_EduMode.T
columns_order = PlotData_EduMode.columns.tolist()
new_columns_order = columns_order[-2:] + columns_order[:-2]
PlotData_EduMode = PlotData_EduMode[new_columns_order]
PlotData_EduMode=PlotData_EduMode.reset_index(drop=True)
PlotData1=PlotData_EduMode.drop(['SchoolBatch','SchoolNum'],axis=1)
Count_Frequency=pd.DataFrame()
# 统计每一列的值的频率
for column in PlotData1.columns:
    value_counts = PlotData1[column].value_counts()
    value_counts=value_counts.rename(column)
    # Count_Frequency=Count_Frequency.append(value_counts)
    Count_Frequency = pd.concat([Count_Frequency,value_counts],axis=1)
Count_Frequency=Count_Frequency.T
Count_Frequency = Count_Frequency / Count_Frequency.sum()
Count_Frequency=Count_Frequency.T
Count_Frequency=Count_Frequency.fillna(0)
Count_Frequency = Count_Frequency.rename(index={
    'OverallExam': '全面应试',
    'EliteExam': '精英应试',
    'EliteQuality': '精英素质',
    'OverallQuality': '全面素质'
})
# 字典定义颜色
colors = {'全面应试': 'orange', '精英应试': 'red', '精英素质': 'blue', '全面素质': 'green'}
Count_Frequency=Count_Frequency.T
# 使用 plot 方法绘制折线图
for column in Count_Frequency.columns:
    plt.plot(Count_Frequency.index, Count_Frequency[column], marker='o', label=column, color=colors[column])

# 显示图形
plt.title('教育模式比例变化情况')
plt.xlabel('学年')
plt.ylabel('不同教育模式所占比例')
plt.legend(title='教育模式')
plt.savefig('H:\\PythonProject\\MachineLearning\\Research\\SchoolEdu\\Figures\\'+"教育模式比例变化情况.png")
plt.show()
# 画每个学校的模式图
School_SocialEvaluation={}
for index_batch in num_school_dict:
    batch_num=num_school_dict[index_batch]
    School_SocialEvaluation[index_batch]=[None]*batch_num
    for index_school in range(batch_num):
        df=SchoolInformation[index_batch][index_school]
        df1=df['社会评价']
        df2=df['Decision_EduMode']
        df3=pd.concat([df1,df2],axis=1)
        School_SocialEvaluation[index_batch][index_school]=df3
for index_batch in num_school_dict:
    batch_num=num_school_dict[index_batch]
    for index_school in range(batch_num):
        df=School_SocialEvaluation[index_batch][index_school]
        # DrawSchool(df,index_batch,str(index_school))
print('------------------')


def DrawSchool(df_list, batch, school_num):
    # 根据 'Category' 列设置点的颜色
    colors = {'OverallExam': 'orange', 'EliteExam': 'red', 'EliteQuality': 'blue', 'OverallQuality': 'green'}

    # 创建自定义图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label) for label, color in colors.items()]

    # 设置图例的外观
    legend = plt.legend(handles=legend_elements, title='教育模式')
    frame = legend.get_frame()
    frame.set_facecolor('lightgray')
    frame.set_edgecolor('gray')
    frame.set_linewidth(1.5)

    previous_color = None

    for df in df_list:
        # 分组绘制折线图
        for i, (index, row) in enumerate(df.iterrows()):
            current_color = row['Color'] if 'Color' in df.columns else colors[row['Decision_EduMode']]
            if i > 0:
                plt.plot([i-1, i], [df['社会评价'].iloc[i-1], row['社会评价']], marker='o', color=previous_color, label=row['Decision_EduMode'])
            previous_color = current_color

    # 设置图形标题和标签
    FigName = batch + str(school_num)
    plt.title('学校社会评价随学年变化情况_' + FigName)
    plt.xlabel('学年')
    plt.ylabel('社会评价')

    # 保存图形
    FileName = 'H:\\PythonProject\\MachineLearning\\Research\\SchoolEdu\\Figures\\学校社会评价随学年变化情况_' + FigName + '.png'
    plt.savefig(FileName)
    plt.show()

def DrawSchool1(df_list, batch, school_num):
    # 根据 'Category' 列设置点的颜色
    colors = {'OverallExam': 'orange', 'EliteExam': 'red', 'EliteQuality': 'blue', 'OverallQuality': 'green'}

    # 创建自定义图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label) for label, color in colors.items()]

    # 设置图例的外观
    legend = plt.legend(handles=legend_elements, title='教育模式')
    frame = legend.get_frame()
    frame.set_facecolor('lightgray')
    frame.set_edgecolor('gray')
    frame.set_linewidth(1.5)

    for df in df_list:
        # 分组绘制散点图
        for i, (index, row) in enumerate(df.iterrows()):
            current_color = row['Color'] if 'Color' in df.columns else colors[row['Decision_EduMode']]
            plt.scatter(i, row['社会评价'], marker='o', color=current_color, label=row['Decision_EduMode'])

    # 设置图形标题和标签
    FigName = batch + str(school_num)
    plt.title('学校社会评价随学年变化情况_' + FigName)
    plt.xlabel('学年')
    plt.ylabel('社会评价')

    # 保存图形
    FileName = 'H:\\PythonProject\\MachineLearning\\Research\\SchoolEdu\\Figures\\学校社会评价随学年变化情况_' + FigName + '.png'
    plt.savefig(FileName)
    plt.show()



for index_batch in num_school_dict:
    batch_num=num_school_dict[index_batch]
    df_list=[]
    for index_school in range(batch_num):
        df=School_SocialEvaluation[index_batch][index_school]
        df_list.append(df)
    DrawSchool1(df_list, index_batch, 'Batch')
