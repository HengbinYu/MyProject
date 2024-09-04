import numpy as np
import pandas as pd
import random
import math
import time
from numba import njit, prange
import os
import matplotlib.pyplot as plt
import multiprocessing

# 设置OpenMP的线程数
os.environ["OMP_NUM_THREADS"] = "4"


def StudentEnrollment(SocialEvaluation):
    InitStduScoreDistrib_Mean = {'Knowledge': 60, 'Ability': 60}
    InitStduScoreDistrib_Var = {'Knowledge': 10, 'Ability': 10}
    PreScore_Knowledge = np.random.normal(loc=InitStduScoreDistrib_Mean['Knowledge'],
                                          scale=InitStduScoreDistrib_Var[
                                              'Knowledge'], size=num_stdu)
    PreScore_Ability = np.random.normal(loc=InitStduScoreDistrib_Mean['Ability'],
                                        scale=InitStduScoreDistrib_Var[
                                            'Ability'], size=num_stdu)
    PreScore_Knowledge = np.clip(PreScore_Knowledge, 0, 100)
    PreScore_Ability = np.clip(PreScore_Ability, 0, 100)
    # 创建DataFrame
    StudentInformation = pd.DataFrame(
        {'PreScore_Knowledge': PreScore_Knowledge, 'PreScore_Ability': PreScore_Ability})
    MiddleSchoolEntranceExamination_Rational = (PreScore_Knowledge + PreScore_Ability) / 2
    MiddleSchoolEntranceExamination_Error = np.random.normal(loc=0, scale=2, size=num_stdu)
    StudentInformation[
        'MiddleSchoolEntranceExamination'] = MiddleSchoolEntranceExamination_Rational + MiddleSchoolEntranceExamination_Error
    StudentInformation = StudentInformation.sort_values(by='MiddleSchoolEntranceExamination',
                                                        ascending=False)
    StudentInformation['ScoreRank'] = StudentInformation['MiddleSchoolEntranceExamination'].rank(
        ascending=False).astype(int)
    accumulate_percentage = [0] + [sum(num_school[:i + 1]) / num_AllSchool for i in
                                   range(len(num_school))]
    StudentInformation = StudentInformation.assign(RankBatch=np.nan)
    for batch_index in range(len(accumulate_percentage) - 1):
        left_bound = accumulate_percentage[batch_index] * num_stdu
        right_bound = (accumulate_percentage[batch_index + 1]) * num_stdu
        StudentInformation.loc[(StudentInformation['ScoreRank'] <= right_bound) & (
                    StudentInformation['ScoreRank'] > left_bound), 'RankBatch'] = AllBatch[
            batch_index]

    ChoosePreferences = {}
    # 计算选择偏好概率
    for index in SocialEvaluation:
        TempSum = np.sum(SocialEvaluation[index])
        ChoosePreferences[index] = SocialEvaluation[index] / TempSum
    # 每个学生选志愿
    StudentInformation = StudentInformation.assign(SchoolPreferences=np.nan)
    for index in ChoosePreferences:
        probabilities = np.array(ChoosePreferences[index])
        # 计算该志愿下的学生人数
        Temp_stud_num = StudentInformation.loc[StudentInformation['RankBatch'] == index].shape[0]
        random_values = np.random.choice(range(num_school_dict[index]), size=Temp_stud_num, \
                                         p=probabilities)
        StudentInformation.loc[
            (StudentInformation['RankBatch'] == index), 'SchoolPreferences'] = random_values
    StudentInformation = StudentInformation.assign(SubmissionResults=np.nan)
    SchoolSelectionResults = dict()
    for index_batch in num_school_dict:
        batch_school_num = num_school_dict[index_batch]
        StudentCounter = np.zeros(batch_school_num, dtype=int)
        each_batch_student_informations = StudentInformation.loc[
            StudentInformation['RankBatch'] == index_batch]
        each_batch_student_num = StudentInformation.loc[
            StudentInformation['RankBatch'] == index_batch].index
        for index_student in each_batch_student_num:
            TempPreferences = int(StudentInformation.loc[index_student, 'SchoolPreferences'])
            if StudentCounter[TempPreferences] < num_EachSchoStud:
                StudentInformation.loc[index_student, 'SubmissionResults'] = TempPreferences
                StudentCounter[TempPreferences] = StudentCounter[TempPreferences] + 1
            else:
                StudentInformation.loc[index_student, 'SubmissionResults'] = -1

        UnfullCounter = num_EachSchoStud * np.ones(batch_school_num, dtype=int) - StudentCounter
        RemainingUnfulfilledPreferences = np.repeat(np.arange(len(UnfullCounter)), UnfullCounter)
        np.random.shuffle(RemainingUnfulfilledPreferences)
        StudentInformation.loc[((StudentInformation['SubmissionResults'] == -1) & (
                    StudentInformation[
                        'RankBatch'] == index_batch)), 'SubmissionResults'] = RemainingUnfulfilledPreferences
        SchoolSelectionResults[index_batch] = []
        for index_school in range(batch_school_num):
            TempSchoolStudentInformation = StudentInformation.loc[(
                        (StudentInformation['RankBatch'] == index_batch) & (
                            StudentInformation['SubmissionResults'] == index_school))]
            # print(TempSchoolStudentInformation)
            TempSchoolStudentInformation = TempSchoolStudentInformation.drop(['SubmissionResults',
                                                                              'SchoolPreferences',
                                                                              'RankBatch'], axis=1)
            SchoolSelectionResults[index_batch].append(TempSchoolStudentInformation)

    return SchoolSelectionResults

def Action_EachSchool(Decision, EachSchoolInformation):
    if Decision == 'EliteQuality':
        score_Knowledge = Term_EduResources * 0.5
        score_Ability = Term_EduResources * 0.5
        elite = int(num_EachSchoStud * 0.2)
        elite_increase_Knowledge = 0.8 * score_Knowledge / elite
        elite_increase_Ability = 0.8 * score_Ability / elite
        nomal_increase_Knowledge = 0.2 * score_Knowledge / (num_EachSchoStud - elite)
        nomal_increase_Ability = 0.2 * score_Ability / (num_EachSchoStud - elite)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Knowledge=np.nan)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Ability=np.nan)
        EachSchoolInformation.sort_values(by='ScoreRank', ascending=True, inplace=True)
        EachSchoolInformation.reset_index(drop=True, inplace=True)
        # 精英投入
        EachSchoolInformation.loc[:elite - 1, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                            'PreScore_Knowledge'].iloc[
                                                                        :elite] + elite_increase_Knowledge
        EachSchoolInformation.loc[:elite - 1, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                          'PreScore_Ability'].iloc[
                                                                      :elite] + elite_increase_Ability
        # 非精英投入
        EachSchoolInformation.loc[elite:, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                        'PreScore_Knowledge'].iloc[
                                                                    elite:] + nomal_increase_Knowledge
        EachSchoolInformation.loc[elite:, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                      'PreScore_Ability'].iloc[
                                                                  elite:] + nomal_increase_Ability
    elif Decision == 'EliteExam':
        score_Knowledge = Term_EduResources * 0.9
        score_Ability = Term_EduResources * 0.1
        elite = int(num_EachSchoStud * 0.2)
        elite_increase_Knowledge = 0.8 * score_Knowledge / elite
        elite_increase_Ability = 0.8 * score_Ability / elite
        nomal_increase_Knowledge = 0.2 * score_Knowledge / (num_EachSchoStud - elite)
        nomal_increase_Ability = 0.2 * score_Ability / (num_EachSchoStud - elite)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Knowledge=np.nan)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Ability=np.nan)
        EachSchoolInformation.sort_values(by='ScoreRank', ascending=True, inplace=True)
        EachSchoolInformation.reset_index(drop=True, inplace=True)
        # 精英投入
        EachSchoolInformation.loc[:elite - 1, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                            'PreScore_Knowledge'].iloc[
                                                                        :elite] + elite_increase_Knowledge
        EachSchoolInformation.loc[:elite - 1, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                          'PreScore_Ability'].iloc[
                                                                      :elite] + elite_increase_Ability
        # 非精英投入
        EachSchoolInformation.loc[elite:, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                        'PreScore_Knowledge'].iloc[
                                                                    elite:] + nomal_increase_Knowledge
        EachSchoolInformation.loc[elite:, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                      'PreScore_Ability'].iloc[
                                                                  elite:] + nomal_increase_Ability

    elif Decision == 'OverallQuality':
        score_Knowledge = Term_EduResources * 0.5
        score_Ability = Term_EduResources * 0.5
        increase_Knowledge = score_Knowledge / num_EachSchoStud
        increase_Ability = score_Ability / num_EachSchoStud
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Knowledge=np.nan)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Ability=np.nan)
        EachSchoolInformation.sort_values(by='ScoreRank', ascending=True, inplace=True)
        EachSchoolInformation.reset_index(drop=True, inplace=True)
        # 全面投入
        EachSchoolInformation.loc[:, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                   'PreScore_Knowledge'] + increase_Knowledge
        EachSchoolInformation.loc[:, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                 'PreScore_Ability'] + increase_Ability
    elif Decision == 'OverallExam':
        score_Knowledge = Term_EduResources * 0.9
        score_Ability = Term_EduResources * 0.1
        increase_Knowledge = score_Knowledge / num_EachSchoStud
        increase_Ability = score_Ability / num_EachSchoStud
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Knowledge=np.nan)
        EachSchoolInformation = EachSchoolInformation.assign(AfterScore_Ability=np.nan)
        EachSchoolInformation.sort_values(by='ScoreRank', ascending=True, inplace=True)
        EachSchoolInformation.reset_index(drop=True, inplace=True)
        # 全面投入
        EachSchoolInformation.loc[:, 'AfterScore_Knowledge'] = EachSchoolInformation[
                                                                   'PreScore_Knowledge'] + increase_Knowledge
        EachSchoolInformation.loc[:, 'AfterScore_Ability'] = EachSchoolInformation[
                                                                 'PreScore_Ability'] + increase_Ability
    return EachSchoolInformation

def Action(SchoolSelectionResults,TermEduMode):
    for batch in SchoolSelectionResults:
        BatchInformation=SchoolSelectionResults[batch]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            Decision=TermEduMode[batch][index_school]
            TempData=Action_EachSchool(Decision, EachSchoolInformation)
            TempData['SchoolNum']=index_school
            TempData['SchoolBatch']=batch
            SchoolSelectionResults[batch][index_school]=TempData
    SchoolDatabase=SchoolSelectionResults
    return SchoolDatabase

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def TestFuction(row):
    k=0.01
    thres_max=0.2
    A=row['AfterScore_Knowledge']
    B=row['AfterScore_Ability']
    FinalScore=(A+(max(B-A,0)/(B-A))*A*thres_max*sigmoid(k*(B-A)))*np.random.normal(1, 1/B)
    return FinalScore

def CollegeTest(SchoolDatabase):
    AllStuFinalScore=pd.DataFrame()

    for batch in SchoolDatabase:
        BatchInformation=SchoolDatabase[batch]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            AllStuFinalScore=pd.concat([AllStuFinalScore,EachSchoolInformation],axis=0)

    AllStuFinalScore['FinalScore'] = AllStuFinalScore.apply(TestFuction, axis=1)
    AllStuFinalScore=AllStuFinalScore.sort_values(by='FinalScore', ascending=False)
    AllStuFinalScore['FinalRank']=AllStuFinalScore['FinalScore'].rank(ascending=False).astype(int)
    AllStuFinalScore['RankPercentage']=AllStuFinalScore['FinalRank']/num_stdu


    for batch in SchoolDatabase:
        BatchInformation=SchoolDatabase[batch]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            TempData=AllStuFinalScore[(AllStuFinalScore['SchoolBatch']==batch)&(AllStuFinalScore['SchoolNum']==index_school)]
            SchoolDatabase[batch][index_school]=TempData
    return SchoolDatabase,AllStuFinalScore

def FinalExamAnalysis(SchoolDatabase,SchoolPerformance):
    EachTerm_AllSchool_Performance=pd.DataFrame(columns=['SchoolBatch','SchoolNum']+list(SchoolPerformance[AllBatch[0]][0].columns))
    for batch in SchoolDatabase:
        BatchInformation=SchoolDatabase[batch]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            df_temp=pd.Series(index=SchoolPerformance[batch][index_school].columns,dtype='float64')
            for index_evaluate in EvaluatingIndicator_rank:
                Temp_threshold=EvaluatingIndicator_rank[index_evaluate]
                Temp_selected=EachSchoolInformation.loc[EachSchoolInformation['RankPercentage']<=Temp_threshold]
                ratio_value=Temp_selected.shape[0]/EachSchoolInformation.shape[0]
                rank_value=Temp_selected['FinalRank'].mean()
                df_temp[index_evaluate]=ratio_value
                df_temp[EvaluatingIndicator_rank2[index_evaluate]]=rank_value
            df_temp['平均分']=EachSchoolInformation['FinalScore'].mean()
            df_temp['平均排名']=EachSchoolInformation['FinalRank'].mean()
            df_temp['方差']=EachSchoolInformation['FinalScore'].std()
            df_temp['学生排名平均净增量']=(EachSchoolInformation['ScoreRank']-EachSchoolInformation['FinalRank']).mean()
            SchoolPerformance[batch][index_school]=SchoolPerformance[batch][index_school].append(df_temp,ignore_index=True)
            df_temp2=df_temp
            df_temp2['SchoolBatch']=batch
            df_temp2['SchoolNum']=index_school
            EachTerm_AllSchool_Performance=EachTerm_AllSchool_Performance.append(df_temp2,ignore_index=True)
    return SchoolPerformance,EachTerm_AllSchool_Performance

def SocialEvaluation_Function(row):
    weight=np.array([0.5,0.2,0.05,0.2,0.05])
    vector_num=row[['清北率','985率','211率','一本率','本科率']].values
    result=np.dot(weight,vector_num)
    return result

def CalculateSocialEvaluation(SchoolPerformance,EachTerm_AllSchool_Performance,Term_Num):
    ColumnsName=['清北率','985率','211率','一本率','本科率']
    df_normalized=EachTerm_AllSchool_Performance.copy()
    df_normalized[ColumnsName] = (df_normalized[ColumnsName] - df_normalized[ColumnsName].min()) / (df_normalized[ColumnsName].max() - df_normalized[ColumnsName].min())
    df_normalized['社会评价']=df_normalized.apply(SocialEvaluation_Function,axis=1)
    EachTerm_AllSchool_Performance['社会评价']=df_normalized['社会评价']

    for index, row in EachTerm_AllSchool_Performance.iterrows():
        TempSocialEvaluation=row['社会评价']
        TempBatch=row['SchoolBatch']
        TempNum=row['SchoolNum']
        SchoolPerformance[TempBatch][TempNum].loc[Term_Num,'社会评价']=TempSocialEvaluation
    SocialEvaluation={}
    for index_batch in num_school_dict:
        index_school_num=num_school_dict[index_batch]
        SocialEvaluation[index_batch]=np.empty(index_school_num)
        for index_school in range(index_school_num):
            SocialEvaluation[index_batch][index_school]=SchoolPerformance[index_batch][index_school].loc[Term_Num,'社会评价']

    return SchoolPerformance,EachTerm_AllSchool_Performance,SocialEvaluation

def GenerateSchoolPerformance():
    # 生成学校表现的空函数
    SchoolPerformance={}
    for batch_index in num_school_dict:
        TempNum=num_school_dict[batch_index]
        TempList=[None for _ in range(TempNum)]
        SchoolPerformance[batch_index]=TempList
        for index_school in range(TempNum):
            ColumnName=list(EvaluatingIndicator_rank.keys())+list(EvaluatingIndicator_rank2.values())+EvaluatingIndicator_score
            df_temp=pd.DataFrame(columns=ColumnName)
            SchoolPerformance[batch_index][index_school]=df_temp
    Performance_ColumnName=list(SchoolPerformance[AllBatch[0]][0].columns)
    return SchoolPerformance,Performance_ColumnName

def apply_exponential(x):
    tau=10
    return np.exp(tau*x)

def CalculateEachModeBenefit(Decision,EachSchoolInformation,AllStuFinalScore,Performance_ColumnName,batch_decision,index_school_decision):
    EachSchoolInformation=Action_EachSchool(Decision, EachSchoolInformation)
    EachSchoolInformation['FinalScore'] = EachSchoolInformation.apply(TestFuction, axis=1)
    EachSchoolInformation['SchoolBatch']=batch_decision
    EachSchoolInformation['SchoolNum']=index_school_decision
    OtherSchoolsStudents=AllStuFinalScore[(AllStuFinalScore['SchoolBatch']!=batch_decision)|(AllStuFinalScore['SchoolNum']!=index_school_decision)]
    TempStuFinalScore=pd.concat([OtherSchoolsStudents,EachSchoolInformation],axis=0)
    TempStuFinalScore=TempStuFinalScore.sort_values(by='FinalScore', ascending=False)
    TempStuFinalScore['FinalRank']=TempStuFinalScore['FinalScore'].rank(ascending=False).astype(int)
    TempStuFinalScore['RankPercentage']=TempStuFinalScore['FinalRank']/num_stdu

    TempSchoolDatabase={}
    for batch in num_school_dict:
        batch_num=num_school_dict[batch]
        TempSchoolDatabase[batch]=[None] * batch_num
        for index_school in range(batch_num):
            TempData=TempStuFinalScore[(TempStuFinalScore['SchoolBatch']==batch)&(TempStuFinalScore['SchoolNum']==index_school)]
            TempSchoolDatabase[batch][index_school]=TempData
    Temp_EachTerm_AllSchool_Performance=pd.DataFrame(columns=['SchoolBatch','SchoolNum']+Performance_ColumnName)
    for batch in TempSchoolDatabase:
        BatchInformation=TempSchoolDatabase[batch]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            df_temp=pd.Series(index=Performance_ColumnName,dtype='float64')
            for index_evaluate in EvaluatingIndicator_rank:
                Temp_threshold=EvaluatingIndicator_rank[index_evaluate]
                Temp_selected=EachSchoolInformation.loc[EachSchoolInformation['RankPercentage']<=Temp_threshold]
                ratio_value=Temp_selected.shape[0]/EachSchoolInformation.shape[0]
                rank_value=Temp_selected['FinalRank'].mean()
                df_temp[index_evaluate]=ratio_value
                df_temp[EvaluatingIndicator_rank2[index_evaluate]]=rank_value
            df_temp['平均分']=EachSchoolInformation['FinalScore'].mean()
            df_temp['平均排名']=EachSchoolInformation['FinalRank'].mean()
            df_temp['方差']=EachSchoolInformation['FinalScore'].std()
            df_temp['学生排名平均净增量']=(EachSchoolInformation['ScoreRank']-EachSchoolInformation['FinalRank']).mean()
            # SchoolPerformance[batch][index_school]=SchoolPerformance[batch][index_school].append(df_temp,ignore_index=True)
            df_temp2=df_temp
            df_temp2['SchoolBatch']=batch
            df_temp2['SchoolNum']=index_school
            Temp_EachTerm_AllSchool_Performance=Temp_EachTerm_AllSchool_Performance.append(df_temp2,ignore_index=True)
    ColumnsName=['清北率','985率','211率','一本率','本科率']
    df_normalized=Temp_EachTerm_AllSchool_Performance.copy()
    df_normalized[ColumnsName] = (df_normalized[ColumnsName] - df_normalized[ColumnsName].min()) / (df_normalized[ColumnsName].max() - df_normalized[ColumnsName].min())
    df_normalized['社会评价']=df_normalized.apply(SocialEvaluation_Function,axis=1)
    Temp_EachTerm_AllSchool_Performance['社会评价']=df_normalized['社会评价']
    NormalizedDenominator=max(abs(Temp_EachTerm_AllSchool_Performance['学生排名平均净增量'].max()),abs(Temp_EachTerm_AllSchool_Performance['学生排名平均净增量'].min()))
    Temp_EachTerm_AllSchool_Performance['学生排名平均净增量']=Temp_EachTerm_AllSchool_Performance['学生排名平均净增量']/NormalizedDenominator
    value_SocialEvaluation=Temp_EachTerm_AllSchool_Performance[(Temp_EachTerm_AllSchool_Performance['SchoolBatch']==batch_decision)&(Temp_EachTerm_AllSchool_Performance['SchoolNum']==index_school_decision)]['社会评价'].values[0]
    value_RankChange=Temp_EachTerm_AllSchool_Performance[(Temp_EachTerm_AllSchool_Performance['SchoolBatch']==batch_decision)&(Temp_EachTerm_AllSchool_Performance['SchoolNum']==index_school_decision)]['学生排名平均净增量'].values[0]
    return value_SocialEvaluation,value_RankChange


def CalculateBenefits(EachSchoolInformation,AllStuFinalScore,Performance_ColumnName,batch_decision,index_school_decision):
    Attraction_SocialEvaluation={}
    Attraction_RankChange={}
    Attraction_Benefit={}
    for index_EduMode in EducationModes:
        value_SocialEvaluation,value_RankChange=CalculateEachModeBenefit(index_EduMode,EachSchoolInformation,AllStuFinalScore,Performance_ColumnName,batch_decision,index_school_decision)
        Attraction_SocialEvaluation[index_EduMode]=value_SocialEvaluation
        Attraction_RankChange[index_EduMode]=value_RankChange
        Attraction_Benefit[index_EduMode]=AttractionWeight_SocialEvaluation*value_SocialEvaluation+(1-AttractionWeight_SocialEvaluation)*value_RankChange
    return Attraction_Benefit

def CalculateAttraction(LastAttraction,Benefits,Decision,EduMode,fai,epsilon,step):
    if Decision==EduMode:
        epsilon2=1-epsilon
    else:
        epsilon2=epsilon
    Attraction=fai*LastAttraction+epsilon2*Benefits*math.exp(-step*0.01)
    return Attraction

def MakeDecision(LastAttraction_Benefit,Attraction_Benefit,LastDecision,step):
    Attraction_MakeDecision={}
    # 将惯性项与收益合成为吸引力
    for index_EduMode in Attraction_Benefit:
        Benefit=Attraction_Benefit[index_EduMode]
        LastAttraction=LastAttraction_Benefit[index_EduMode]
        Attraction_MakeDecision[index_EduMode]=CalculateAttraction(LastAttraction,Benefit,LastDecision,index_EduMode,fai,epsilon,step)
    # 选取策略
    Decision_keys = list(Attraction_MakeDecision.keys())
    Temp_p = np.array(list(Attraction_MakeDecision.values()))
    exp_function = np.vectorize(apply_exponential)
    Decision_p=exp_function(Temp_p)
    Decision_p=(Decision_p/sum(Decision_p)).tolist()
    Decision_EduMode = random.choices(Decision_keys, weights=Decision_p, k=1)[0]
    return Decision_EduMode,Attraction_MakeDecision

def Generate_AllSchoolDecision():
    AllSchoolDecision={}
    for batch_decision in num_school_dict:
        # 遍历每一批次，以便对每一批次的学校进行决策选择
        AllSchoolDecision[batch_decision]=[None] * num_school_dict[batch_decision]
        for index in range( num_school_dict[batch_decision]):
            # 遍历每一批次的每个学校，得到的迭代器是一个dataframe
            AllSchoolDecision[batch_decision][index]=pd.DataFrame(columns=(['Decision_EduMode']+EducationModes))
    return AllSchoolDecision

def Decision_AllSchool(SchoolSelectionResults,AllStuFinalScore,Term_Num,AllSchoolDecision):
    # 整合函数，所有学校做决策
    TermEduMode = dict()
    for index in num_school_dict:
        TermEduMode[index]=[None] * num_school_dict[index]
    for batch_decision in num_school_dict:
        # 遍历每一批次，以便对每一批次的学校进行决策选择
        BatchInformation=SchoolSelectionResults[batch_decision]
        for index_school, EachSchoolInformation in enumerate(BatchInformation):
            # 遍历每一批次的每个学校，得到的迭代器是一个dataframe
            AllSchoolDecision[batch_decision][index_school] = AllSchoolDecision[batch_decision][index_school].append(pd.Series(dtype='float64'), ignore_index=True,)
            if Term_Num==0:
                AllSchoolDecision[batch_decision][index_school].loc[Term_Num,'Decision_EduMode']=random.choice(EducationModes)
                for index_mode in EducationModes:
                    AllSchoolDecision[batch_decision][index_school].loc[Term_Num,index_mode]=0
            else:
                Attraction_Benefit=CalculateBenefits(EachSchoolInformation,AllStuFinalScore,Performance_ColumnName,batch_decision,index_school)
                LastDecision=AllSchoolDecision[batch_decision][index_school].loc[Term_Num-1,'Decision_EduMode']
                LastAttraction_Benefit={}
                for index_EduMode in EducationModes:
                    LastAttraction_Benefit[index_EduMode]=AllSchoolDecision[batch_decision][index_school].loc[Term_Num-1,index_EduMode]
                Decision_EduMode,Attraction_MakeDecision=MakeDecision(LastAttraction_Benefit,Attraction_Benefit,LastDecision,Term_Num)
                AllSchoolDecision[batch_decision][index_school].loc[Term_Num,'Decision_EduMode']=Decision_EduMode
                for index_EduMode in EducationModes:
                    AllSchoolDecision[batch_decision][index_school].loc[Term_Num,index_EduMode]=Attraction_MakeDecision[index_EduMode]
            TermEduMode[batch_decision][index_school]=AllSchoolDecision[batch_decision][index_school].loc[Term_Num,'Decision_EduMode']
    return AllSchoolDecision,TermEduMode


TotalTerm=30
num_school = [4, 6, 4, 6]
num_AllSchool = sum(num_school)
num_EachSchoStud = 500
num_stdu = sum(num_school) * num_EachSchoStud
Term_EduResources = 5000
AllBatch = ['First', 'Second', 'Third', 'Forth']
num_school_dict = dict(zip(AllBatch, num_school))
EducationModes = ['EliteQuality', 'EliteExam', 'OverallQuality', 'OverallExam']
AttractionWeight_SocialEvaluation = 0.7
epsilon=0.2
fai=0.5
EvaluatingIndicator_rank={'清北率':0.01,'985率':0.05,'211率':0.15,'一本率':0.45,'本科率':0.75}
EvaluatingIndicator_score=['平均分','平均排名','方差','学生排名平均净增量','社会评价']
EvaluatingIndicator_rank2={'清北率':'清北平均排名','985率':'985平均排名','211率':'211平均排名','一本率':'一本平均排名','本科率':'本科平均排名'}
# 初始化社会评价
SocialEvaluation = {AllBatch[0]: np.random.uniform(90, 100, num_school[0]),
                    AllBatch[1]: np.random.uniform(80, 90, num_school[1]),
                    AllBatch[2]: np.random.uniform(70, 80, num_school[2]),
                    AllBatch[3]: np.random.uniform(60, 70, num_school[3])}
AllSchoolDecision=Generate_AllSchoolDecision()
SchoolPerformance,Performance_ColumnName=GenerateSchoolPerformance()
AllStuFinalScore=None

with pd.ExcelWriter('EachTerm_AllSchool_Performance.xlsx') as writer3:
    with pd.ExcelWriter('AllStudentsFinalScore.xlsx') as writer4:
        for Term_Num in range(TotalTerm):
            Name_Sheet='Term'+str(Term_Num)
            print('--------------------------------------------第{}期开始----------------------------------------'.format(Term_Num))
            # 学生投档
            start_time = time.time()
            SchoolSelectionResults = StudentEnrollment(SocialEvaluation)
            end_time = time.time()
            print('第{}期学校投档结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            # 学校做决策
            start_time = time.time()
            AllSchoolDecision,TermEduMode=Decision_AllSchool(SchoolSelectionResults,AllStuFinalScore,Term_Num,AllSchoolDecision)
            end_time = time.time()
            print('第{}期学校决策结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            # 学校行动
            start_time = time.time()
            SchoolDatabase=Action(SchoolSelectionResults,TermEduMode)
            end_time = time.time()
            print('第{}期学校行动结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            # 高考
            start_time = time.time()
            SchoolDatabase,AllStuFinalScore=CollegeTest(SchoolDatabase)
            end_time = time.time()
            print('第{}期学校高考结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            # 统计高考结果
            start_time = time.time()
            SchoolPerformance,EachTerm_AllSchool_Performance=FinalExamAnalysis(SchoolDatabase,SchoolPerformance)
            end_time = time.time()
            print('第{}期学校统计高考结果结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            # 计算社会评价
            start_time = time.time()
            SchoolPerformance,EachTerm_AllSchool_Performance,SocialEvaluation=CalculateSocialEvaluation(SchoolPerformance,EachTerm_AllSchool_Performance,Term_Num)
            end_time = time.time()
            print('第{}期学校计算社会评价结束，耗时{}秒'.format(Term_Num,end_time-start_time))
            EachTerm_AllSchool_Performance.to_excel(writer3, sheet_name=Name_Sheet, index=False)
            AllStuFinalScore.to_excel(writer4, sheet_name=Name_Sheet, index=False)
print('================================================代码运行完成，下面开始写入文件====================================================')
with pd.ExcelWriter('SchoolInformation.xlsx') as writer1:
    with pd.ExcelWriter('SchoolDatabase.xlsx') as writer2:
        for index_batch in num_school_dict:
            print('正在写入第{}批次'.format(index_batch))
            batch_num=num_school_dict[index_batch]
            for index_school in range(batch_num):
                df1=SchoolPerformance[index_batch][index_school]
                df2=AllSchoolDecision[index_batch][index_school]
                df=pd.concat([df1,df2],axis=1)
                df['SchoolBatch']=index_batch
                df['SchoolNum']=index_school
                Name_Sheet=index_batch+str(index_school)
                df.to_excel(writer1, sheet_name=Name_Sheet, index=False)
                df0=SchoolDatabase[index_batch][index_school]
                df0.to_excel(writer2, sheet_name=Name_Sheet, index=False)

print('================================================数据写入完成，下面开始可视化绘图==================================================')


