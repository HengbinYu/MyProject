import numpy as np
import pandas as pd


num_school=[4,6,4,6]
num_AllSchool=sum(num_school)
num_EachSchoStud=500
num_stdu=sum(num_school)*num_EachSchoStud
AllBatch=['First','Second','Third','Forth']
num_school_dict=dict(zip(AllBatch,num_school))
InitStduScoreDistrib_Mean={'Knowledge':60,'Ability':60}
InitStduScoreDistrib_Var={'Knowledge':10,'Ability':10}
PreScore_Knowledge = np.random.normal(loc=InitStduScoreDistrib_Mean['Knowledge'], scale=InitStduScoreDistrib_Var[
    'Knowledge'], size=num_stdu)
PreScore_Ability = np.random.normal(loc=InitStduScoreDistrib_Mean['Ability'], scale=InitStduScoreDistrib_Var[
    'Ability'], size=num_stdu)
PreScore_Knowledge=np.clip(PreScore_Knowledge, 0, 100)
PreScore_Ability=np.clip(PreScore_Ability, 0, 100)
# 创建DataFrame
StudentInformation = pd.DataFrame({'PreScore_Knowledge': PreScore_Knowledge, 'PreScore_Ability': PreScore_Ability})
MiddleSchoolEntranceExamination_Rational=(PreScore_Knowledge+PreScore_Ability)/2
MiddleSchoolEntranceExamination_Error=np.random.normal(loc=0, scale=2, size=num_stdu)
StudentInformation['MiddleSchoolEntranceExamination']=MiddleSchoolEntranceExamination_Rational+MiddleSchoolEntranceExamination_Error
StudentInformation=StudentInformation.sort_values(by='MiddleSchoolEntranceExamination',ascending=False)
StudentInformation['ScoreRank']= StudentInformation['MiddleSchoolEntranceExamination'].rank(ascending=False).astype(int)
accumulate_percentage = [0]+[sum(num_school[:i+1])/num_AllSchool for i in range(len(num_school))]
StudentInformation = StudentInformation.assign(RankBatch=np.nan)
for batch_index in range(len(accumulate_percentage)-1):
    left_bound=accumulate_percentage[batch_index]*num_stdu
    right_bound=(accumulate_percentage[batch_index+1])*num_stdu
    StudentInformation.loc[(StudentInformation['ScoreRank']<=right_bound)&(StudentInformation['ScoreRank']>left_bound),'RankBatch']=AllBatch[batch_index]

SocialEvaluation={AllBatch[0]:np.random.uniform(90, 100,num_school[0]),AllBatch[1]:np.random.uniform(80,90,num_school[1]),AllBatch[2]:np.random.uniform(70, 80,num_school[2]),AllBatch[3]:np.random.uniform(60, 70,num_school[3])}

ChoosePreferences={}
# 计算选择偏好概率
for index in SocialEvaluation:
    TempSum=np.sum(SocialEvaluation[index])
    ChoosePreferences[index]=SocialEvaluation[index]/TempSum
# 每个学生选志愿
StudentInformation = StudentInformation.assign(SchoolPreferences=np.nan)
for index in ChoosePreferences:
    probabilities = np.array(ChoosePreferences[index])
    # 计算该志愿下的学生人数
    Temp_stud_num=StudentInformation.loc[StudentInformation['RankBatch']==index].shape[0]
    random_values = np.random.choice(range(num_school_dict[index]), size=Temp_stud_num, \
                                                                      p=probabilities)
    StudentInformation.loc[(StudentInformation['RankBatch']==index),'SchoolPreferences']=random_values
StudentInformation = StudentInformation.assign(SubmissionResults=np.nan)
SchoolSelectionResults=dict()
for index_batch in num_school_dict:
    batch_school_num=num_school_dict[index_batch]
    StudentCounter=np.zeros(batch_school_num, dtype = int)
    each_batch_student_informations=StudentInformation.loc[StudentInformation['RankBatch']==index_batch]
    each_batch_student_num=StudentInformation.loc[StudentInformation['RankBatch']==index_batch].index
    for index_student in each_batch_student_num:
        TempPreferences=int(StudentInformation.loc[index_student,'SchoolPreferences'])
        if StudentCounter[TempPreferences]<num_EachSchoStud:
            StudentInformation.loc[index_student,'SubmissionResults']=TempPreferences
            StudentCounter[TempPreferences]=StudentCounter[TempPreferences]+1
        else:
            StudentInformation.loc[index_student,'SubmissionResults']=-1

    UnfullCounter=num_EachSchoStud*np.ones(batch_school_num, dtype = int)-StudentCounter
    RemainingUnfulfilledPreferences = np.repeat(np.arange(len( UnfullCounter)),  UnfullCounter)
    np.random.shuffle(RemainingUnfulfilledPreferences)
    StudentInformation.loc[((StudentInformation['SubmissionResults']==-1)&(StudentInformation['RankBatch']==index_batch)),'SubmissionResults']=RemainingUnfulfilledPreferences
    SchoolSelectionResults[index_batch]=[]
    for index_school in range(batch_school_num):
        TempSchoolStudentInformation=StudentInformation.loc[((StudentInformation['RankBatch']==index_batch)&(StudentInformation['SubmissionResults']==index_school))]
        # print(TempSchoolStudentInformation)
        TempSchoolStudentInformation=TempSchoolStudentInformation.drop(['SubmissionResults',
                                                                       'SchoolPreferences',
                                                                        'RankBatch'], axis=1)
        SchoolSelectionResults[index_batch].append(TempSchoolStudentInformation)

print(SchoolSelectionResults)
with pd.ExcelWriter('SchoolSelectionResults.xlsx') as writer:
    for index in SchoolSelectionResults:
        TempList=SchoolSelectionResults[index]
        ListLen=len(TempList)
        temp_index=0
        NameList=range(ListLen)
        for index_list in TempList:
            SheetName=index+str(NameList[temp_index])
            temp_index=temp_index+1
            index_list.to_excel(writer, sheet_name=SheetName, index=False)