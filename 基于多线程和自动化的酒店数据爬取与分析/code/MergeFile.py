mport pandas as pd

City=['Beijing','Changsha','Chengdu','Chongqing','Guangzhou','Hangzhou','Nanjing','Qingdao','Sanya','Shanghai','Shenzhen','Tianjin','Wuhan','Xiamen','Xian','Suzhou','Kunming','Dali']

Poisition='H:\\PythonProject\\MachineLearning\\DataAnalysis\\Hotel\\HotelData\\EachCity_excel\\'
SavePath='H:\\PythonProject\\MachineLearning\\DataAnalysis\\Hotel\\HotelData\\HotelData.xlsx'
FileHead='HotelDetails_'

for index in City:
    ReadFile=Poisition+FileHead+index+'.xlsx'
    print(ReadFile)
    df=pd.read_excel(ReadFile)
    df1=df[1:]
    if index=='Beijing':
        df1=df
        df2=df
    # df2=pd.merge(df2,df1)
    df2=pd.concat([df2,df1],axis=0)
    print(df2)
    print('%s城市已写入' % (index))
df2.to_excel(SavePath)
print('所有城市已写入')
