import re
from selenium import webdriver
from selenium.webdriver.common.by import By  # 按照什么方式查找元素
from selenium.webdriver.chrome.service import Service
import time
import requests as req
import threading
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from lxml import etree as et
import pandas as pd
# 爬取的城市名称
City = 'Shanghai'

# 爬取出来的每个酒店的数据的存储目录
SavePath = 'H:\\PythonProject\\MachineLearning\\DataAnalysis\\Hotel\\Details\\'
# 第一步中爬取到的搜索页数据的文件目录，就是Data_url文件夹中的csv文件
ReadPath = 'H:\\PythonProject\\MachineLearning\\DataAnalysis\\Hotel\\Data_url\\'
# 读取和写入文件的名称，这两项不用变
OutputName = 'HotelDetails_'
InputName = 'HotelPage_'

SaveFile = SavePath + OutputName + City + '.csv'
ReadFile = ReadPath + InputName + City + '.csv'

df = pd.read_csv(ReadFile)
Headers = df.columns
# print(df)
pause=0
df1=df[pause:]

def UrlToXhr(url):
    xhr1 = 'https://hotel.elong.com/tapi/gethotelDetailInfo?hotelId='
    xhr2 = '&inDate=2022-07-25&outDate=2022-07-26&traceToken='
    xhr3 = '&indate=2022-07-25&outdate=2022-07-26&hotelid='
    xhr4 = '&couponOrder='
    ID = re.findall(r"hotelId=(.*)&inDate", url)[0]
    Temp = re.findall(r"traceToken=(.*)", url)[0]
    Temp = Temp.replace('%2a', '*')
    Temp = Temp.replace('%3A', ':')
    InfoUrl = xhr1 + ID + xhr2 + Temp + xhr3 + ID + xhr4
    return InfoUrl

def LoadHotelInfo(HotelUrl):
    resp = req.get(HotelUrl, UserAgent().random)
    # time.sleep(0.0001)
    resp.encoding = 'utf-8'  # 此页面编码方式为 GBK，设置为 utf-8 同样也会乱码
    soup = BeautifulSoup(resp.text, "lxml")
    Info = soup.findAll('body')[0].text
    try:
        Name_Hotel = re.findall(r"hotelName(.*)hotelNameEn", Info)[0].strip('":"').strip('",').replace(',','，')
    except:
        Name_Hotel = None

    try:
        StarLevel_Hotel = re.findall(r"starLevelDesc(.*)goodCommentRate", Info)[0].strip('":"').strip('",')
    except:
        StarLevel_Hotel = None

    try:
        Address_Hotel = re.findall(r"hotelAddress(.*)hotelOpenDate", Info)[0].strip('":"').strip('",').replace(',','，')
    except:
        Address_Hotel = None

    try:
        OpenDate_Hotel = re.findall(r"hotelOpenDate(.*)hotelOpenYear", Info)[0].strip('":"').strip('",')
    except:
        OpenDate_Hotel = None

    try:
        GoodCommentRate_Hotel = re.findall(r"goodCommentRate(.*?)commentScore", Info)[0].strip('":"').strip('",')
    except:
        GoodCommentRate_Hotel = None

    try:
        CommentScore = re.findall(r"commentScore(.*)commentDes", Info)[0].strip('":"').strip('",')
    except:
        CommentScore = None

    try:
        TotalCommerNum_Hotel = re.findall(r"totalCommentNumber(.*?)hotelAddress", Info)[0].strip('":"').strip('",')
    except:
        TotalCommerNum_Hotel = None

    try:
        DecorateDate_Hotel = re.findall(r"decorateDate(.*?)hotelName", Info)[0].strip('":"').strip('",')
    except:
        DecorateDate_Hotel = None

    try:
        Tel_Hotel = re.findall(r"hotelTel(.*?)hotelTelInfo", Info)[0].strip('":"').strip('",')
    except:
        Tel_Hotel = None

    try:
        RoomNum_Hotel = re.findall("roomNum(.*?)themeTips", Info)[0].strip('":"').strip('",').split('roomNum')[
            -1].strip('":')
    except:
        RoomNum_Hotel = None

    try:
        Description_Hotel = re.findall(r"featureInfo(.*?)facilityList", Info)[0].strip('":"').strip('",\\').replace(',','，')
    except:
        Description_Hotel = None

    try:
        CheckInTime_Hotel = \
        re.findall("checkInTime(.*?)checkOutTime", Info)[0].strip('":"').strip('",').split('checkInTime')[-1].strip(
            '":')
    except:
        CheckInTime_Hotel = None

    try:
        CheckOutTime_Hotel = re.findall("checkOutTime(.*?)gradeFacilities", Info)[0].strip('":"').strip('",')
    except:
        CheckOutTime_Hotel = None

    try:
        NearestSubway = re.findall("nearestAreaPosition(.*?)nearby_traffic_infos", Info)[0].strip('":"').strip('",')
        NearestSubway=NearestSubway.replace(',','，')
    except:
        NearestSubway = None

    try:
        StraightDistanceofSubway_Hotel = NearestSubway.split('直线')[1].split(',')[0]
    except:
        StraightDistanceofSubway_Hotel = None

    try:
        WalkTimeToSubway = NearestSubway.split('预计')[-1]
    except:
        WalkTimeToSubway = None

    try:
        PositionScore = re.findall("positionScore(.*?)facilityScore", Info)[0].strip('":"').strip('",')
    except:
        PositionScore = None

    try:
        FacilityScore = re.findall("facilityScore(.*?)serviceScore", Info)[0].strip('":"').strip('",')
    except:
        FacilityScore = None

    try:
        ServiceScore = re.findall("serviceScore(.*?)sanitationScore", Info)[0].strip('":"').strip('",')

    except:
        ServiceScore = None
    try:
        SanitationScore = re.findall("sanitationScore(.*?)costScore", Info)[0].strip('":"').strip('",')
    except:
        SanitationScore = None

    try:
        CostScore = re.findall("costScore(.*?)defaultScore", Info)[0].strip('":"').strip('",')
    except:
        CostScore = None

    try:
        FacilitySupport = re.findall("facilityList(.*?)clockFacilityList", Info)[0].strip('":"').strip('",'
                                                                                                       '').split(
            'facilityList')[-1].split('","frontStyleName":null},{"type":')
        FacilityInfo = list()
        FacilityName = []
        for each in FacilitySupport:
            Temp = each.split('hasSupport')[-1].split('hasFree')[0].strip('":,')
            Temp2 = each.split('name')[-1].strip('":,').split('"')[0]
            FacilityInfo.append(Temp)
            FacilityName.append(Temp2)
        FacilitySupport = dict(zip(FacilityName, FacilityInfo))
    except:
        FacilitySupport = None

    InfoName=['Name_Hotel', 'StarLevel_Hotel', 'Address_Hotel', 'OpenDate_Hotel', 'GoodCommentRate_Hotel', 'CommentScore', 'TotalCommerNum_Hotel', 'DecorateDate_Hotel', 'Tel_Hotel', 'RoomNum_Hotel', 'Description_Hotel', 'CheckInTime_Hotel', 'CheckOutTime_Hotel', 'NearestSubway', 'StraightDistanceofSubway_Hotel', 'WalkTimeToSubway', 'PositionScore', 'FacilityScore', 'ServiceScore', 'SanitationScore', 'CostScore', 'FacilitySupport']
    InfoValue=[Name_Hotel, StarLevel_Hotel, Address_Hotel, OpenDate_Hotel, GoodCommentRate_Hotel, CommentScore,TotalCommerNum_Hotel, DecorateDate_Hotel, Tel_Hotel, RoomNum_Hotel, Description_Hotel, CheckInTime_Hotel,CheckOutTime_Hotel, NearestSubway, StraightDistanceofSubway_Hotel, WalkTimeToSubway, PositionScore,FacilityScore, ServiceScore, SanitationScore, CostScore, FacilitySupport]
    HotelInfo=dict(zip(InfoName,InfoValue))
    return HotelInfo

with open(SaveFile, "a+", newline=None, encoding='utf-8') as f:
    f.write("酒店名称,酒店类型,最低房价,客房数,地址,最近地铁站直线距离,最近地铁站步行至时间,最近地铁站,好评率,总评分,点评总数,设施评分,服务评分,卫生评分,位置评分,性价比评分,入住时间,退房时间,"
            "停车场,免费WIFI,餐厅,健身房,游泳池,接机服务,会议室,行李寄存,装修日期,开业日期,电话,酒店简介,酒店链接\n")
    DataNum=df.shape[0]
    index=pause
    for row in df1.itertuples():
        index=index+1
        Row_name = row[1]
        Row_price = row[2]
        Row_url = row[3]
        Url_Original = 'https://hotel.elong.com' + Row_url
        HotelUrl=UrlToXhr(Url_Original)
        HotelInfo=LoadHotelInfo(HotelUrl)
        # print(HotelInfo)

        try:
            try:
                Keys=list(HotelInfo['FacilitySupport'].keys())
                WrittemDetails='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (HotelInfo['Name_Hotel'],HotelInfo['StarLevel_Hotel'],Row_price,HotelInfo['RoomNum_Hotel'],
                                       HotelInfo['Address_Hotel'],HotelInfo['StraightDistanceofSubway_Hotel'],
                                       HotelInfo['WalkTimeToSubway'],HotelInfo['NearestSubway'],HotelInfo[
                                                                  'GoodCommentRate_Hotel'],HotelInfo['CommentScore'],
                                       HotelInfo['TotalCommerNum_Hotel'],HotelInfo['FacilityScore'],
                                       HotelInfo['ServiceScore'],HotelInfo['SanitationScore'],HotelInfo['PositionScore'],
                                       HotelInfo['CostScore'],HotelInfo['CheckInTime_Hotel'],HotelInfo['CheckOutTime_Hotel'],
                                       HotelInfo['FacilitySupport'][Keys[0]],HotelInfo['FacilitySupport'][Keys[1]],
                                       HotelInfo['FacilitySupport'][Keys[2]],HotelInfo['FacilitySupport'][Keys[3]],
                                       HotelInfo['FacilitySupport'][Keys[4]],HotelInfo['FacilitySupport'][Keys[5]],
                                       HotelInfo['FacilitySupport'][Keys[6]],HotelInfo['FacilitySupport'][Keys[7]],
                                       HotelInfo['DecorateDate_Hotel'],HotelInfo['OpenDate_Hotel'],HotelInfo['Tel_Hotel'],
                                       HotelInfo['Description_Hotel'],Url_Original)
                f.write(WrittemDetails)
            except:
                WrittemDetails='%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (HotelInfo['Name_Hotel'],HotelInfo['StarLevel_Hotel'],Row_price,HotelInfo['RoomNum_Hotel'],
                                       HotelInfo['Address_Hotel'],HotelInfo['StraightDistanceofSubway_Hotel'],
                                       HotelInfo['WalkTimeToSubway'],HotelInfo['NearestSubway'],HotelInfo[
                                                                  'GoodCommentRate_Hotel'],HotelInfo['CommentScore'],
                                       HotelInfo['TotalCommerNum_Hotel'],HotelInfo['FacilityScore'],
                                       HotelInfo['ServiceScore'],HotelInfo['SanitationScore'],HotelInfo['PositionScore'],
                                       HotelInfo['CostScore'],HotelInfo['CheckInTime_Hotel'],HotelInfo['CheckOutTime_Hotel'],
                                       None,None,None,None,None,None,None,None,
                                       HotelInfo['DecorateDate_Hotel'],HotelInfo['OpenDate_Hotel'],HotelInfo['Tel_Hotel'],
                                       HotelInfo['Description_Hotel'],Url_Original)
                print('酒店设施无法读取,设置为全空')
                f.write(WrittemDetails)
        except:
            print('第%d条信息无效,该酒店链接为%s' % (index,Url_Original))
        # print(WrittemDetails)
        print('当前写入第%d条/共%d条' % (index,DataNum))
