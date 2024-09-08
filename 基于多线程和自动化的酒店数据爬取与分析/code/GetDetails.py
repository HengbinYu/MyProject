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

CsvPosition='H:\\PythonProject\\MachineLearning\\DataAnalysis\\Hotel\\Data_url\\HotelPage_Shanghai.csv'
df=pd.read_csv(CsvPosition)
print(df)

HotelUrl='/hotel/hoteldetail?hotelId=64153350&inDate=2022-07-25&outDate=2022-07-26&traceToken=%7C%2a%7CcityId%3A201%7C%2a%7CqId%3A4f5b6651-c425-492c-9a4b-84b4b85a1863%7C%2a%7Cst%3Acity%7C%2a%7CsId%3A201%7C%2a%7Cscene_ids%3A0%7C%2a%7Cbkt%3Ar1%7C%2a%7Cpos%3A0%7C%2a%7ChId%3A64153350%7C%2a%7CTp%3Adefault%7C%2a%7Cpage_index%3A0%7C%2a%7Cpage_size%3A20%7C%2a%7C'
HotelUrl='https://hotel.elong.com'+HotelUrl
print(HotelUrl)
resp = req.get(HotelUrl, UserAgent().random)
print(resp)
resp.encoding = 'utf-8'  # 此页面编码方式为 GBK，设置为 utf-8 同样也会乱码
soup = BeautifulSoup(resp.text, "lxml")
print('soup:',soup)

HouseMessage=soup.find('body')
Temp=HouseMessage.find('div',{'class':'hotel-content'})
print(HouseMessage)
print(Temp)
