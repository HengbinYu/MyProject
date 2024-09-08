from selenium import webdriver
from selenium.webdriver.common.by import By  # 按照什么方式查找元素
from selenium.webdriver.chrome.service import Service
import time
import requests as req
import threading
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from lxml import etree as et
city='Shanghai_New3'
driver_path = Service('D:\\系统资源下载\\Compressed\\chromedriver-win64\\chromedriver.exe')
FileName = 'HotelPage_'+city+'.csv'
chrome = webdriver.Chrome(service=driver_path)
time.sleep(1)
url ='https://hotel.elong.com/hotel/hotellist?city=0201&inDate=2022-07-25&outDate=2022-07-26&filterList=8888_1&pageSize=20&t=1658324999695'
chrome.get(url)
time.sleep(2)
chrome.find_element(By.XPATH, '//*[@id="login_btn"]').click()
time.sleep(1)
chrome.find_element(By.XPATH, '//*[@id="UserName"]').click()
time.sleep(0.5)
chrome.find_element(By.XPATH, '//*[@id="UserName"]').send_keys('18032602791')
time.sleep(0.5)
chrome.find_element(By.XPATH, '//*[@id="password_tip"]').click()
time.sleep(0.5)
chrome.find_element(By.XPATH, '//*[@id="PassWord"]').send_keys('15929955150')
chrome.find_element(By.XPATH, '//*[@id="ElongLogin"]/a').click()
time.sleep(10)
chrome.find_element(By.XPATH, '//*[@id="hotel-page"]/div/div[5]/div[1]/div[1]/ul/li[5]').click()
Total_Num = chrome.find_element(By.XPATH, '//*[@id="hotel-page"]/div/div[4]/div/em').text
Total_Pages = int(int(Total_Num) / 20)
with open(FileName, "a+", newline=None, encoding='utf-8') as f:
    f.write("HotelName,HotelPrice,HotelLink\n")

def WriteHotelInfo(FileName, chrome):
    with open(FileName, "a+", newline=None, encoding='utf-8') as f:
        soup = BeautifulSoup(chrome.page_source, "lxml")
        Info1 = soup.findAll('div', {'class': 'hotelMsg'})
        Info2 = soup.findAll('div', {'class': 'hotelInfo clearfix'})
        # Info_url=list()
        # Info_name=list()
        # Info_price=list()
        Info = list()
        for i in range(len(Info1)):
            Temp1 = Info1[i].find('a').get('href')
            Temp2 = Info1[i].find('span', {'class': 'name'}).get('title')
            Temp3 = Info2[i].find('p', {'class': 'newPrice'}).text
            # Info_url.append(Temp1)
            # Info_name.append(Temp2)
            # Info_price.append(Temp3)
            Temp = "%s,%s,%s\n" % (Temp2, Temp3, Temp1)
            f.write(Temp)
            Info.append(Temp)


for PageNum in range(Total_Pages):
    try:
        WriteHotelInfo(FileName, chrome)
    except:
        time.sleep(1)
    time.sleep(3)
    NextPage='//*[@id="hotel-page"]/div[1]/div[5]/div[1]/div[3]/a[2]'
    if PageNum==0:
        NextPage='//*[@id="hotel-page"]/div[1]/div[5]/div[1]/div[3]/a'
    chrome.find_element(By.XPATH,NextPage ).click()
    relax=PageNum%5
    cur_url=chrome.current_url
    print('当前页面的链接为：',cur_url)
    print('当前爬取第%d页/共%d页' % (PageNum+1,Total_Pages))
    if relax==0 and PageNum!=0:
        print('当前已爬取%d条房源信息'% ((PageNum+1)*20))
        time.sleep(5)

input()
