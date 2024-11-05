from selenium import webdriver
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import pandas as pd
import time

url = 'https://www.macromicro.me/charts/117199/global-share-of-international-payments-via-swift-by-major-currency'

driver = uc.Chrome()
driver.get(url)
time.sleep(5)
data = driver.find_elements(By.XPATH,'//*[@id="ccApp"]/div/div[2]/div[1]/div/div/div[2]/div[4]/ul')[0].text.split()
df = {
    '幣別':[],
    '最新公告時間':[],
    '占比':[],
    # '前一期':[]
}
for i in range(0,len(data), 4):
    df['幣別'].append(data[i])
    df['最新公告時間'].append(data[i+1])
    df['占比'].append(data[i+2])
    # df['前一期'].append(data[i+3])

df = pd.DataFrame(df)
df.to_excel('12_SWIFT各幣別支付占比.xlsx')