from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys

import pandas as pd
from datetime import date, timedelta
import os

from tqdm import tqdm

CR = '20241213' # 時間
os.makedirs(f"{CR}", exist_ok=True)

def crawler(pair = 'CNY', page_end = 137):
    # options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    driver = webdriver.Chrome()
    currency = pair
    driver.get(f"https://www.esunbank.com/zh-tw/personal/deposit/rate/forex/exchange-rate-chart#currency={currency}") # 更改網址以前往不同網頁
    driver.maximize_window()
    start_date = str(date.today() - timedelta(days = 365*3))

    time.sleep(2)
    # driver.find_element(By.XPATH, '//*[@id="historyTrend"]/div/div[1]/div[2]/div[2]/label[1]').click()
    # 輸入日期
    ele = driver.find_element(By.XPATH, '//input[@id="fromDate"]')
    ele.send_keys(Keys.CONTROL,'a')
    ele.send_keys(Keys.DELETE)
    driver.find_element(By.XPATH, '//input[@id="fromDate"]').send_keys(f"{start_date}")

    # 查詢
    
    driver.find_element(By.XPATH, '/html/body/section[2]/div/div/div/div[3]/a').click()

    # 切數據表
    time.sleep(1)
    driver.find_element(By.XPATH, '//*[@id="historyTrend"]/section/div/ul/li[2]').click()

    tables = []
    for i in tqdm(range(0,page_end,1)):
        loc = i+1
        # Load匯率
        xpath =  f'//*[@id="datasheet"]/div[1]/div[{loc}]/table/tbody'
        table = driver.find_element(By.XPATH, xpath).text
        table = table.split('\n')[2:]
        tables.extend(table)

        # next_page
        try:
            driver.find_element(By.XPATH, '//*[@id="datasheet"]/nav/ul/li[8]').click()
        except:
            break

    tables = [tmp.split() for tmp in tables]
    output = pd.DataFrame(tables, columns = ['日期', 'Bid', 'Ask'])
    output.to_excel(f'{CR}/FX_{currency}_spot.xlsx', index = False)
    driver.close()
    return 1

if __name__ == '__main__':
    pairs = ['USD', 'CNY', 'EUR', 'AUD', 'GBP','JPY']
    page_end = 137
    for pair in pairs:
        crawler(pair, page_end)
        print(f'輸出檔案{pair}.xlsx')

