import requests
from bs4 import BeautifulSoup
import pandas as pd

def monthly_report(year, month):
    data_dict = {
    "供應鏈分類": {
        "軸承與齒輪": ["鈞興KY", "健樁", "羅昇", "上銀", "台灣精銳"],
        "線性滑軌": ["上銀", "直得", "全球傳動"],
        "元件": ["亞德客KY", "上銀", "氣立"],
        "控制器": ["新漢", "和樁", "盟立"],
        "可程式邏輯控制器（PLC）": ["台達電", "艾訊", "泓格"],
        "編碼器": ["大銀微", "羅昇", "直得", "台達電"],
        "工控系統": ["大銀微系統", "羅昇", "樺漢", "盟立", "研華", "新漢", "佳市達", "威強電", "瑞傳", "振樺電"],
        "電源模組": ["台達電", "時碩工業", "正威"]
    },
    "重點技術領域": {
        "機器視覺（軟體）": ["所羅門", "亞光", "聰泰", "慧友", "佳能", "宸曜", "凌華", "鈺創", "昆盈"],
        "距離感測器（機器人最貴零件）": ["恒達"],
        "滾珠螺旋": ["亞德客KY", "上銀", "全球傳動", "台達電", "直得"],
        "馬達": ["大銀微系統", "東元", "大同", "士電"],
        "減速器": ["台灣精銳", "宇隆", "鈞興", "上銀", "羅昇", "盟英", "盟立"],
        "機器人搭配工具機": ["亞德客", "氣立", "上銀", "大銀微", "直得", "全球傳動", "台灣精銳", "鈞興", "羅昇"]
    }
    }

    # Flatten all values under '供應鏈分類'
    all_supply_chain_companies = [company for companies in data_dict["供應鏈分類"].values() for company in companies]

    # Flatten all values under '重點技術領域'
    all_tech_field_companies = [company for companies in data_dict["重點技術領域"].values() for company in companies]

    all_related_companies = all_supply_chain_companies + all_tech_field_companies


    # 假如是西元，轉成民國
    if year > 1990:
        year -= 1911

    url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'_0.html'
    if year <= 98:
        url = 'https://mops.twse.com.tw/nas/t21/sii/t21sc03_'+str(year)+'_'+str(month)+'.html'

    # 偽瀏覽器
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    # 下載該年月的網站，並用pandas轉換成 dataframe
    response = requests.get(url, headers=headers)
    response.encoding = 'big5'

    # 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到表格
    tables = soup.find_all('table')[0].find_all('table')
    tables = [tables[i] for i in range(0,len(tables)) if i%2 == 0]

    DF = pd.DataFrame()
    for i in range(0,len(tables)-2):
        try:
            df = []
            table_len = len(tables[i].find_all('tr'))
            tmp = tables[i].find_all('tr')[0].text
            sector = re.search(r"產業別：(.*)單位：千元", tmp).group(1)

            cells = tables[i].find_all('tr')[3].find_all(['th'])
            columns = ([cell.get_text(strip=True).replace('\n', '') for cell in cells])
            
            for j in range(4,table_len-1):
                cells = tables[i].find_all('tr')[j].find_all(['td'])
                data = ([cell.get_text(strip=True).replace('\n', '') for cell in cells])
                if len(data) <= len(columns)+5:
                    df.append(data)
            try:
                df = pd.DataFrame(df, columns = columns)
            except:
                columns.append('備註')
                df = pd.DataFrame(df, columns = columns)
            df['產業別'] = sector

            col = df.pop('產業別')  # 移除「產業別」欄位並保存
            df.insert(0, '產業別', col)  # 將「產業別」插入到第一列
        except Exception as error:
            print(error)
        if isinstance(df, pd.DataFrame):
            DF = pd.concat([DF, df])
    DF['相關公司'] = DF['公司名稱'].apply(lambda x: 'YES' if x in all_related_companies else 'NO')
    return DF

# 民國100年1月
df = monthly_report(113,1)
df.to_excel('test.xlsx')