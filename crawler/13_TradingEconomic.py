query = 'currencies'
url = f'https://tradingeconomics.com/{query}'
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
}

res = requests.get(url, headers = headers)
soup = BeautifulSoup(res.content, 'lxml')

columns = []
data_column = soup.find_all('th',{'class':'te-sort'})
for i in range(len(data_column)):
    tmp = data_column[i].text.strip()
    columns.append(tmp)


DF = []
    
data_tables = soup.find_all('table')
for i in range(0,len(data_tables)):
    data_table = data_tables[i]

    # Extract rows and columns
    rows = []
    for row in data_table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if query == 'stocks':
            cols = [col.text.strip() for col in cols][1:][:-1]
        elif query == 'currencies' or query == 'bonds':
            cols = [col.text.strip() for col in cols][1:]
        else:
            cols = [col.text.strip() for col in cols]
        rows.append(cols)

    # Convert to a pandas DataFrame
    differnet = 0
    if query == 'bonds':
        num = 8
    elif query == 'crypto':
        if i == 0:
            num = 10
        else:
            num = 9
            differnet = 1
    else:
        num = 9

    df = pd.DataFrame(rows, columns = columns[(i*num+differnet):(i*num)+num+differnet])
    df['time'] = datetime.now().strftime("%H:%M:%S")
    DF.append(df)

# 開啟json
try:
    updatepath = 'auto/data/'
    path_update = f'{updatepath}13_TradingEconomics.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df_update = json.loads(f)
    jsonFile.close() 
    data = df_update
except:
    data = {}

if query not in data.keys():
    data[query] = {}
for i in range(0,len(DF)):
    if query == 'commodities':
        for j in range(len(DF[i].iloc[:,0].values)):
            text = DF[i].iloc[:,0].values[j]
            DF[i].iloc[:,0].values[j] = re.sub(r'\n\n', ',', text.strip())

    for tmp in DF[i].iloc[:,0].values:
        if tmp not in data[query].keys():
            data[query][tmp] = {}
            if query != 'bonds':
                data[query][tmp]['Price'] = []
            data[query][tmp]['%'] = []
            data[query][tmp]['Day'] = []
            data[query][tmp]['Weekly'] = []
            data[query][tmp]['Monthly'] = []
            data[query][tmp]['YTD'] = []
            data[query][tmp]['YoY'] = []
            data[query][tmp]['Date'] = []
            data[query][tmp]['time'] = []

            if query == 'bonds':
                data[query][tmp]['Yield'] = []
            if query == 'crypto':
                data[query][tmp]['MarketCap'] = []
        
        if DF[i][DF[i][DF[i].columns[0]] == tmp]['Date'].values[0] in data[query][tmp]['Date'] and DF[i][DF[i][DF[i].columns[0]] == tmp]['time'].values[0] in data[query][tmp]['time']:
            break
        
        for column in DF[i].columns[1:]:
            data[query][tmp][column].append(DF[i][DF[i][DF[i].columns[0]] == tmp][column].values[0])

json_data = json.dumps(data)
with open(path_update, "w") as json_file:
    json_file.write(json_data)