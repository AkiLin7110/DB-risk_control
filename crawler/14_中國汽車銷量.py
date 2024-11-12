i = 0
url = f'https://xl.16888.com/month-{i}.html'
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
}

res = requests.get(url, headers = headers)
soup = BeautifulSoup(res.content, 'lxml')
page_row = soup.find_all('div',{'class':"xl-data-pageing lbBox"})[0]
last_page = int(page_row.find_all('a')[-2].text)


data = []

for i in range(0, last_page):    
    url = f'https://xl.16888.com/month-{i}.html'

    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.content, 'lxml')

    table = soup.find("table", {"class": "xl-table-def xl-table-a"})
    rows = table.find_all("tr")

    for row in rows[1:]:  # Skip header row
        cols = row.find_all("td")
        time = cols[0].text.strip()
        sales = int(cols[1].text.strip())
        yoy = cols[2].text.strip()
        data.append([time, sales, yoy])

df = pd.DataFrame(data, columns=["时间", "销量", "同比"]).drop_duplicates().sort_values('时间', ignore_index = True)
df.to_excel('auto/new_data/14_中國汽車銷量.xlsx')