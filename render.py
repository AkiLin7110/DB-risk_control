from flask import Flask, request, render_template, redirect, url_for, jsonify
from auto.auto_crawl import get_data1, get_data2, get_data3, get_data4, get_data5, get_data6, get_data7_1, get_data7_2, get_data7_3, get_data7_4, get_data7_5, get_data7_6, get_data8_1, get_data8_2, get_data8_3, get_data9_1, get_data9_2, get_data_10, get_data_12, get_data13, get_data14
import json
import pandas as pd
from operator import itemgetter
import os
from Forex_dashboard.cnyesnews_crawler import savefile, parse, crawler, main
from Forex_dashboard.cnyesnews_calculator import cnyesnews_category, cnyesnews_calculator
from Forex_dashboard.cnyesnews_drawer import FX_corr_overtime
import arrow
from dash import Dash, dcc, html
import plotly.express as px

app = Flask(__name__)
file_source  = "auto/new_data"
file_destination = "auto/data"
file_previous = "auto/previous_data"

def move_file(file_destination, file_previous, file_name):
    os.replace("{file_destination}/{file_name}", "{file_previous}/{file_name}" )
    return "{file_destination}/{file_name} -> {file_previous}/{file_name} successfully!"


@app.route('/')
@app.route('/hello')
def hello():
    return 'Hello World'


@app.route('/text')
def text():
    return '<html><body><h1>Hello World</h1></body></html>'


@app.route('/home')
def home():
    fig1, fig2 = FX_corr_overtime()
    fig1_html = fig1.to_html(full_html=False)
    fig2_html = fig2.to_html(full_html=False)
    return render_template('home.html', fig1_html=fig1_html, fig2_html=fig2_html)


# @app.route('/loginurl', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         return redirect(url_for('product_analysis'))

#     return render_template('login.html')

@app.route('/product_analysis', methods=['GET', 'POST'])
def product_analysis():
    print('重新讀取頁面')
    '''進出口總額'''
    df = pd.read_excel('auto/data/1_中國大陸_進出口總值(美元).xlsx', index_col=0)
    df = df.fillna('Null')
    indexs = df['date'].tolist()
    titles = ['可互換之搪孔或拉孔工具，工具機用']
    label_im = ['進口總值(含復進口)']
    data_im_values = df['import_美元'].tolist()
    label_ex = ['出口總值(含復出口)']
    data_ex_values = df['export_美元'].tolist()

    '''各國產品搜索量'''
    path_update = 'auto/data/2_google_trends.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df2 = json.loads(f)
    jsonFile.close()
    df2_labels = list([tmp for tmp in df2.keys()])


    '''工業生產增加率'''
    df = pd.read_excel('auto/data/3_主要國家工業生產增加率.xlsx', index_col = 0)
    df = df.fillna('Null') 
    label_3 = df.columns.to_list()
    indexs_3 = df.index.to_list()
    data = {}
    for i in range(0,df.shape[1]):
        tmp = df.iloc[:,i].tolist()
        data[label_3[i]] = tmp
    
    '''政府支援研發計畫'''
    df4 = pd.read_excel('auto/data/4_政府推動計畫名單.xlsx')


    '''概況總覽'''
    jsonFile = open("auto/data/5_經濟數據_Overview.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df5 = json.loads(f)      # 再使用 loads
    # print(df5)
    jsonFile.close()
    df5_labels = list([tmp for tmp in df5.keys() if '_date' not in tmp and '_unit' not in tmp])

    '''航運數據'''
    jsonFile = open('auto/data/7_上海航運交易所_航運數據.json','r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df7 = json.loads(f)
    jsonFile.close()

    '''鋼鐵價格'''
    updatepath = 'auto/data/'
    datatype = '9_1'
    query = '原料價格'
    path_update = f'{updatepath}{datatype}{query}.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df9_1 = json.loads(f)
    jsonFile.close()

    df9_1_labels = list(df9_1.keys())

    '''鋼鐵交易量'''
    updatepath = 'auto/data/'
    datatype = '9_2'
    query = '原料交易量'
    path_update = f'{updatepath}{datatype}{query}.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df9_2 = json.loads(f)
    jsonFile.close()

    df9_2_labels = list(df9_2.keys())


    return render_template('product_analysis.html', indexs=indexs, titles=titles,
                            label_im=label_im, data_im_values=data_im_values, 
                            label_ex=label_ex, data_ex_values=data_ex_values,
                            df2 = df2, df2_labels = df2_labels,
                            indexs_3 = indexs_3, label_3 = label_3, data = data,
                            tables = [df4.to_html(classes='data', header="true")],
                            df5_labels = df5_labels, df5 = df5,
                            df7 = df7,
                            df9_1 = df9_1, df9_1_labels = df9_1_labels,
                            df9_2 = df9_2, df9_2_labels = df9_2_labels
                           )

@app.route('/operation_condition', methods=['GET', 'POST'])
def operation_condition():
    return render_template('operation_condition.html')

@app.route('/government', methods=['GET', 'POST'])
def government():
    return render_template('government.html')

@app.route('/update/get_data0_1', methods=['GET', 'POST'])
def update_data0_1():
    # update_date = (request.form.get('month'))
    # parsed_date = arrow.get(update_date, "YYYY-MM")

    # # 提取年份、月份和日期
    # beginyear = parsed_date.year
    # beginmonth = parsed_date.month
    # stopmonth = parsed_date.month

    # if update_date:
    #     for i in range(1,3):
    #         main(beginyear,beginmonth,i,stopmonth)

    cnyesnews_category()
    cnyesnews_calculator()
    message = '更新完成: 貨幣對資訊'
    return render_template('update_home.html', message = message)  

@app.route('/update/get_data1', methods=['GET', 'POST'])
def update_data1():
    filepath = 'auto/new_data/'
    GEO = '中國大陸'
    data_import = get_data1(filepath,'import')
    data_export = get_data1(filepath,'export')
    test = pd.merge(data_import, data_export, how = 'outer', on = 'date')
    def convert_roc_to_gregorian(roc_date):
        # Split year and month
        year_str, month_str = roc_date.split('年')
        year = int(year_str)
        month = int(month_str.replace('月', ''))
        
        # Convert ROC year to Gregorian year
        gregorian_year = year + 1911
        
        # Return as formatted string
        return f"{gregorian_year}-{month:02d}"

    # Apply the conversion to the dataframe
    test['gregorian_date'] = test['date'].apply(convert_roc_to_gregorian)
    test = test.sort_values('gregorian_date')
    test.to_excel(f'{filepath}1_{GEO}_進出口總值(美元).xlsx')
    message = '更新完成: 進出口資料'
    return render_template('update_product.html', message = message)


@app.route('/update/get_data2', methods=['POST'])
def update_data2():
    files = os.listdir('auto/new_data')
    google_files = [file for file in files if "google" in file.lower()]

    try:
        updatepath = 'auto/data/'
        path_update = f'{updatepath}2_google_trends.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        df_update = json.loads(f)
        jsonFile.close()
    except:
        df_update = {}

    for google_file in google_files:
        GEO = google_file.split('_')[-2]
        query = google_file.split('_')[-1].split('.')[0]
        df = pd.read_excel(f'auto/new_data/{google_file}')
        df = df.astype(str)
        if GEO not in df_update.keys():
            df_update[f'{GEO}'] = {}
        if query not in df_update[f'{GEO}'].keys():
            df_update[f'{GEO}'][f'{query}'] = []
            df_update[f'{GEO}'][f'{query}_date'] = []

        for i in range(0,len(df)):
            if df.iloc[i]['date'] in df_update[f'{GEO}'][f'{query}_date']:
                continue
            df_update[f'{GEO}'][f'{query}_date'].append(df.iloc[i]['date'])
            df_update[f'{GEO}'][f'{query}'].append(df.iloc[i][f'{query}'])

    json_data = json.dumps(df_update)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)
    message = '更新完成: 搜尋資料'

    # return redirect(url_for('product_analysis', GEO=GEO))
    return render_template('update_product.html', message = message)

@app.route('/update/get_data3', methods=['GET', 'POST'])
def update_data3():
    filepath_update = 'auto/previous_data/'
    filepath_new = 'auto/new_data'
    filepath_store = 'auto/data'

    lastet_month = '11308'

    file_path = f'{filepath_update}/3_主要國家工業生產增加率.xlsx'
    if not os.path.exists(file_path):
        DF = get_data3(lastet_month, function_type = 0)
        DF.to_excel(f'{filepath_update}/3_主要國家工業生產增加率.xlsx')
    DF = pd.read_excel(f'{filepath_update}/3_主要國家工業生產增加率.xlsx')

    df = get_data3(lastet_month, function_type = 1)
    df.to_excel(f'{filepath_new}/3_主要國家工業生產增加率.xlsx')
    df = pd.read_excel(f'{filepath_new}/3_主要國家工業生產增加率.xlsx')

    if df.iloc[-1]['日期'] not in DF['日期'].values:
        DF = pd.concat([DF,pd.DataFrame(df.iloc[-1]).T], axis = 0)
    
    DF.to_excel(f'{filepath_store}/3_主要國家工業生產增加率.xlsx', index = 0)
    
    message = '更新完成: 工業生產增加率'
    return render_template('update_product.html', message = message)

@app.route('/update/get_data4', methods=['GET', 'POST'])
def update_data4():
    df = get_data4()
    df.to_excel('auto/new_data/4_政府推動計畫名單.xlsx', index = 0)

    # 讀取新的資料
    df = pd.read_excel('auto/new_data/4_政府推動計畫名單.xlsx')

    # 更新檔案路徑
    updatepath = 'auto/data/'
    path_update = f'{updatepath}4_政府推動計畫名單.xlsx'

    # 嘗試讀取舊資料，若檔案不存在，則建立空的 DataFrame
    try:
        DF = pd.read_excel(path_update)
    except FileNotFoundError:
        DF = pd.DataFrame(columns=['公司名稱', '計畫名稱', '核定日期'])

    # 合併新資料與舊資料，並去除重複資料
    DF = pd.concat([DF, df], ignore_index=True).drop_duplicates(subset=['公司名稱', '計畫名稱', '核定日期'])

    # 依照 '核定日期' 排序
    DF = DF.sort_values(by='核定日期', ignore_index=True)

    # 儲存更新後的資料
    DF.to_excel(path_update, index=False)

    message = f'更新完成: 政府推動計畫名單'

    return render_template('update_product.html', message=message)

@app.route('/update/get_data5', methods=['GET', 'POST'])
def update_data5():
    updated = request.form.get('updated_item')  # Retrieve updated_item from the form
    # updated = 'all'

    GEOs = ['china', 'india', 'malaysia', 'turkey', 'united-states']
    filepath = 'auto/new_data/'

    # 取得當前時間
    now = arrow.now()

    formatted_time = now.format("DD/MMM/YY HH:mm:ss")

    for GEO in GEOs:

        if updated == 'all':
            df, updated_items = get_data5(GEO, formatted_time)
        else:
            df, _ = get_data5(GEO, formatted_time)
            updated_items = [updated]

        df.to_excel(f'{filepath}5_經濟數據_{GEO}.xlsx', index = 0)
        path_update = f'auto/data/5_經濟數據_{updated}.json'

        try:
            jsonFile = open(path_update,'r')
            f =  jsonFile.read()   # 要先使用 read 讀取檔案
            df_update = json.loads(f)
            jsonFile.close()
        except:
            df_update = {}

        if GEO not in df_update.keys():
            df_update[GEO] = {}

        # print(f"Processing data for {GEO}: {df_update[GEO]}")

        # 寫入到舊有的excel檔的不同sheet裡面
        path_new = f'{filepath}5_經濟數據_{GEO}.xlsx'
        for updated_item in updated_items:

            df = pd.read_excel(path_new) 
            index_names = df[df['總經種類'] == f'{updated_item}']['指標名稱'].values
            print(index_names)
            


            for tmp in index_names:
                if tmp not in df_update[GEO].keys():
                    df_update[GEO][tmp] = []
                    df_update[GEO][f'{tmp}_date'] = []
                    df_update[GEO][f'{tmp}_unit'] = []
                if len(df_update[GEO][f'{tmp}_date']) > 0:
                    if df_update[GEO][f'{tmp}_date'][-1] == df[df['指標名稱'] == tmp]['公告日期'].values[0]:
                        print(GEO, tmp)
                        continue 
                # print(f"Updating {GEO}, {tmp} data...")       
                df_update[GEO][tmp].append(float(df[df['指標名稱'] == tmp]['Last'].values[0]))
                df_update[GEO][f'{tmp}_date'].append(df[df['指標名稱'] == tmp]['公告日期'].values[0])
                df_update[GEO][f'{tmp}_unit'].append(str(df[df['指標名稱'] == tmp]['單位'].values[0]))
                # print(f"Updated {GEO}: {df_update[GEO]}")

            json_data = json.dumps(df_update)
            with open(path_update, "w") as json_file:
                json_file.write(json_data)

        message = f'更新完成: {GEOs} 概況'

    if updated == 'all':
        return render_template('update_economic.html', message = message)
    else:
        return render_template('update_product.html', message=message)

@app.route('/update/get_data7', methods=['GET', 'POST'])
def update_data7():
    filepath = 'auto/new_data/'

    df1, df2 = get_data7_1()
    df1.to_excel(f'{filepath}7_上海航運交易所_全球主幹航線綜合準班率指數.xlsx', index = 0)
    df2.to_excel(f'{filepath}7_上海航運交易所_全球主幹航線到離港與收發獲準班率指數.xlsx', index = 0)

    filepath = 'auto/new_data/'
    df1 = pd.read_excel(f'{filepath}7_上海航運交易所_全球主幹航線綜合準班率指數.xlsx')
    df2 = pd.read_excel(f'{filepath}7_上海航運交易所_全球主幹航線到離港與收發獲準班率指數.xlsx')
    df2_1 = df2.iloc[:,[0,1,3]]
    df2_2 = df2.iloc[:,[0,2,4]]

    df7_2 = get_data7_2()
    df7_2.to_excel(f'{filepath}7_上海航運交易所_港口班輪準確率.xlsx', index = 0)
    df7_2 = pd.read_excel(f'{filepath}7_上海航運交易所_港口班輪準確率.xlsx').astype(str)
    df7_2_1 = df7_2.iloc[:,[0,1,7]]
    df7_2_2 = df7_2.iloc[:,[0,2,8]]
    df7_2_3 = df7_2.iloc[:,[0,3,9]]
    df7_2_4 = df7_2.iloc[:,[0,4,10]]
    df7_2_5 = df7_2.iloc[:,[0,5,11]]

    df7_3 = get_data7_3()
    df7_3.to_excel(f'{filepath}/7_上海航運交易所_一帶一路航貿指數.xlsx', index = 0)
    df7_3 = pd.read_excel(f'{filepath}/7_上海航運交易所_一帶一路航貿指數.xlsx').astype(str)

    df7_4 = get_data7_4()
    df7_4.to_excel(f'{filepath}/7_上海航運交易所_一帶一路貿易額指數.xlsx', index = 0)
    df7_4 = pd.read_excel(f'{filepath}/7_上海航運交易所_一帶一路貿易額指數.xlsx').astype(str)

    df7_5 = get_data7_5()
    df7_5.to_excel(f'{filepath}/7_上海航運交易所_集裝箱海運量指數.xlsx')
    df7_5 = pd.read_excel('auto/new_data/7_上海航運交易所_集裝箱海運量指數.xlsx', index_col= 0).astype(str)

    df7_6 = get_data7_6()
    df7_6.to_excel(f'{filepath}/7_上海航運交易所_海上絲綢之路運價指數.xlsx')
    df7_6 = pd.read_excel('auto/new_data/7_上海航運交易所_海上絲綢之路運價指數.xlsx', index_col= 0).astype(str)



    jsonFile = open('auto/data/7_上海航運交易所_航運數據.json','r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    shipping_db = json.loads(f)      # 再使用 loads
    # shipping_db = {}
    jsonFile.close()

    df1_keys = list(df1['指數名稱'].values+'_time')+list(df1['指數名稱'].values)
    df2_keys = list(df2['航線'].values+'_time')+list(df2['航線'].values)
    df7_2_keys = list(df7_2['港口'].values+'_time')+list(df7_2['港口'].values)
    df7_3_keys = list(df7_3['指數名稱'].values+'_time')+list(df7_3['指數名稱'].values)
    df7_4_keys = list(df7_4['指數名稱'].values+'_time')+list(df7_4['指數名稱'].values)
    df7_5_keys = list(df7_5['指數名稱'].values+'_time')+list(df7_5['指數名稱'].values)
    df7_6_keys = list(df7_6['指數名稱'].values+'_time')+list(df7_6['指數名稱'].values)



    def update_db(update_keys, update_data, shipping_db, data_type):
        if data_type == '7_1':
            query = '指數名稱'
            column = '全球主幹航線綜合準班率指數'
        elif data_type == '7_1_1':
            query = '航線'
            column = '全球主幹航線到離港準班率指數'
        elif data_type == '7_1_2':
            query = '航線'
            column = '全球主幹航線收發貨準班率指數'
        elif data_type == '7_2_1':
            query = '港口'
            column = '港口班輪準確率:準班率'
        elif data_type == '7_2_2':
            query = '港口'
            column = '港口班輪準確率:掛靠數'
        elif data_type == '7_2_3':
            query = '港口'
            column = '港口班輪準確率:班期综合服务水平'
        elif data_type == '7_2_4':
            query = '港口'
            column = '港口班輪準確率:在港时间(天)'
        elif data_type == '7_2_5':
            query = '港口'
            column = '港口班輪準確率:在泊時間(天)'
        elif data_type == '7_3':
            query = '指數名稱'
            column = '一帶一路航貿指數'  
        elif data_type == '7_4':
            query = '指數名稱'
            column = '一帶一路貿易額指數'
        elif data_type == '7_5':
            query = '指數名稱'
            column = '集裝箱海運量指數'
        elif data_type == '7_6':
            query = '指數名稱'
            column = '海上絲綢之路運價指數'
            

        # 創立
        if column not in shipping_db.keys():
            shipping_db[f'{column}'] = {}


        shipping_keys = update_keys
        for tmp in shipping_keys:
            # 創立
            if tmp not in shipping_db[f'{column}'].keys():
                shipping_db[f'{column}'][tmp] = []
            
            # 防止重複增加
            tmp_new = tmp.split('_')[0]
            
            new_time = update_data[update_data[f'{query}'] == tmp_new].iloc[:,2].values[0]
            if '_time' in tmp:
                if len(shipping_db[f'{column}'][tmp_new+'_time']) > 0:
                    if shipping_db[f'{column}'][tmp_new+'_time'][-1] == new_time:
                        break

            if '_time' in tmp:
                shipping_db[f'{column}'][tmp_new+'_time'].append(update_data[update_data[f'{query}'] == tmp_new].iloc[:,2].values[0])
            else:
                shipping_db[f'{column}'][tmp_new].append(update_data[update_data[f'{query}'] == tmp_new].iloc[:,1].values[0])
        print(f'航運資料庫更新完成{query}:{column}')
        return shipping_db

    shipping_db = update_db(update_keys = df1_keys, update_data = df1, shipping_db = shipping_db, data_type = '7_1')
    shipping_db = update_db(update_keys = df2_keys, update_data = df2_1, shipping_db = shipping_db, data_type = '7_1_1')
    shipping_db = update_db(update_keys = df2_keys, update_data = df2_2, shipping_db = shipping_db, data_type = '7_1_2')
    shipping_db = update_db(update_keys = df7_2_keys, update_data = df7_2_1, shipping_db = shipping_db, data_type = '7_2_1')
    shipping_db = update_db(update_keys = df7_2_keys, update_data = df7_2_2, shipping_db = shipping_db, data_type = '7_2_2')
    shipping_db = update_db(update_keys = df7_2_keys, update_data = df7_2_3, shipping_db = shipping_db, data_type = '7_2_3')
    shipping_db = update_db(update_keys = df7_2_keys, update_data = df7_2_4, shipping_db = shipping_db, data_type = '7_2_4')
    shipping_db = update_db(update_keys = df7_2_keys, update_data = df7_2_5, shipping_db = shipping_db, data_type = '7_2_5')
    shipping_db = update_db(update_keys = df7_3_keys, update_data = df7_3, shipping_db = shipping_db, data_type = '7_3')
    shipping_db = update_db(update_keys = df7_4_keys, update_data = df7_4, shipping_db = shipping_db, data_type = '7_4')
    shipping_db = update_db(update_keys = df7_5_keys, update_data = df7_5, shipping_db = shipping_db, data_type = '7_5')
    shipping_db = update_db(update_keys = df7_6_keys, update_data = df7_6, shipping_db = shipping_db, data_type = '7_6')

    path_update = 'auto/data/7_上海航運交易所_航運數據.json'
    json_data = json.dumps(shipping_db)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)

    message = f'更新完成: 所有航運數據'
    return render_template('update_product.html', message = message)

@app.route('/update/get_data8', methods=['GET', 'POST'])
def update_data8():
    filepath = 'auto/new_data/'
    # Retrieve the query from the form submission
    # querys = request.form.get('query', 'fanuc')  # Default to 'fanuc' if no query provided
    querys = ""
    if querys == "":
        querys = ['syntec','fanuc','siemens']
    forums = ['Machinists', 'robotics']
    
    # Call functions with the dynamic query
    for query in querys:
        df = get_data8_1(query, forums)
        df.to_excel(f'{filepath}8_競品分析_{query}.xlsx', index=False)
        
        filename = f'{filepath}8_競品分析_{query}.xlsx'
        output = get_data8_2(filename)
        names = ['questions','comments']
        for i in range(0,len(output)):
            with open(f"{filepath}8_{query}_{names[i]}.txt","w", encoding = 'UTF-8') as file:
                file.write(output[i])
        message = f'更新完成: 文字雲資料 for {query}'
    
    get_data8_3()
    
    return render_template('update_product.html', message=message)


@app.route('/update/get_data9', methods=['GET', 'POST'])
def update_data9():
    filepath = 'auto/new_data/'
    storepath = 'auto/data/'

    # 9_1: 交易價格
    get_data9_1()
    # Load新data
    try:
        updatepath = 'auto/new_data/'
        path_update = f'{updatepath}9_1原料價格.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        df_update = json.loads(f)
        jsonFile.close()
    except:
        df_update = {}

    # 初始化變數
    signal_first = 0
    storepath = 'auto/data/'
    path_store = f'{storepath}9_1原料價格.json'

    # 檢查路徑是否存在，若不存在則建立路徑
    os.makedirs(storepath, exist_ok=True)

    # 讀取舊的 JSON 文件
    try:
        with open(path_store, 'r') as jsonFile:
            df_store = json.load(jsonFile)
    except FileNotFoundError:
        # 若文件不存在，則初始化為空字典
        signal_first = 1
        df_store = {}
    except json.JSONDecodeError:
        # 若 JSON 格式錯誤，初始化為空字典
        signal_first = 1
        df_store = {}

    if signal_first == 1:
        df_store = df_update.copy()
    else:
        for key1 in df_update.keys():
            for key2 in df_update[key1].keys():
                for key3 in df_update[key1][key2].keys():
                    if df_update[key1][key2][key3]['date'][0] in df_store[key1][key2][key3]['date']:
                        break
                    for key4 in df_update[key1][key2][key3].keys():
                        for i in range(0,len(df_update[key1][key2][key3][key4])):
                            df_store[key1][key2][key3][key4].append(df_update[key1][key2][key3][key4][i])

    json_data = json.dumps(df_store)
    with open(path_store, "w") as json_file:
        json_file.write(json_data) 

    get_data9_2()
    file_path = os.listdir('auto/new_data')
    file_paths = [path for path in file_path if 'ferrous'in path or 'Ferrous' in path]
    filenames = file_paths.copy()
    first_time = 0

    try:
        updatepath = 'auto/data/'
        path_update = f'{updatepath}9_2原料交易量.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        DF = json.loads(f)
        jsonFile.close()
    except:
        first_time = 1
        DF = {}
        DF['Ferrous'] = {}
        DF['Non-ferrous'] = {}

    for file_path in file_paths:
        df = pd.read_excel(f'auto/new_data/{file_path}', index_col=0)
        cleaned_filename = file_path.replace('9_2_', '').replace('.xlsx', '').split('_')
        if first_time == 1:
            DF[cleaned_filename[0]][cleaned_filename[1]] = {}
        data = DF[cleaned_filename[0]][cleaned_filename[1]]
        for i in range(0,len(df)):
            if i == 0:
                columns = df.columns
                for column in columns:
                    if column not in data.keys():
                        data[column] = []
            if df.iloc[i]['date'] in data['date'] and df.iloc[i]['CONTRACT'] == data['CONTRACT'][-1]:
                continue
            for column in columns:
                data[column].append(str(df.iloc[i][column]))

    storepathpath = 'auto/data/'
    path_store = f'{updatepath}9_2原料交易量.json'
    json_data = json.dumps(DF)
    with open(path_store, "w") as json_file:
        json_file.write(json_data) 


    message = f'更新完成: 鋼鐵價格資料'
    return render_template('update_product.html', message=message)


@app.route('/economic_analysis', methods=['GET', 'POST'])
def economic_analysis():
    '''概況總覽'''
    jsonFile = open("auto/data/5_經濟數據_all.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df5 = json.loads(f)      # 再使用 loads
    jsonFile.close()
    df5_labels = list([tmp for tmp in df5.keys() if '_date' not in tmp and '_unit' not in tmp])

    '''IMD競爭力指標'''
    updatepath = 'auto/data/'
    path_update = f'{updatepath}6_IMD競爭力指標.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df6 = json.loads(f)
    jsonFile.close()
    df6_labels = list(tmp for tmp in df6.keys() if '_score' not in tmp and '_time' not in tmp)

    '''機場吞吐量'''
    jsonFile = open("auto/data/10_機場吞吐量.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df10 = json.loads(f)      # 再使用 loads
    jsonFile.close()
    df10_labels = list([tmp for tmp in df10.keys() if '_時間'])

    '''SWIFT'''
    jsonFile = open("auto/data/12_SWIFT各幣別支付占比.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df12 = json.loads(f)      # 再使用 loads
    jsonFile.close()
    df12_labels = list(tmp for tmp in df12.keys())

    '''TradingEconomics'''
    jsonFile = open("auto/data/13_TradingEconomics.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df13 = json.loads(f)      # 再使用 loads
    jsonFile.close()
    df13_labels = list(tmp for tmp in df13.keys())

    '''中國汽車數據'''
    updatepath = 'auto/data/'
    path_update = f'{updatepath}14_中國汽車銷量.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df14 = json.loads(f)
    jsonFile.close()
    df14_labels = list(tmp for tmp in df14.keys())


    return render_template('economic_analysis.html',
                           df5_labels = df5_labels, df5 = df5,
                           df6_labels = df6_labels, df6 = df6,
                           df10_labels = df10_labels, df10 = df10,
                           df12_labels = df12_labels, df12 = df12, 
                           df13_labels = df13_labels, df13 = df13,
                           df14_labels = df14_labels, df14 = df14)

@app.route('/update/get_data6', methods=['GET', 'POST'])
def update_data6():
    # 更新資料
    get_data6()

    # 寫入json
    paths = os.listdir('auto/new_data')
    dirs = [tmp for tmp in paths if '6_IMD' in tmp ]

    try:
        updatepath = 'auto/data/'
        path_update = f'{updatepath}6_IMD競爭力指標.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        DF = json.loads(f)
        jsonFile.close()
    except:
        DF = {}

    for i in range(0,len(dirs)):
        df = pd.read_excel(f'auto/new_data/{dirs[i]}', index_col = 0)
        GEO = dirs[i].split('_')[-1].split('.')[0]
        if GEO not in DF.keys():
            DF[GEO] = {}
        for j in range(0,len(df)):
            for column in df.columns:
                if column not in DF[GEO].keys() or f'{column}_time' not in DF[GEO].keys():
                    DF[GEO][column] = []
                    DF[GEO][f'{column}_time'] = []
                if j == 0:
                    if f"{column}_{df.index[j].split('_')[0]}" not in DF[GEO].keys() or f"{column}_{df.index[j].split('_')[1]}" not in DF[GEO].keys():
                        DF[GEO][f"{column}_{df.index[j].split('_')[0]}"] = []
                        DF[GEO][f"{column}_{df.index[j].split('_')[0]}_time"] = []
                    if df.index[j].split('_')[1] in DF[GEO][f"{column}_{df.index[j].split('_')[0]}_time"]:
                        continue
                    DF[GEO][f"{column}_{df.index[j].split('_')[0]}"].append(df[column].iloc[j])
                    DF[GEO][f"{column}_{df.index[j].split('_')[0]}_time"].append(df.index[j].split('_')[1])
                else:
                    if df.index[j] in DF[GEO][f'{column}_time']:
                        continue
                    DF[GEO][column].append(df[column].iloc[j])
                    DF[GEO][f'{column}_time'].append(df.index[j])

    path_store = 'auto/data/'
    path_store = f'{path_store}6_IMD競爭力指標.json'
    json_data = json.dumps(DF)
    with open(path_store, "w") as json_file:
        json_file.write(json_data) 


    message = f'更新完成: IMD競爭力資料'
    return render_template('update_economic.html', message=message)


@app.route('/update/get_data10', methods=['GET', 'POST'])
def update_data10():
    filepath = 'auto/new_data/'
    df = get_data_10()
    df.to_excel(f'{filepath}10_機場吞吐量.xlsx')

    df = pd.read_excel('auto/new_data/10_機場吞吐量.xlsx')
    try:
        updatepath = 'auto/data/'
        path_update = f'{updatepath}10_機場吞吐量.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        DF = json.loads(f)
        jsonFile.close()
    except:
        DF = {}


    for i in range(0,len(df)):
        if df.iloc[i]['機場'] not in DF.keys():
            DF[df.iloc[i]['機場']] = []
            DF[f"{df.iloc[i]['機場']}_時間"] = []
        if str(df.iloc[i]['時間']) in DF[f"{df.iloc[i]['機場']}_時間"]:
            break
        DF[df.iloc[i]['機場']].append(str(df.iloc[i]['航班數量']))
        DF[f"{df.iloc[i]['機場']}_時間"].append(str(df.iloc[i]['時間']))

    path_store = 'auto/data/'
    path_store = f'{path_store}10_機場吞吐量.json'
    json_data = json.dumps(DF)
    with open(path_store, "w") as json_file:
        json_file.write(json_data) 


    message = f'更新完成: 機場吞吐量 資料'
    return render_template('update_economic.html', message=message)

@app.route('/update/get_data12', methods=['GET', 'POST'])
def update_data12():
    filepath = 'auto/new_data/'
    df = get_data_12()
    df.to_excel(f'{filepath}12_SWIFT各幣別支付占比.xlsx')

    filepath = 'auto/new_data/'
    df = pd.read_excel(f'{filepath}12_SWIFT各幣別支付占比.xlsx', index_col = 0)

    updatepath = 'auto/data/'
    path_update = f'{updatepath}12_SWIFT各幣別支付占比.json'
    jsonFile = open(path_update,'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df_update = json.loads(f)
    jsonFile.close()

    for i in range(0,len(df)):
        if df['幣別'].iloc[i] not in df_update.keys():
            df_update[df['幣別'].iloc[i]] = {}
            df_update[df['幣別'].iloc[i]]['時間'] = []
            df_update[df['幣別'].iloc[i]]['SWIFT占比'] = []

        if len(df_update[df['幣別'].iloc[i]]['時間']) > 0:
            if df['最新公告時間'].iloc[i] == df_update[df['幣別'].iloc[i]]['時間'][-1]:
                break

        df_update[df['幣別'].iloc[i]]['時間'].append(df['最新公告時間'].iloc[i])
        df_update[df['幣別'].iloc[i]]['SWIFT占比'].append(df['占比'].iloc[i])

    json_data = json.dumps(df_update)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)


    message = f'更新完成: SWIFT 資料'
    return render_template('update_economic.html', message=message)

@app.route('/update/get_data13', methods=['GET', 'POST'])
def update_data13():
    querys = ['stocks','currencies', 'commodities', 'crypto', 'bonds']
    for query in querys:
        print(query)
        get_data13(query)

    message = f'更新完成: {query} 資料'
    return render_template('update_economic.html', message = message)

@app.route('/update/get_data14', methods=['GET', 'POST'])
def update_data14():
    get_data14()

    data = pd.read_excel('auto/new_data/14_中國汽車銷量.xlsx', index_col=0)
    try:
        updatepath = 'auto/data/'
        path_update = f'{updatepath}14_中國汽車銷量.json'
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        df_update = json.loads(f)
        jsonFile.close()
    except:
        df_update = {}
        
    df_update['中國汽車銷量'] = {}
    for i in range(0,len(data)):
        for j in range(len(data.columns)):
            if data.columns[j] not in df_update['中國汽車銷量'].keys():
                df_update['中國汽車銷量'][data.columns[j]] = []
            if j == 0:
                if data[data.columns[j]].iloc[i] in df_update['中國汽車銷量'][data.columns[j]]:
                    break
            df_update['中國汽車銷量'][data.columns[j]].append(str(data[data.columns[j]].iloc[i]))

    json_data = json.dumps(df_update)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)

    message = f'更新完成: 中國汽車銷量 資料'
    return render_template('update_economic.html', message = message)




if __name__ == '__main__':
    app.debug = True
    app.run()
