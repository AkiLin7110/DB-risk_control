from flask import Flask, request, render_template, redirect, url_for, jsonify
from auto.auto_crawl import get_data1, get_data2, get_data3, get_data4, get_data5, get_data6, get_data7_1, get_data7_2, get_data7_3, get_data7_4, get_data7_5, get_data7_6, get_data8_1, get_data8_2, get_data9_1, get_data9_2, get_data_10
import json
import pandas as pd
from operator import itemgetter

app = Flask(__name__)


@app.route('/')
@app.route('/hello')
def hello():
    return 'Hello World'


@app.route('/text')
def text():
    return '<html><body><h1>Hello World</h1></body></html>'


@app.route('/home')
def home():
    return render_template('home.html')

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
    GEO = request.args.get('GEO', "")  # Get GEO from URL parameters, default to empty
    if GEO:
        df = pd.read_excel(f'auto/data/2_google_{GEO}.xlsx', index_col=0)
        data_indexes = df.index.astype(str).to_list()
        label_2_1 = ['tool machine']
        label_2_2 = ['controller']
        data_controller = df.iloc[:, 0].tolist()
        data_machine = df.iloc[:, 1].tolist()
    else:
        data_indexes = []
        label_2_1 = []
        label_2_2 = []
        data_controller = []
        data_machine = []

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
    jsonFile = open("auto/data/5_經濟數據_united-states_overview.json",'r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df5 = json.loads(f)      # 再使用 loads
    jsonFile.close()
    df5_labels = list([tmp for tmp in df5.keys() if '_date' not in tmp and '_unit' not in tmp])

    '''航運數據'''
    jsonFile = open('auto/data/7_上海航運交易所_航運數據.json','r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    df7 = json.loads(f)
    jsonFile.close()



    return render_template('product_analysis.html', indexs=indexs, titles=titles,
                           label_im=label_im, data_im_values=data_im_values, 
                           label_ex=label_ex, data_ex_values=data_ex_values,
                           GEO=GEO, data_indexes=data_indexes, 
                           label_2_1=label_2_1, label_2_2=label_2_2, 
                           data_controller=data_controller, data_machine=data_machine,
                           indexs_3 = indexs_3, label_3 = label_3, data = data,
                           tables = [df4.to_html(classes='data', header="true")],
                           df5_labels = df5_labels, df5 = df5,
                        #    df7_1 = df7_1, df7_1_labels = df7_1_labels, df7_1_indexs = df7_1_indexs
                            df7 = df7
                           )

@app.route('/operation_condition', methods=['GET', 'POST'])
def operation_condition():
    return render_template('operation_condition.html')

@app.route('/economic_analysis', methods=['GET', 'POST'])
def economic_analysis():
    return render_template('economic_analysis.html')

@app.route('/government', methods=['GET', 'POST'])
def government():
    return render_template('government.html')

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
    filepath = 'auto/new_data/'
    GEO = request.form.get('GEO')
    print(GEO)
    df = get_data2(GEO)
    df.to_excel(f'{filepath}2_google_{GEO}.xlsx')
    message = '更新完成: 搜尋資料'
    # return redirect(url_for('product_analysis', GEO=GEO))
    return render_template('update_product.html', message = message, GEO=GEO)

@app.route('/update/get_data3', methods=['GET', 'POST'])
def update_data3():
    filepath = 'auto/new_data/'
    lastet_month = '11308'
    df = get_data3(lastet_month)
    df.to_excel(f'{filepath}/3_主要國家工業生產增加率.xlsx')
    message = '更新完成: 工業生產增加率'
    return render_template('update_product.html', message = message)

@app.route('/update/get_data5', methods=['GET', 'POST'])
def update_data5():
    filepath = 'auto/new_data/'
    GEO = 'united-states'
    df = get_data5(GEO)
    df.to_excel(f'{filepath}/5_經濟數據_{GEO}.xlsx', index = 0)

    # 寫入到舊有的excel檔的不同sheet裡面
    path_new = 'auto/new_data/5_經濟數據_united-states.xlsx'
    path_update = 'auto/data/5_經濟數據_united-states_overview.json'

    df = pd.read_excel(path_new) 
    index_names = df[df['總經種類'] == 'Overview']['指標名稱'].values

    try:
        jsonFile = open(path_update,'r')
        f =  jsonFile.read()   # 要先使用 read 讀取檔案
        df_update = json.loads(f)
        jsonFile.close()
    except:
        df_update = {}

    for tmp in index_names:
        if tmp not in df_update.keys():
            df_update[tmp] = []
            df_update[f'{tmp}_date'] = []
            df_update[f'{tmp}_unit'] = []
        # if len(df_update[f'{tmp}_date']) > 0:
        #     if df_update[f'{tmp}_date'][-1] == df[df['指標名稱'] == tmp]['公告日期'].values[0]:
        #         break        
        df_update[tmp].append(df[df['指標名稱'] == tmp]['Last'].values[0])
        df_update[f'{tmp}_date'].append(df[df['指標名稱'] == tmp]['公告日期'].values[0])
        df_update[f'{tmp}_unit'].append(df[df['指標名稱'] == tmp]['單位'].values[0])

    json_data = json.dumps(df_update)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)

    message = f'更新完成: {GEO} 概況'
    return render_template('update_product.html', message = message)

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
    df7_2.to_excel('new_data/7_上海航運交易所_港口班輪準確率.xlsx', index = 0)
    df7_2 = pd.read_excel('auto/new_data/7_上海航運交易所_港口班輪準確率.xlsx').astype(str)
    df7_2_1 = df7_2.iloc[:,[0,1,7]]
    df7_2_2 = df7_2.iloc[:,[0,2,8]]
    df7_2_3 = df7_2.iloc[:,[0,3,9]]
    df7_2_4 = df7_2.iloc[:,[0,4,10]]
    df7_2_5 = df7_2.iloc[:,[0,5,11]]

    df7_3 = get_data7_3()
    df7_3.to_excel('new_data/7_上海航運交易所_一帶一路航貿指數.xlsx', index = 0)
    df7_3 = pd.read_excel('auto/new_data/7_上海航運交易所_一帶一路航貿指數.xlsx', index_col = 0).astype(str)

    df7_4 = get_data7_4()
    df7_4.to_excel('new_data/7_上海航運交易所_一帶一路貿易額指數.xlsx', index = 0)
    df7_4 = pd.read_excel('auto/new_data/7_上海航運交易所_一帶一路航貿指數.xlsx', index_col = 0).astype(str)
    


    jsonFile = open('auto/data/7_上海航運交易所_航運數據.json','r')
    f =  jsonFile.read()   # 要先使用 read 讀取檔案
    shipping_db = json.loads(f)      # 再使用 loads
    # shipping_db = {}
    jsonFile.close()

    df1_keys = list(df1['指數名稱'].values+'_time')+list(df1['指數名稱'].values)
    df2_keys = list(df2['航線'].values+'_time')+list(df2['航線'].values)
    df7_2_keys = list(df7_2['港口'].values+'_time')+list(df7_2['港口'].values)
    df7_3_keys = list(df7_3['指數名稱'].values+'_time')+list(df7_3['指數名稱'].values)



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
            
            # new_time = update_data[update_data[f'{query}'] == tmp_new].iloc[:,2].values[0]
            # if '_time' in tmp:
            #     if len(shipping_db[f'{column}'][tmp_new+'_time']) > 0:
            #         if shipping_db[f'{column}'][tmp_new+'_time'][-1] == new_time:
            #             break

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

    path_update = 'auto/data/7_上海航運交易所_航運數據.json'
    json_data = json.dumps(shipping_db)
    with open(path_update, "w") as json_file:
        json_file.write(json_data)

    message = f'更新完成: 所有航運數據'
    return render_template('update_product.html', message = message)




if __name__ == '__main__':
    app.debug = True
    app.run()
