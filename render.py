from flask import Flask, request, render_template, redirect, url_for
from auto.auto_crawl import get_data1, get_data2, get_data3, get_data4, get_data5, get_data6, get_data7_1, get_data7_2, get_data7_3, get_data7_4, get_data7_5, get_data7_6, get_data8_1, get_data8_2, get_data9_1, get_data9_2, get_data_10

import pandas as pd

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
    # 寫入到舊有的excel檔的不同sheet裡面
    df5 = pd.read_excel('auto/data/5_經濟數據_united-states.xlsx')
    df5_labels = df5[(df5['總經種類'] == 'Overview')]['指標名稱'].to_list()

    return render_template('product_analysis.html', indexs=indexs, titles=titles,
                           label_im=label_im, data_im_values=data_im_values, 
                           label_ex=label_ex, data_ex_values=data_ex_values,
                           GEO=GEO, data_indexes=data_indexes, 
                           label_2_1=label_2_1, label_2_2=label_2_2, 
                           data_controller=data_controller, data_machine=data_machine,
                           indexs_3 = indexs_3, label_3 = label_3, data = data,
                           tables = [df4.to_html(classes='data', header="true")],
                           df5_labels = df5_labels)

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
    path_update = 'auto/data/5_經濟數據_united-states.xlsx'

    df = pd.read_excel(path_new) 
    index_names = df[df['總經種類'] == 'Overview']['指標名稱'].values
    with pd.ExcelWriter(engine = 'openpyxl', path = path_update, mode = 'a', if_sheet_exists = 'overlay') as writer: # mode='a'現有檔案讀寫
        tmp_df = pd.read_excel(path_update, sheet_name = None) 
        # 存到指定的sheet
        for col in index_names:
            if col in tmp_df.keys():
                old_df = pd.read_excel(path_update, sheet_name = col, index_col = 0) # 舊資料
                tmp = df[(df['總經種類'] == 'Overview') & (df['指標名稱'] == col)] # 新資料
                old_df = pd.concat([old_df, tmp])
                old_df.to_excel(writer, sheet_name = col) 
            else:
                print(col)
                tmp = tmp_df['Sheet1'][(tmp_df['Sheet1']['總經種類'] == 'Overview') & (tmp_df['Sheet1']['指標名稱'] == col)]
                tmp.to_excel(writer, sheet_name = col)
    message = f'更新完成: {GEO} 概況'
    return render_template('update_product.html', message = message)



if __name__ == '__main__':
    app.debug = True
    app.run()
