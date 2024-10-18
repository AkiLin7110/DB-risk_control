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
    df = pd.read_excel('auto/data/1_中國大陸_進出口總值(美元).xlsx', index_col = 0)
    df = df.fillna('Null')
    # df = df[df['category_x'] == '進口總值(含復進口)']
    indexs = df['date'].tolist()
    titles = ['可互換之搪孔或拉孔工具，工具機用']
    label_im = ['進口總值(含復進口)']
    # data_im_indexs = df.iloc[:, 1].tolist()  # Convert to list
    data_im_values = df['import_美元'].tolist()  # Convert to list

    # df = pd.read_excel('auto/data/1_中國大陸_進出口總值(美元).xlsx', index_col = 0)
    # df = df[df['category_y'] == '出口總值(含復出口)']
    label_ex = ['出口總值(含復出口)']
    # data_ex_indexs = df.iloc[:, 1].tolist()  # Convert to list
    data_ex_values = df['export_美元'].tolist()  # Convert to list

    return render_template('product_analysis.html', indexs = indexs, titles = titles,
                            label_im = label_im, data_im_values = data_im_values, 
                            label_ex = label_ex, data_ex_values = data_ex_values)

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





if __name__ == '__main__':
    app.debug = True
    app.run()
