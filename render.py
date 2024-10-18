from flask import Flask, request, render_template, redirect, url_for

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
    return render_template('product_analysis.html')

@app.route('/operation_condition', methods=['GET', 'POST'])
def operation_condition():
    return render_template('operation_condition.html')

@app.route('/economic_analysis', methods=['GET', 'POST'])
def economic_analysis():
    return render_template('economic_analysis.html')

@app.route('/government', methods=['GET', 'POST'])
def government():
    return render_template('government.html')

if __name__ == '__main__':
    app.run()
