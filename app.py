from flask import Flask, render_template

app = Flask(__name__)
app.static_folder = "static/"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)