from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from task import *

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Limit file size to 16 MB

json_result = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process(mp4file):
    # Replace this function with your actual processing logic
    # For demonstration, this function just prints the filename
    print(f"Processing file: {mp4file}")
    processor = VideoProcessing()
    result = processor.process(mp4file)
    print("Done processing:\n", result,"\n")
    return result

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("processing-----------",file_path)
            result = process(file_path)
            global json_result
            json_result = result
            return redirect(url_for('result'))

    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    global json_result
    return render_template('result.html', result = json_result)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
