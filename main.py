from flask import Flask, request, render_template
from language import sentence_translator
app = Flask(__name__)
@app.route("/")
def my_form():
    return render_template('input_info.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = sentence_translator(text)
    return render_template('output_info.html', text = text, processed_text = processed_text)

'''
if __name__ == '__main__':
    app.run(debug=True)
'''