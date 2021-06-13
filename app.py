import os
from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime
from custom_ranking_algorithm import load_data, readFile


app = Flask(__name__)
fileFolder = os.path.dirname(os.path.realpath('__file__'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/table_chart')
def tableChart():
    return render_template('table_chart.html')

@app.route('/table_chart', methods=['POST', 'GET'])
def tableChartWithMethods():
    req = request.form.getlist("multiInput")

    # Checking whether req has other hashtags than in the list or not
    hashFile = os.path.join(fileFolder, 'static/hashtags.txt')
    hashtags = readFile(hashFile)
    hashtagsList = hashtags.split("\n")

    query = []
    for ele in req:
        if ele in hashtagsList:
            query.append(ele)

    if not query:
        hideH1 = "visible"
        hideTable = "hidden"
    else:
        hideH1 = "hidden"
        hideTable = "visible"
        load_data(query)

    return render_template('table_chart.html', hideH1=hideH1, hideTable=hideTable)


if __name__ == "__main__":
    app.run(debug=True)
