from flask import Flask, url_for, render_template, request, redirect
from wtforms import Form, FloatField, validators
import algocrew
import algoimposter
import convertGameLength

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        gamerole = request.form["role"]
        return redirect(url_for("role", rle=gamerole))
    else:
        return render_template("index.html")


@app.route("/<rle>", methods=["POST", "GET"])
def role(rle):
    if request.method == "POST":
        gamerole = request.form["role"]
        return redirect(url_for("role", rle=gamerole))
    else:
        return render_template(rle)


@app.route('/crewmate')
def crewmate():
    return render_template('crewmate.html')


@app.route('/crewdata/', methods=['POST', 'GET'])
def crewdata():
    if request.method == 'POST':
        tasks = int(request.form.get('tasks'))
        if request.form.get('completed') == 'Yes':
            completed = 1
        else:
            completed = 0
        if request.form.get('killed') == 'Yes':
            killed = 1
        else:
            killed = 0
        time = convertGameLength.convert(request.form.get('time'))
        if request.form.get('ejected') == 'Yes':
            ejected = 1
        else:
            ejected = 0
        sabotages = int(request.form.get('sabotages'))
        calculation = str(algocrew.calculateCrew(tasks, completed, killed, time, ejected, sabotages) * 100)
        return render_template('crewdata.html', calculation=calculation[1:calculation.index('d')])
    else:
        return render_template("crewdata.html")


@app.route('/imposter')
def imposter():
    return render_template('imposter.html')


@app.route('/imposterdata/', methods=['POST', 'GET'])
def imposterdata():
    if request.method == 'POST':
        crewkilled = int(request.form.get('crewkilled'))
        time = convertGameLength.convert(request.form.get('time'))
        if request.form.get('ejected') == 'Yes':
            ejected = 1
        else:
            ejected = 0
        calculation = str(algoimposter.calculateImposter(crewkilled, time, ejected) * 100)
        return render_template('imposterdata.html', calculation=calculation[1:calculation.index('d')])
    else:
        return render_template("imposterdata.html")


if __name__ == '__main__':
    app.run(debug=True)
