from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///results.db'
db = SQLAlchemy(app)

# Ensure the static directory exists
os.makedirs('static', exist_ok=True)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500))
    emotion_result = db.Column(db.String(2000))
    gibberish_result = db.Column(db.String(2000))
    max_emotion_label = db.Column(db.String(50))
    max_emotion_score = db.Column(db.Float)
    max_gibberish_label = db.Column(db.String(50))
    max_gibberish_score = db.Column(db.Float)

    def __init__(self, text, emotion_result, gibberish_result, max_emotion_label, max_emotion_score, max_gibberish_label, max_gibberish_score):
        self.text = text
        self.emotion_result = emotion_result
        self.gibberish_result = gibberish_result
        self.max_emotion_label = max_emotion_label
        self.max_emotion_score = max_emotion_score
        self.max_gibberish_label = max_gibberish_label
        self.max_gibberish_score = max_gibberish_score

with app.app_context():
    db.create_all()

emotion_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
gibberish_classifier = pipeline(task="text-classification", model="wajidlinux99/gibberish-text-detector", top_k=None)

@app.route('/')
def index():
    results = Result.query.order_by(Result.id.desc()).all()  # Latest entry on top
    return render_template('index.html', results=results, emotion_plot=None, gibberish_plot=None, highest_emotion=None, highest_gibberish=None)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    text = request.form['text']

    emotion_results = emotion_classifier(text)
    gibberish_results = gibberish_classifier(text)

    if isinstance(emotion_results, list):
        max_emotion = max(emotion_results[0], key=lambda x: x['score'])
    else:
        max_emotion = max(emotion_results[0].items(), key=lambda x: x[1])[0]

    highest_emotion = (max_emotion['label'], max_emotion['score'])

    if isinstance(gibberish_results, list):
        max_gibberish = max(gibberish_results[0], key=lambda x: x['score'])
    else:
        max_gibberish = max(gibberish_results[0].items(), key=lambda x: x[1])[0]

    highest_gibberish = (max_gibberish['label'], max_gibberish['score'])

    new_result = Result(
        text=text,
        emotion_result=json.dumps(emotion_results),
        gibberish_result=json.dumps(gibberish_results),
        max_emotion_label=max_emotion['label'],
        max_emotion_score=max_emotion['score'],
        max_gibberish_label=max_gibberish['label'],
        max_gibberish_score=max_gibberish['score']
    )
    db.session.add(new_result)
    db.session.commit()

    # Plot emotion results
    plt.figure(figsize=(20, 10))
    if isinstance(emotion_results, list):
        emotion_labels = [result['label'] for result in emotion_results[0]]
        emotion_scores = [result['score'] for result in emotion_results[0]]
    else:
        emotion_labels = list(emotion_results[0].keys())
        emotion_scores = list(emotion_results[0].values())
    plt.bar(emotion_labels, emotion_scores)
    plt.xlabel('Emotion')
    plt.ylabel('Score')
    plt.title('Emotion Analysis')
    plt.xticks(rotation=45)
    emotion_plot_path = os.path.join('static', 'emotion_plot.png')
    plt.savefig(emotion_plot_path)
    plt.close()

    # Plot gibberish results
    plt.figure(figsize=(15, 7.5))
    if isinstance(gibberish_results, list):
        gibberish_labels = [result['label'] for result in gibberish_results[0]]
        gibberish_scores = [result['score'] for result in gibberish_results[0]]
    else:
        gibberish_labels = list(gibberish_results[0].keys())
        gibberish_scores = list(gibberish_results[0].values())
    plt.bar(gibberish_labels, gibberish_scores)
    plt.xlabel('Gibberish Level')
    plt.ylabel('Score')
    plt.title('Gibberish Analysis')
    gibberish_plot_path = os.path.join('static', 'gibberish_plot.png')
    plt.savefig(gibberish_plot_path)
    plt.close()

    return render_template('index.html', results=Result.query.order_by(Result.id.desc()).all(),
                           emotion_plot=url_for('static', filename='emotion_plot.png'),
                           gibberish_plot=url_for('static', filename='gibberish_plot.png'),
                           highest_emotion=highest_emotion, highest_gibberish=highest_gibberish)

@app.route('/reset', methods=['POST'])
def reset():
    db.session.query(Result).delete()
    db.session.commit()

    if os.path.exists(os.path.join('static', 'emotion_plot.png')):
        os.remove(os.path.join('static', 'emotion_plot.png'))
    if os.path.exists(os.path.join('static', 'gibberish_plot.png')):
        os.remove(os.path.join('static', 'gibberish_plot.png'))

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

