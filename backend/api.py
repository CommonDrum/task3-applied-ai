from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

recommender = None
df = None

def setup():
    global recommender, df
    df = pd.read_csv('song_dataset.csv')
    with open('recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)

def get_recommendations(user_id):
    try:
        user = df[df['user_id'] == user_id].iloc[0]['user_id']
        song_id, score = recommender.recommend_songs(user)
        
        output = []
        for i, song in enumerate(song_id):
            song_title = df[df['song_id'] == song]['title'].values[0]
            output.append(f"{i+1}. {song_title} with score {score[i]}")
        return output
    except Exception as e:
        return [f"Error getting recommendations: {str(e)}"]

@app.route('/recommendations', methods=['GET'])
def recommendations():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id parameter is required"}), 400
    return jsonify(get_recommendations(user_id))

if __name__ == '__main__':
    setup()
    app.run(port=5000)