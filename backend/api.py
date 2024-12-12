from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load everything at startup
df = pd.read_csv('song_dataset.csv', header=None,
                 names=['user_id', 'song_id', 'play_count', 'title', 'album', 'artist', 'year'])
df = df[1:]

with open('model.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    model = saved_data['model']
    user_mapping = saved_data['user_mapping']
    song_mapping = saved_data['song_mapping']

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        user_id = request.args.get('user_id')
        n_recommendations = int(request.args.get('n', 10)) 
        
        if not user_id:
            return jsonify({"error": "user_id parameter is required"}), 400
            
        if user_id not in user_mapping:
            return jsonify({"error": "User ID not found"}), 404

        user_idx = user_mapping[user_id]
        n_songs = len(song_mapping)
        
        # Get predictions
        scores = model.predict(user_idx, np.arange(n_songs))
        
        # Get top N songs
        top_indices = np.argsort(-scores)[:n_recommendations]
        
        # Convert indices back to song IDs
        reverse_mapping = {idx: song for song, idx in song_mapping.items()}
        recommended_songs = [reverse_mapping[idx] for idx in top_indices]
        
        # Format response
        recommendations = [
            {
                "title": df[df['song_id'] == song]['title'].values[0],
                "artist": df[df['song_id'] == song]['artist'].values[0],
                "score": float(scores[idx])
            }
            for idx, song in zip(top_indices, recommended_songs)
        ]
        
        return jsonify(recommendations)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)