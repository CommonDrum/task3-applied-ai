from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from scipy import sparse

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



def add_new_user(user_id, song_ids):
    if user_id in user_mapping:
        raise ValueError("User already exists")
    
    n_users = model.user_embeddings.shape[0]  # Original number of users
    n_songs = len(song_mapping)
    
    new_user_idx = n_users - 1
    user_mapping[user_id] = new_user_idx
    
    mapped_songs = [song_mapping[song] for song in song_ids if song in song_mapping]
    
    if not mapped_songs:
        raise ValueError("No valid songs found in input")
    
    new_interaction = sparse.coo_matrix(
        (np.ones(len(mapped_songs)), 
         ([new_user_idx] * len(mapped_songs), mapped_songs)),
        shape=(n_users, n_songs)
    ).tocsr()
    
    model.fit_partial(new_interaction)
    
    return new_user_idx

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        song_ids = data.get('song_ids', [])
        
        if not user_id or not song_ids:
            return jsonify({"error": "Missing required parameters"}), 400
            
        new_user_idx = add_new_user(user_id, song_ids)
        
        return jsonify({
            "message": "User added successfully",
            "user_idx": new_user_idx
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_user_index(user_id):
    if not user_id:
        raise ValueError("user_id parameter is required")
    
    if user_id not in user_mapping:
        raise ValueError("User ID not found")
        
    return user_mapping[user_id]

def get_song_recommendations(user_idx, n_recommendations):
    scores = model.predict(user_idx, np.arange(len(song_mapping)))
    top_indices = np.argsort(-scores)[:n_recommendations]
    
    reverse_mapping = {idx: song for song, idx in song_mapping.items()}
    recommendations = []
    
    for idx in top_indices:
        song_id = reverse_mapping[idx]
        song = df[df['song_id'] == song_id]
        recommendations.append({
            "title": song['title'].values[0],
            "artist": song['artist'].values[0],
            "score": float(scores[idx])
        })
    
    return recommendations

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        user_id = request.args.get('user_id')
        n_recommendations = int(request.args.get('n', 10))
        
        user_idx = get_user_index(user_id)
        recommendations = get_song_recommendations(user_idx, n_recommendations)
        
        return jsonify(recommendations)
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)