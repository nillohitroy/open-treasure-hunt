import os
import time
import requests
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from flask import Flask, render_template, request, jsonify
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from movement_classifier import analyze_movement

app = Flask(__name__)
geolocator = Nominatim(user_agent="open_treasure_hunt_app")

# Global state to track live positions in memory
game_state = {
    "protector": {"lat": None, "lon": None, "last_time": time.time()},
    "pirate": {"lat": None, "lon": None, "last_time": time.time()}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    try:
        data = request.json
        role = data.get('role')
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))

        if role not in game_state:
            return jsonify({"status": "error", "message": "Invalid role"}), 400

        # Update global state
        game_state[role]['lat'] = lat
        game_state[role]['lon'] = lon
        game_state[role]['last_time'] = time.time()

        prot_loc = (game_state['protector']['lat'], game_state['protector']['lon'])
        pir_loc = (game_state['pirate']['lat'], game_state['pirate']['lon'])

        # Check if both players have sent coordinates
        if prot_loc[0] is None or pir_loc[0] is None:
            return jsonify({
                "status": "success",
                "distance": "--",
                "geography": "Waiting for opponent...",
                "ai_message": "Waiting for the other team to join and share GPS...",
                "movement_status": "Paused"
            })

        # Calculate physical distance
        distance = int(geodesic(prot_loc, pir_loc).meters)

        # Reverse geocode terrain/address
        try:
            current_loc = prot_loc if role == 'protector' else pir_loc
            geo_info = geolocator.reverse(f"{current_loc[0]}, {current_loc[1]}", timeout=5)
            geography = geo_info.address if geo_info else "the wilderness"
        except Exception as e:
            print(f"Geopy error: {e}")
            geography = "an unknown location"

        # TabPFN Analysis
        speed = 1.2
        direction_changes = 15
        movement_status = analyze_movement(distance, speed, direction_changes)

        # Gemma Prompt
        prompt = f"""
        You are the game master of a real-world pirate treasure hunt.
        The current player is a {role} located at or near: {geography}.
        The distance between the pirate and the protector is {distance} meters.
        TabPFN analysis says the hunters are currently: {movement_status}.

        Write a short, tense 1-sentence message directed to the {role} mentioning their real-world surroundings.
        """

        if not HF_TOKEN:
            ai_message = "Error: HF_TOKEN missing from .env file."
        else:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "google/gemma-2-9b-it:featherless-ai",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 50
            }

            response = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20
            )

            if response.status_code != 200:
                print(f"HF ROUTER ERROR: {response.text}")
                ai_message = f"API Error: {response.json().get('error', 'Hugging Face call failed')}"
            else:
                ai_message = response.json()['choices'][0]['message']['content']

        return jsonify({
            "status": "success",
            "distance": distance,
            "geography": geography,
            "ai_message": ai_message,
            "movement_status": movement_status
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "distance": "--",
            "geography": "Error",
            "ai_message": f"Crash Log: {str(e)}",
            "movement_status": "Error"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)