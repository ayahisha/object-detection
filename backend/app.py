from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Example historical data (location to image mapping)
historical_data = {
    "San Francisco": "historical_sf.png",
    "New York": "historical_ny.png"
}

# Serve historical data
@app.route('/get_historical_data', methods=['GET'])
def get_historical_data():
    location = request.args.get('location')
    image = historical_data.get(location)
    if image:
        return jsonify({"image_url": f"/static/{image}"})
    else:
        return jsonify({"error": "No historical data found"}), 404

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
