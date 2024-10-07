from flask import Flask, request, jsonify
import module_arousal as arousal
import os

app = Flask(__name__)

@app.route('/infer_arousal', methods=['POST'])
def infer_arousal():
    data = request.get_json()
    if 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400

    file_path = data['file_path']
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File does not exist'}), 400

    # Run the arousal inference
    arousal_label, arousal_confidence = arousal.API(file_path)

    return jsonify({'arousal_label': arousal_label, 'arousal_confidence': arousal_confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)