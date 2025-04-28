from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app) 

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # 暂时返回一个简单回答，后面加检索和LLM
    fake_answer = f"你问的是：'{user_question}'，但这里还没接真实RAG哦！"

    return jsonify({"answer": fake_answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
