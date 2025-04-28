from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb

app = Flask(__name__)
CORS(app) 

chroma_client = chromadb.PersistentClient(path="./.model")
# 加载你的 collection
collection = chroma_client.get_collection(name="ragger")

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # 向量检索
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )

    documents = results["documents"][0]
    context = "\n".join(documents)

    answer = f"根据资料：\n{context}\n\n回答你的问题：'{user_question}'。"
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
