from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
import openai

app = Flask(__name__)
CORS(app) 

gpt_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

chroma_client = chromadb.PersistentClient(path="./.model")
# 加载你的 collection
collection = chroma_client.get_collection(name="ragger")

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')

    # 检索知识库
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )

    documents = results["documents"][0]
    context = "\n".join(documents)

    # 组织Prompt
    final_prompt = f"""你是一个智能问答助手，以下是一些参考资料：
{context}

根据以上资料，回答用户的问题：{user_question}
如果资料中找不到答案，请礼貌地告诉用户不知道。"""

    # 调用 DeepSeek大模型
    response = gpt_client.chat.completions.create(
        model="deepseek/deepseek-v3-base:free",
        messages=[
            {"role": "system", "content": "你是一个知识问答机器人。"},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
    )

    gpt_answer = response["choices"][0]["message"]["content"]

    return jsonify({"answer": gpt_answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
