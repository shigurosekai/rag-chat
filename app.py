from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from openai import OpenAI
from huggingface_hub import InferenceClient
import requests
import logging

app = Flask(__name__)
CORS(app) 
app.logger.setLevel(logging.DEBUG)

gpt_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

chroma_client = chromadb.PersistentClient(path="./.model")
# 加载你的 collection
collection = chroma_client.get_collection(name="ragger")

hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token="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"
)

def get_embedding(text: str) -> list[float]:
    try:
        arr = hf_client.feature_extraction(text)
        return arr[0].tolist()
    except Exception as e:
        app.logger.error(f"Hugging Face 推理失败：{e}")
        return []


@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"answer": "请提供问题内容！"})

    try:
        app.logger.info(f"收到提问：{user_question}")
        query_embedding = get_embedding(user_question)
        if not query_embedding:
            return jsonify({"answer": "生成embedding失败"})

        app.logger.info(f"embedding前5位: {query_embedding[:5]}")
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        documents = results.get("documents", [[]])[0]
        # 再用embedding去chroma检索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        app.logger.info(f"检索到的文档数：{len(results['documents'][0])}")

        documents = results["documents"][0] if results["documents"] else []
        context = "\n".join(documents)

        if not context.strip():
            return jsonify({"answer": "未找到相关资料。请换个问题试试。"})

    # 组织Prompt
        final_prompt = f"""你是一个智能问答助手，以下是一些参考资料：
{context}

根据以上资料，回答用户的问题：{user_question}
如果资料中找不到答案，可以自行回答或礼貌的告诉用户不知道。"""

    # 调用 DeepSeek大模型
        response = gpt_client.chat.completions.create(
            model="deepseek/deepseek-v3-base:free",
            messages=[
                {"role": "system", "content": "你是一个知识问答机器人。"},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.2,
        )

        if response.choices:
            gpt_answer = response.choices[0].message.content
        else:
            gpt_answer = "抱歉，未能生成回答。"
        app.logger.info(f"最终发送给模型的Prompt：{final_prompt}")
    

        return jsonify({"answer": gpt_answer})

    except Exception as e:
        app.logger.error(f"服务器异常：{str(e)}")
        return jsonify({"answer": "服务器内部错误，请稍后再试～"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
