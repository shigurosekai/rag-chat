from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from openai import OpenAI

app = Flask(__name__)
CORS(app) 

gpt_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

chroma_client = chromadb.PersistentClient(path="./.model")
# 加载你的 collection
collection = chroma_client.get_collection(name="ragger")

def get_embedding_from_deepseek(text):
    response = gpt_client.embeddings.create(
        model="text-embedding-openrouter",   # OpenRouter统一用这个embedding入口
        input=text
    )
    embedding = response.data[0].embedding
    return embedding

@app.route('/api/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"answer": "请提供问题内容！"})

    try:
        # 先把用户问题变成向量
        query_embedding = get_embedding_from_deepseek(user_question)

        # 再用embedding去chroma检索
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        documents = results["documents"][0] if results["documents"] else []
        context = "\n".join(documents)

        if not context.strip():
            return jsonify({"answer": "未找到相关资料。请换个问题试试。"})

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

        if response.choices:
            gpt_answer = response.choices[0].message.content
        else:
            gpt_answer = "抱歉，未能生成回答。"
        print(f"最终发送给模型的Prompt：{final_prompt}")
    

        return jsonify({"answer": gpt_answer})

    except Exception as e:
        print("异常：", str(e))
        return jsonify({"answer": "服务器内部错误，请稍后再试～"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
