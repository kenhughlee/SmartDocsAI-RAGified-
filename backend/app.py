from flask import Flask, request, jsonify
from llama_model import LlamaChat
from document_loader import process_pdf
from embeddings.retriever import retrieve_documents

app = Flask(__name__)
llama_chat = LlamaChat()

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    embeddings = process_pdf(file)
    return jsonify({"message": "PDF processed successfully", "status": "success"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    query = data.get('query')
    results = retrieve_documents(query)
    response = llama_chat.generate(query, context=results)
    return jsonify({"response": response}), 200

if __name__ == '__main__':
    app.run(debug=True)
