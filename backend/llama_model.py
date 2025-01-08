import os
from transformers import LlamaTokenizer, LlamaForCausalLM

class LlamaChat:
    def __init__(self):
        #.env file needs to be created and LLAMA model path needs to be added.
        #dev needs to get LLAMA model path after getting access eg. MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(os.environ.get("MODEL_PATH"))
        self.model = LlamaForCausalLM.from_pretrained(os.environ.get("MODEL_PATH"))

    def generate(self, query, context):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
