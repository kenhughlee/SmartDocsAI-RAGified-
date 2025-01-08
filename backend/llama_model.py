import os
from transformers import LlamaTokenizer, LlamaForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.getenv('HUGGINGFACE_TOKEN'))
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

class LlamaChat:
    def __init__(self):
        #.env file needs to be created and LLAMA model path and access token needs to be added.
        #dev needs to login to HuggingFace to get access to the model path. A token will be given if approved
        self.tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH,use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
        self.model = LlamaForCausalLM.from_pretrained(MODEL_PATH,use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))

    def generate(self, query, context):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
