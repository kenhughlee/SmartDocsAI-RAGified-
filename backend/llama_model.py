import os
import logging
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login

print("XPU available:", torch.xpu.is_available())
print("Device name:", torch.xpu.get_device_name(0) if torch.xpu.is_available() else "No XPU found")

logging.basicConfig(level=logging.INFO)
load_dotenv()
login(os.getenv('HUGGINGFACE_TOKEN'))
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

class LlamaChat:
    def __init__(self):

        max_memory = {
            0: "8GiB",  # Assign 8GiB for GPU 0
            "cpu": "4GiB",  # Assign 4GiB for CPU
        }

        #.env file needs to be created and LLAMA model path and access token needs to be added.
        #dev needs to login to HuggingFace to get access to the model path. A token will be given if approved
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,token=os.getenv('HUGGINGFACE_TOKEN'))
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            token=os.getenv('HUGGINGFACE_TOKEN'),
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            offload_folder="./offload"
        )
    
    def generate(self, query, context):
        input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
