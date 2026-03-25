from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import os
from threading import Thread

# Optimization: Use all CPU cores for inference
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(2)

# Load once
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Optimization: bfloat16 is faster and uses less memory than float32
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cpu")

# Optimization: JIT compile the model for faster inference (first call is slower, then 1.5-2x faster)
model = torch.compile(model)

system_prompt = f"<|system|>\nYou are a helpful assistant.\n"


def generate_llm_response(user_input):
    print("Model received input:", repr(user_input))  # Debug
    prompt = f"{system_prompt}<|user|>\n{user_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    '''if stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer'''
    outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.4,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = response.split("<|assistant|>\n")[-1]
    return reply.strip()
