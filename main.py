from gpt4all import GPT4All

model = GPT4All(model_name="rift-coder-v0-7b-q4_0.gguf")
output =model.generate("how do I download from an https url in Python?")
print(output)