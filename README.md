LLM RAG for local documents

# setup
1. create a folder called docs, and put the documents(*.pdf, *.md) that you want to index in there.
2. Download mistral-7b-openorca.gguf2.Q4_0.gguf and all-MiniLM-L6-v2-f16.gguf from gpt4all and store them in the root directory.

# build and run
```
sudo docker build -t llm .
sudo docker run -p 5001:5001 llm
```

Then wait until the web app is ready. You will see this:
```bash
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://172.17.0.2:5001
Press CTRL+C to quit
```
In your web browser go to localhost:5001. Enter a query, and then wait for the answer. 

