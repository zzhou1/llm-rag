A project for knowledge embedding from a set of pdf files. It is designed so that the container will not need to use the network at all while building.

Very much a work in prorgress. Currently it will create all of the embeddings and run a single query.

# setup
1. create a folder called pdfs, and put the pdfs that you want to index in there.
2. Download mistral-7b-openorca.gguf2.Q4_0.gguf and all-MiniLM-L6-v2-f16.gguf from gpt4all and store them in the root directory.

# build and run

```
sudo docker build -t llm .
sudo docker run -a stdin -a stdout -t -i llm /bin/bash
#python main.py
```

