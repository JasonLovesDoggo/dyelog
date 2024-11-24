FROM ollama:latest

RUN ollama install llama3.2

EXPOSE 11434
ENV OLLAMA_HOST=0.0.0.0 OLLAMA_PORT=11434

CMD ["ollama", "serve"]
