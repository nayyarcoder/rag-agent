# Multi-Provider LLM Configuration Guide

This RAG agent now supports multiple LLM providers through LiteLLM integration, making it compatible with various cloud providers and local models.

## Supported Providers

### 1. Groq (Default)
Fast inference with open-source models.

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_groq_api_key_here
export LLM_MODEL=llama-3.1-8b-instant
```

**Available Models:**
- `llama-3.1-8b-instant` (fastest)
- `llama-3.1-70b-versatile` (balanced)
- `llama3-8b-8192` (long context)
- `mixtral-8x7b-32768` (expert model)

### 2. OpenAI
GPT models from OpenAI.

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key_here
export LLM_MODEL=gpt-4o-mini
```

**Available Models:**
- `gpt-4o` (most capable)
- `gpt-4o-mini` (cost-effective)
- `gpt-4-turbo` (balanced)
- `gpt-3.5-turbo` (fastest)

### 3. Anthropic
Claude models from Anthropic.

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
export LLM_MODEL=claude-3-haiku-20240307
```

**Available Models:**
- `claude-3-5-sonnet-20241022` (most capable)
- `claude-3-haiku-20240307` (fastest)
- `claude-3-opus-20240229` (most intelligent)

### 4. Ollama (Local)
Run models locally with Ollama (no API key required).

```bash
export LLM_PROVIDER=ollama
export OLLAMA_API_BASE=http://localhost:11434
export LLM_MODEL=llama3.1:8b
```

**Setup Ollama:**
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Pull a model: `ollama pull llama3.1:8b`
3. Start Ollama: `ollama serve`

**Available Models:**
- `llama3.1:8b` (8B parameters)
- `llama3.1:70b` (70B parameters)  
- `mistral:7b` (Mistral 7B)
- `codellama:7b` (Code-focused)
- `gemma:7b` (Google's Gemma)

## Environment Configuration

Copy `.env.example` to `.env` and configure your preferred provider:

```env
# LLM Provider Configuration
LLM_PROVIDER=groq

# API Keys (set only the one you need)
GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Ollama (local inference)
# OLLAMA_API_BASE=http://localhost:11434

# Model Configuration
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
```

## Docker Configuration

For Ollama with Docker:

```yaml
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    
  rag-agent:
    build: .
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_API_BASE=http://ollama:11434
      - LLM_MODEL=llama3.1:8b
    depends_on:
      - ollama

volumes:
  ollama_data:
```

## Usage Examples

### Switching Providers Dynamically

The Streamlit interface allows you to:
1. Select your LLM provider from the dropdown
2. Choose from provider-specific models
3. See API key configuration status
4. Initialize the chatbot with your selection

### Programmatic Usage

```python
from src.chatbot import RAGChatbot

# Initialize with specific provider
chatbot = RAGChatbot(
    collection_name="my_documents",
    llm_provider="openai",
    model_name="gpt-4o-mini"
)

# The provider is automatically configured from environment variables
response = chatbot.get_response("What is this document about?", [])
```

## Benefits

- **Provider Flexibility**: Switch between cloud and local models easily
- **Cost Optimization**: Choose models based on performance/cost needs
- **Privacy Options**: Use local Ollama for sensitive documents
- **Reliability**: Fallback to different providers if one is unavailable
- **Development**: Use faster local models for development, production models for deployment

## Migration from Groq-only

Existing installations continue to work unchanged. To migrate:

1. Set `LLM_PROVIDER=groq` to maintain current behavior
2. Gradually experiment with other providers by changing the environment variable
3. No code changes required - just configuration updates