import streamlit as st
import json
import faiss
import numpy as np
import cohere
import openai
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (fallback)
load_dotenv()

# ✅ Load JSON Data with error handling
@st.cache_data
def load_tax_data():
    try:
        file_path = "chapter_III.json"
        with open(file_path, "r", encoding="utf-8") as file:
            tax_data = json.load(file)
        
        documents = []
        for section, section_data in tax_data.get("Chapter III", {}).items():
            for subsection, details in section_data.get("sub_sections", {}).items():
                doc_text = f"{section} - {subsection}\nSummary: {details.get('summary', '')}\nConditions: {details.get('conditions', '')}\nExceptions: {details.get('exceptions', '')}"
                documents.append(doc_text)
        
        if not documents:
            st.error("No tax documents found in the JSON file.")
            st.stop()
            
        return documents
    except FileNotFoundError:
        st.error("chapter_III.json file not found. Please ensure the file exists.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON format in chapter_III.json.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading tax data: {e}")
        st.stop()

# ✅ Cache embeddings and FAISS index
@st.cache_resource
def create_faiss_index(documents):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return model, index
    except Exception as e:
        st.error(f"Failed to create FAISS index: {e}")
        st.stop()

def validate_api_key(api_key, provider):
    """Basic validation for API key format."""
    if not api_key or api_key.strip() == "":
        return False
    
    # Basic format checks
    if provider == "OpenAI" and not api_key.startswith("sk-"):
        return False
    elif provider == "Anthropic" and not api_key.startswith("sk-ant-"):
        return False
    elif provider == "Hugging Face" and not api_key.startswith("hf_"):
        return False
    elif provider == "Cohere" and len(api_key.strip()) < 10:
        return False
    
    return True

def test_api_connection(api_key, provider):
    """Test API connection with a simple request."""
    try:
        if provider == "OpenAI":
            openai.api_key = api_key
            response = openai.Model.list()
            return True, "✅ OpenAI connection successful"
        
        elif provider == "Anthropic":
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            # Test with a minimal request
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ Anthropic connection successful"
            else:
                return False, f"❌ Anthropic API error: {response.status_code}"
        
        elif provider == "Hugging Face":
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                return True, "✅ Hugging Face connection successful"
            else:
                return False, f"❌ Hugging Face API error: {response.status_code}"
        
        elif provider == "Cohere":
            client = cohere.Client(api_key)
            # Test with a simple generate request
            response = client.generate(prompt="Test", max_tokens=1)
            return True, "✅ Cohere connection successful"
    
    except Exception as e:
        return False, f"❌ {provider} connection failed: {str(e)}"

def rerank_results(query, retrieved_results, cohere_api_key):
    """Rerank results using Cohere with fallback."""
    try:
        client = cohere.Client(cohere_api_key)
        rerank_response = client.rerank(
            model="rerank-english-v2.0", 
            query=query, 
            documents=retrieved_results, 
            top_n=min(6, len(retrieved_results))
        )
        return [retrieved_results[item.index] for item in rerank_response.results]
    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Using original order.")
        return retrieved_results[:6]  # Return top 6 without reranking

def retrieve_relevant_sections(query, embedding_model, faiss_index, documents, cohere_api_key, top_k=10, threshold=0.5):
    """Retrieve relevant sections with error handling."""
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = faiss_index.search(query_embedding, top_k)
        similarities = distances[0]
        
        if max(similarities) < threshold:
            return []
            
        retrieved_results = [documents[idx] for idx in indices[0] if idx < len(documents)]
        
        if cohere_api_key:
            return rerank_results(query, retrieved_results, cohere_api_key)
        else:
            return retrieved_results[:6]  # Return top 6 without reranking
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

def call_openai_api(prompt, api_key, temperature=0.7, max_tokens=256):
    """Call OpenAI API with error handling."""
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tax law assistant specializing in Indian tax laws. Provide accurate, concise answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return f"Error calling OpenAI API: {str(e)}"

def call_anthropic_api(prompt, api_key, temperature=0.7, max_tokens=256):
    """Call Anthropic Claude API with error handling."""
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            return f"Anthropic API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        return f"Error calling Anthropic API: {str(e)}"

def call_huggingface_api(prompt, api_key, temperature=0.7, max_tokens=256):
    """Call Hugging Face Inference API with error handling."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            return str(result)
        else:
            return f"Hugging Face API error: {response.status_code} - {response.text}"
            
    except Exception as e:
        logger.error(f"Hugging Face API call failed: {e}")
        return f"Error calling Hugging Face API: {str(e)}"

def generate_answer(query, embedding_model, faiss_index, documents, provider, api_key, cohere_api_key, temperature=0.7):
    """Generate answer using RAG approach with API-based LLM."""
    retrieved_docs = retrieve_relevant_sections(query, embedding_model, faiss_index, documents, cohere_api_key)
    
    if not retrieved_docs:
        return "I don't have enough information to answer this question about Indian tax laws. Please rephrase or ask about a different tax-related topic."
    
    context = "\n".join(retrieved_docs)
    
    if provider == "OpenAI":
        prompt = f"""Based on the following Indian tax law context, answer the user's question. Provide accurate, concise information based ONLY on the given context.

CONTEXT:
{context}

USER QUERY: {query}

Instructions:
- Use only information from the provided context
- If context is insufficient, clearly state this limitation
- Be concise and factual
- Quote specific sections when relevant"""
        
        return call_openai_api(prompt, api_key, temperature)
    
    elif provider == "Anthropic":
        prompt = f"""You are a tax law assistant specializing in Indian tax laws. Answer the question based ONLY on the provided context.

CONTEXT:
{context}

USER QUERY: {query}

Instructions:
1. Respond exclusively with information from the context
2. If context doesn't contain sufficient information, say so clearly
3. Never make up information or use outside knowledge
4. Extract and quote specific sections from the context
5. Keep your answer concise and factual

Answer:"""
        
        return call_anthropic_api(prompt, api_key, temperature)
    
    elif provider == "Hugging Face":
        prompt = f"""<s>[INST] You are a tax law assistant that answers questions about Indian tax laws. Your answers must be based ONLY on the information provided in the context below.

CONTEXT:
{context}

USER QUERY: {query}

IMPORTANT INSTRUCTIONS:
1. Respond EXCLUSIVELY with information found in the context
2. If the context doesn't contain enough information to answer fully, say: "The provided tax documents don't contain sufficient information to answer this question completely."
3. NEVER make up information or rely on outside knowledge
4. Extract and quote SPECIFIC sections from the context
5. Keep your answer concise and factual

[/INST] Based on the provided tax documents: """
        
        return call_huggingface_api(prompt, api_key, temperature)
    
    else:
        return "Invalid provider selected."

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="Tax Law Assistant", page_icon="🔍", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
.api-key-input {
    background-color: #f0f2f6;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.status-success {
    color: #00ff00;
    font-weight: bold;
}
.status-error {
    color: #ff0000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for API keys
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'cohere': os.getenv("COHERE_API_KEY", ""),
        'openai': os.getenv("OPENAI_API_KEY", ""),
        'anthropic': os.getenv("ANTHROPIC_API_KEY", ""),
        'huggingface': os.getenv("HUGGINGFACE_API_KEY", "")
    }

if 'api_status' not in st.session_state:
    st.session_state.api_status = {}

# Sidebar for API configuration
with st.sidebar:
    st.title("🔍 Tax Law Assistant")
    st.write("Configure your API keys to get started")
    
    # API Keys Configuration
    st.subheader("🔑 API Keys Configuration")
    
    # Provider selection
    provider = st.selectbox(
        "Choose AI Provider:",
        ["OpenAI", "Anthropic", "Hugging Face"],
        help="Select the AI provider for generating responses"
    )
    
    # Cohere API Key (Required for reranking)
    st.markdown("**Cohere API Key** (Required for better results)")
    cohere_key = st.text_input(
        "Cohere API Key:",
        value=st.session_state.api_keys['cohere'],
        type="password",
        help="Get your free API key from https://cohere.ai/"
    )
    st.session_state.api_keys['cohere'] = cohere_key
    
    # Test Cohere connection
    if st.button("Test Cohere Connection", key="test_cohere"):
        if cohere_key:
            success, message = test_api_connection(cohere_key, "Cohere")
            st.session_state.api_status['cohere'] = (success, message)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning("Please enter Cohere API key")
    
    if 'cohere' in st.session_state.api_status:
        success, message = st.session_state.api_status['cohere']
        if success:
            st.markdown(f'<p class="status-success">{message}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="status-error">{message}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Provider-specific API Key
    if provider == "OpenAI":
        st.markdown("**OpenAI API Key** (Required)")
        provider_key = st.text_input(
            "OpenAI API Key:",
            value=st.session_state.api_keys['openai'],
            type="password",
            help="Get your API key from https://platform.openai.com/"
        )
        st.session_state.api_keys['openai'] = provider_key
        test_key = "test_openai"
        
    elif provider == "Anthropic":
        st.markdown("**Anthropic API Key** (Required)")
        provider_key = st.text_input(
            "Anthropic API Key:",
            value=st.session_state.api_keys['anthropic'],
            type="password",
            help="Get your API key from https://console.anthropic.com/"
        )
        st.session_state.api_keys['anthropic'] = provider_key
        test_key = "test_anthropic"
        
    elif provider == "Hugging Face":
        st.markdown("**Hugging Face API Key** (Required)")
        provider_key = st.text_input(
            "Hugging Face API Key:",
            value=st.session_state.api_keys['huggingface'],
            type="password",
            help="Get your token from https://huggingface.co/settings/tokens"
        )
        st.session_state.api_keys['huggingface'] = provider_key
        test_key = "test_huggingface"
    
    # Test provider connection
    if st.button(f"Test {provider} Connection", key=test_key):
        if provider_key:
            success, message = test_api_connection(provider_key, provider)
            st.session_state.api_status[provider.lower()] = (success, message)
            if success:
                st.success(message)
            else:
                st.error(message)
        else:
            st.warning(f"Please enter {provider} API key")
    
    if provider.lower() in st.session_state.api_status:
        success, message = st.session_state.api_status[provider.lower()]
        if success:
            st.markdown(f'<p class="status-success">{message}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="status-error">{message}</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Temperature slider
    temperature = st.slider(
        "Temperature", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.7, 
        step=0.01,
        help="Higher values make responses more creative, lower values more focused"
    )
    
    # Clear API status
    if st.button("Clear Status", key="clear_status"):
        st.session_state.api_status = {}
        st.rerun()

# Check if required API keys are provided
required_keys_present = bool(provider_key and cohere_key)

if not required_keys_present:
    st.warning("⚠️ Please configure your API keys in the sidebar to start using the assistant.")
    st.info("💡 **Quick Start:**\n1. Get a free Cohere API key from https://cohere.ai/\n2. Get an API key from your chosen provider\n3. Enter both keys in the sidebar\n4. Test the connections\n5. Start asking questions!")
    
    # API Provider Information
    with st.expander("📋 API Provider Information"):
        st.markdown("""
        ### API Providers & Pricing
        
        **🔄 Cohere (Required for reranking)**
        - Free tier: 1,000 calls/month
        - Sign up: https://cohere.ai/
        
        **🤖 OpenAI**
        - GPT-3.5-turbo: $0.0015/1K tokens (input), $0.002/1K tokens (output)
        - GPT-4: Higher pricing, better quality
        - Sign up: https://platform.openai.com/
        
        **🎭 Anthropic**
        - Claude 3 Haiku: $0.25/1M tokens (input), $1.25/1M tokens (output)  
        - Claude 3 Sonnet: Higher pricing, better quality
        - Sign up: https://console.anthropic.com/
        
        **🤗 Hugging Face**
        - Free tier available with rate limits
        - Paid plans for higher usage
        - Sign up: https://huggingface.co/
        """)
else:
    # Load components with progress indicators
    try:
        with st.spinner("Loading tax data..."):
            documents = load_tax_data()
        
        with st.spinner("Creating search index..."):
            embedding_model, faiss_index = create_faiss_index(documents)
        
        st.success("✅ All components loaded successfully!")
        
        # Main interface
        st.title("🔍 RAG-based Tax Law Assistant")
        st.markdown(f"**Current Provider:** {provider}")
        st.write("Ask questions about Indian tax laws and get AI-powered responses based on Chapter III documents.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hello! I'm your tax law assistant powered by {provider}. How can I help you with Indian tax law questions today?"}
            ]
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your tax-related question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner(f"Searching documents and generating response using {provider}..."):
                    response_text = generate_answer(
                        prompt, embedding_model, faiss_index, documents, 
                        provider, provider_key, cohere_key, temperature
                    )
                st.markdown(response_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    except Exception as e:
        st.error(f"Error initializing the application: {e}")

# Help section
with st.expander("❓ Help & Instructions"):
    st.markdown("""
    ### How to Use:
    1. **Configure API Keys**: Enter your API keys in the sidebar
    2. **Test Connections**: Use the test buttons to verify your keys work
    3. **Choose Provider**: Select your preferred AI provider
    4. **Ask Questions**: Start asking questions about Indian tax laws
    
    ### Troubleshooting:
    - **Connection Failed**: Check your API key format and internet connection
    - **No Response**: Try a different provider or check API quotas
    - **Empty Responses**: Adjust temperature or rephrase your question
    
    ### Tips:
    - Start with simple questions to test the system
    - Be specific about tax topics for better results
    - Lower temperature (0.1-0.3) for more factual responses
    - Higher temperature (0.7-0.9) for more creative responses
    """)
