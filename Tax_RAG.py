import streamlit as st
import torch
import json
import faiss
import numpy as np
import cohere
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import asyncio

# Ensure async loop is properly handled
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Load Environment Variables for Cohere API Key
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    st.error("Cohere API key not found. Set it as an environment variable.")
    st.stop()

cohere_client = cohere.Client(COHERE_API_KEY)

# ✅ Define model paths
MISTRAL_PATH = r"huggingface_models/Mistral-7B"
LLAMA_PATH = r"huggingface_models/Llama-2-7B"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ✅ Cache model loading to avoid repeated loading
@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH,
        torch_dtype=torch_dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        quantization_config=bnb_config,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, local_files_only=True)
    return model, tokenizer

llama_model, llama_tokenizer = load_model()
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer
)

st.write("✅ Model loaded successfully!")

# ✅ Load JSON Data
file_path = "chapter_III.json"
with open(file_path, "r", encoding="utf-8") as file:
    tax_data = json.load(file)

documents = []
for section, section_data in tax_data["Chapter III"].items():
    for subsection, details in section_data.get("sub_sections", {}).items():
        doc_text = f"{section} - {subsection}\nSummary: {details.get('summary', '')}\nConditions: {details.get('conditions', '')}\nExceptions: {details.get('exceptions', '')}"
        documents.append(doc_text)

# ✅ Cache embeddings and FAISS index
@st.cache_resource
def create_faiss_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index

embedding_model, faiss_index = create_faiss_index()

st.write("✅ FAISS index created!")

def rerank_results(query, retrieved_results):
    rerank_response = cohere_client.rerank(model="rerank-english-v2.0", query=query, documents=retrieved_results, top_n=6)
    return [retrieved_results[item.index] for item in rerank_response.results]

def retrieve_relevant_sections(query, top_k=10, threshold=0.5):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = faiss_index.search(query_embedding, top_k)
    similarities = distances[0]
    if max(similarities) < threshold:
        return []
    retrieved_results = [documents[idx] for idx in indices[0]]
    return rerank_results(query, retrieved_results)

def generate_answer(query):
    retrieved_docs = retrieve_relevant_sections(query)
    if not retrieved_docs:
        return "I don't have enough information to answer this question about Indian tax laws. Please rephrase or ask about a different tax-related topic."
    
    context = "\n".join(retrieved_docs)
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
6. If unsure about any details, explicitly acknowledge the limitations in the available information

Remember: It's better to provide a shorter, accurate answer than a comprehensive but potentially incorrect one. [/INST]</s>
"""

    response = llama_pipeline(prompt, max_new_tokens=256, temperature=temperature, do_sample=True)

    
    # Clean up the response to remove instruction text
    output = response[0]['generated_text']
    
    # Remove everything up to and including [/INST]
    if "[/INST]" in output:
        output = output.split("[/INST]", 1)[1].strip()
    
    # Remove <s> and </s> tags if present
    output = output.replace("<s>", "").replace("</s>", "").strip()
    
    # If the output is empty after cleanup, return a fallback message
    if not output:
        return "I couldn't generate a proper response. Please try rephrasing your question."
        
    return output

    response = llama_pipeline(prompt, max_new_tokens=256, temperature=0.4, do_sample=True)
    # Modified to handle Llama 2's output format
    response_text = response[0]['generated_text']
    # Find the position after our prompt
    start_marker = "Based on the provided tax documents:"
    if start_marker in response_text:
        start_pos = response_text.find(start_marker)
        extracted_response = response_text[start_pos:]
        return extracted_response
    else:
        # Fallback extraction method if the marker isn't found
        split_text = response_text.split("[/INST]")
        if len(split_text) > 1:
            return split_text[-1].strip()
        return response_text.split("USER QUERY:")[-1].strip()
# ✅ Streamlit UI
st.title("🔍 RAG-based Tax Law Assistant")
st.write("Ask questions about tax laws and get AI-powered responses.")

with st.sidebar:
    st.title("🦙💬 Llama 2 Chatbot")
    st.write("This chatbot is created using the open-source Llama 2 LLM model from Meta.")
    temperature = st.sidebar.slider("Temperature", min_value=0.01, max_value=1.0, value=0.7, step=0.01)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your tax-related question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response_text = generate_answer(prompt)
        message_placeholder.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
