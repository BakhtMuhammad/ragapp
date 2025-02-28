Multi-Source RAG App  

Overview  
This is a Retrieval-Augmented Generation (RAG) application that enables users to process and query information from multiple sources, including text files, PDFs, YouTube videos, and websites. The application leverages vector databases for document storage and LLM-based question answering using models from OpenAI, Anthropic, and Hugging Face.  

The app is built with Streamlit and LangChain, and it uses ChromaDB for vector storage.  

Features  
- Supports multiple data sources: Text files, PDFs, YouTube videos, and websites  
- Retrieval-Augmented Generation (RAG) for enhanced LLM responses  
- Multiple LLM providers: OpenAI, Anthropic, and Hugging Face  
- Vector database with ChromaDB for document storage  
- Interactive UI with Streamlit for easy querying and data management  

Tech Stack  
- Python (Primary Language)  
- Streamlit (Frontend UI)  
- LangChain (Document processing, LLMs, and retrieval)  
- ChromaDB (Vector database for document storage)  
- LLMs: OpenAI GPT models, Anthropic Claude models, and Hugging Face models  

Installation  

1. Clone the repository  
git clone https://github.com/your-username/multi-source-rag.git  
cd multi-source-rag  

2. Create a virtual environment (optional but recommended)  
python -m venv venv  
source venv/bin/activate  (On Windows: venv\Scripts\activate)  

3. Install dependencies  
pip install -r requirements.txt  

4. Set up API keys  
Create a `.env` file in the project root and add the necessary API keys:  
OPENAI_API_KEY=your_openai_api_key  
ANTHROPIC_API_KEY=your_anthropic_api_key  
HUGGINGFACE_API_KEY=your_huggingface_api_key  

5. Run the application  
streamlit run app.py  

Usage  

Sidebar Configuration  
- Select LLM Provider (OpenAI, Anthropic, or Hugging Face)  
- Enter the API Key for the chosen provider  
- Upload documents or provide URLs for data sources  
- Manage the vector database (add new documents, clear database)  

Asking Questions  
- Enter a question in the query box  
- Choose whether to show retrieved documents only or get an LLM-generated answer  
- Adjust the number of retrieved results  

File Structure  
multi-source-rag/  
- app.py (Main Streamlit application)  
- requirements.txt (Dependencies)  
- .env (API keys, not included in repo)  
- chroma_db/ (Vector database storage, auto-created)  
- README.md (Project documentation)  
- utils/ (Utility functions, if needed)  

Future Improvements  
- Add support for additional LLM providers like Mistral or Cohere  
- Implement fine-tuned models for better retrieval accuracy  
- Improve document chunking and metadata handling  

License  
This project is open-source under the MIT License.  

Contributions  
Pull requests are welcome! If you'd like to contribute, feel free to open an issue or submit a PR.  
