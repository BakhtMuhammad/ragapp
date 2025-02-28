# import streamlit as st
# from langchain.document_loaders import(
#     TextLoader,
#     PyPDFLoader,
#     YoutubeLoader,
#     UnstructuredURLLoader
# )
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import tempfile
# import os
# import shutil
# import validators

# #Importing LLM Related Modules
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import retrieval_qa
# from langchain.prompts import PromptTemplate


# #Define constant values used throughout the application
# PERSIST_DIRECTORY = 'chroma_db'
# COLLECTION_NAME = 'multi_source_docs' #Name of th collection in the Chroma DB

# #C1: Document Processing Class
# #Functions: There are four functions in this class for each file loader
# class DocumentProcessor:
#     def __init__(self):
#         #Initialize the text splitter with specific chunk size and overlap
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size = 1000,
#             chunk_overlap = 200
#         )
#     #F1: Text File Processing
#     def ProcessTextFile(self, file):
#         try:
#             #Createing a temporary file
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
#             try:
#                 temp_file.write(file.getbuffer()) #write the uploaded file to the temporary file
#                 temp_filepath = temp_file.name  #get path of the temp file
#             finally:
#                 temp_file.close() #close the temporary file to release resources
#             #Load and process the text file
#             loader = TextLoader(temp_filepath, encoding='utf-8')
#             docs = loader.load() #Load the document
#             os.unlink(temp_filepath) #Delete the temporary file
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None

#     #F2: For PDF File Processing
#     def ProcessPDFFile(self, file):
#         try:
#             #Createing a temporary file
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
#             try:
#                 temp_file.write(file.getbuffer()) #write the uploaded file to the temporary file
#                 temp_filepath = temp_file.name  #get path of the temp file
#             finally:
#                 temp_file.close() #close the temporary file to release resources

#             #Load and process the PDF file
#             loader = PyPDFLoader(temp_filepath)
#             docs = loader.load() #Load the document
#             os.unlink(temp_filepath) #Delete the temporary file
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
#     #F3: For Youtube File Processing
#     def ProcessYTFile(self, url):
#         try:
#             #Load Youtube Video Transcription
#             loader = YoutubeLoader.from_youtube_url(
#                 url,
#                 add_video_info = True, #Include video metadata
#                 language = ['en'] #English Transcription only
#             )
#             docs = loader.load() #Load the document
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
#     #F4: For Website File Processing
#     def ProcessWebsiteFile(self, url):
#         try:
#             #Load Youtube Video Transcription
#             loader = UnstructuredURLLoader(
#                 urls = [url]
#             )
#             docs = loader.load() #Load the document
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
    

# #C2: RAG App Class
# class RAGApp:
#     #Initialize the Embedding Model
#     def __init__(self):
#         self.embeddingFunction = SentenceTransformerEmbeddings(
#             model_name = 'all-MiniLM-L6-v2' #Embedding Model to Use
#         )
#         self.processor = DocumentProcessor() #Creating "DocumentProcessor" class instance
        
#     def InitializeDatabase(self):
#         #Create database directory if it doesn't exist already
#         if not os.path.exists(PERSIST_DIRECTORY):
#             #Make Directory/Folder
#             os.makedirs(PERSIST_DIRECTORY)
#         self.db = Chroma(
#             persist_directory = PERSIST_DIRECTORY, #chroma_db
#             embedding_function=self.embeddingFunction, #embedding model
#             collection_name=COLLECTION_NAME #multi_source_doc
#         )
#     def ClearDatabase(self):
#         try:
#             #Use chroma's methods to delete the collection
#             import chromadb
#             #Release the current database reference
#             if hasattr(self, 'db'):
#                 del self.db

#                 #Force garbage collection
#                 import gc
#                 gc.collect()

#                 #Create a direct client to ChromaDB
#                 client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

#                 #Delete the collection if it exists
#                 try:
#                     client.delete_collection(COLLECTION_NAME)
#                 except Exception:
#                     #Collection might not exist, which is fine
#                     pass
#                 #Reinitialize our database
#                 self.InitializeDatabase()

#                 st.success('Database Cleared Successfully.')
#         except Exception as e:
#             st.error(f'Error Clearing Database: {str(e)}')

#     def Search(self, query, num_results = 4):
#         """
#         Search for relevant documents based on a query using Chroma similarity search
#         """
#         try:
#             #Perform Similarity Search
#             results = self.db.similarity_search(query, k = num_results)
#             return results
#         except Exception as e:
#             st.error(f'Error during search: {str(e)}')
#             return None
        
# def ProcessAndAddToDB(docs, app):
#     """
#     Adds processed documents to the Chroma database
#     """
#     try:
#         #Ensure docs are added to the database
#         if docs:
#             app.db.add_documents(docs)
#             st.success(f'{len(docs)} document chunks added to the database.')
#         else:
#             st.error('No document to add to the database.')
#     except Exception as e:
#         st.error(f'Error adding documents to the database: {str(e)}')



# def main():
#     st.title('Multi-Source RAG App!')
#     #Initialize the RAG App class
#     app = RAGApp()
#     app.InitializeDatabase()  # Ensure the database is initialized
#     #Left side panel for source selection and database statistics
#     st.sidebar.title('Data Sources')
#     sourceType = st.sidebar.selectbox(
#         'Select Source Type:',
#         ['Text File', 'PDF Document', 'Youtube Video', 'Website']
#     )
    
#     #Main content area for the data input
#     if sourceType == 'Text File':
#         file = st.file_uploader('Upload a text file: ', type=['txt'])
#         if file:
#             with st.spinner('Processing...'):
#                 docs = app.processor.ProcessTextFile(file)
#                 ProcessAndAddToDB(docs, app)
#     elif sourceType == 'PDF Document':
#         file = st.file_uploader('Upload a PDF file: ', type=['pdf'])
#         if file:
#             docs = app.processor.ProcessPDFFile(file)
#             ProcessAndAddToDB(docs, app)
#     elif sourceType == 'Youtube Video':
#         url = st.text_input('Enter Youtube URL: ')
#         if url and st.button('Process Video'):
#             if 'youtube.com' in url or 'youtu.be' in url:
#                  docs = app.processor.ProcessYTFile(url)
#                  ProcessAndAddToDB(docs, app)
#             else: 
#                 st.error('Please enter a valid youtube link.')
#     elif sourceType == 'Website':
#         url = st.text_input('Enter Website URL: ')
#         if url and st.button('Process Website'):
#             if validators.url(url):
#                  docs = app.processor.ProcessWebsiteFile(url)
#                  ProcessAndAddToDB(docs, app)
#             else: 
#                 st.error('Please enter a valid website link.')

#     #Database Management
#     st.sidebar.subheader('Database Management')
#     if st.sidebar.button('Clear Database'):
#         app.ClearDatabase()

#     #Display Database Statistics
#     try:
#         collection = app.db.get()
#         st.sidebar.subheader('Database Statistics')
#         st.sidebar.write(f'Number of Document: {len(collection.get('documents',[]))}')
#     except Exception as e:
#         st.sidebar.error('Error Loading the Database Statistics.')
#     #Search Interface
#     st.subheader('Search')
#     query = st.text_input('Enter Your Search Query')
#     numResults = st.slider('Number of results', min_value=1, max_value=10, value=4)

#     if query:
#         results = app.Search(query, numResults)
#         if results:
#             st.subheader('Search Results')
#             for i,doc in enumerate(results):
#                 with st.expander(f'Result {i+1}'):
#                     #Display source information if available
#                     if hasattr(doc.metadata, 'source'):
#                         st.write(f'Source: {doc.metadata.get('source', 'Unknown')}')
#                     if hasattr(doc.metadata, 'page'):
#                         st.write(f'Page: {doc.metadata.get('page', 'Unknown')}')
#                     st.write(doc.page_content)


# main()
########################################   VERSION 2   #####################################################
# import streamlit as st
# from langchain.document_loaders import(
#     TextLoader,
#     PyPDFLoader,
#     YoutubeLoader,
#     UnstructuredURLLoader
# )
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import tempfile
# import os
# import shutil
# import validators

# #Importing LLM Related Modules
# from langchain.chat_models import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# #Load environment variables (for API keys)
# load_dotenv()

# #Define constant values used throughout the application
# PERSIST_DIRECTORY = 'chroma_db'
# COLLECTION_NAME = 'multi_source_docs' #Name of th collection in the Chroma DB

# #C1: Document Processing Class
# #Functions: There are four functions in this class for each file loader
# class DocumentProcessor:
#     def __init__(self):
#         #Initialize the text splitter with specific chunk size and overlap
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size = 1000,
#             chunk_overlap = 200
#         )
#     #F1: Text File Processing
#     def ProcessTextFile(self, file):
#         try:
#             #Createing a temporary file
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
#             try:
#                 temp_file.write(file.getbuffer()) #write the uploaded file to the temporary file
#                 temp_filepath = temp_file.name  #get path of the temp file
#             finally:
#                 temp_file.close() #close the temporary file to release resources
#             #Load and process the text file
#             loader = TextLoader(temp_filepath, encoding='utf-8')
#             docs = loader.load() #Load the document
#             os.unlink(temp_filepath) #Delete the temporary file
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None

#     #F2: For PDF File Processing
#     def ProcessPDFFile(self, file):
#         try:
#             #Createing a temporary file
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
#             try:
#                 temp_file.write(file.getbuffer()) #write the uploaded file to the temporary file
#                 temp_filepath = temp_file.name  #get path of the temp file
#             finally:
#                 temp_file.close() #close the temporary file to release resources

#             #Load and process the PDF file
#             loader = PyPDFLoader(temp_filepath)
#             docs = loader.load() #Load the document
#             os.unlink(temp_filepath) #Delete the temporary file
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
#     #F3: For Youtube File Processing
#     def ProcessYTFile(self, url):
#         try:
#             #Load Youtube Video Transcription
#             loader = YoutubeLoader.from_youtube_url(
#                 url,
#                 add_video_info = True, #Include video metadata
#                 language = ['en'] #English Transcription only
#             )
#             docs = loader.load() #Load the document
#             return self.text_splitter.split_documents(docs)     #split into chunks
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
#     #F4: For Website File Processing
#     def ProcessWebsiteFile(self, url):
#         try:
#             #Load Youtube Video Transcription
#             loader = UnstructuredURLLoader(
#                 urls = [url]# import streamlit as st
# from langchain.document_loaders import(
#     TextLoader,
#     PyPDFLoader,
#     YoutubeLoader,
#     UnstructuredURLLoader
# )
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import tempfile
# import os
# import shutil
# import validators

# # Define constant values used throughout the application
# PERSIST_DIRECTORY = 'chroma_db'
# COLLECTION_NAME = 'multi_source_docs' # Name of the collection in the Chroma DB

# # C1: Document Processing Class
# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
    
#     def ProcessTextFile(self, file):
#         try:
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
#             try:
#                 temp_file.write(file.getbuffer())
#                 temp_filepath = temp_file.name
#             finally:
#                 temp_file.close()
#             loader = TextLoader(temp_filepath, encoding='utf-8')
#             docs = loader.load()
#             os.unlink(temp_filepath)
#             return self.text_splitter.split_documents(docs)
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
    
#     def ProcessPDFFile(self, file):
#         try:
#             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
#             try:
#                 temp_file.write(file.getbuffer())
#                 temp_filepath = temp_file.name
#             finally:
#                 temp_file.close()
#             loader = PyPDFLoader(temp_filepath)
#             docs = loader.load()
#             os.unlink(temp_filepath)
#             return self.text_splitter.split_documents(docs)
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
    
#     def ProcessYTFile(self, url):
#         try:
#             loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=['en'])
#             docs = loader.load()
#             return self.text_splitter.split_documents(docs)
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None
    
#     def ProcessWebsiteFile(self, url):
#         try:
#             loader = UnstructuredURLLoader(urls=[url])
#             docs = loader.load()
#             return self.text_splitter.split_documents(docs)
#         except Exception as e:
#             st.error(f'Error processing the file: {str(e)}')
#             return None

# # C2: RAG App Class
# class RAGApp:
#     def __init__(self):
#         self.embeddingFunction = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
#         self.processor = DocumentProcessor()
    
#     def InitializeDatabase(self):
#         if not os.path.exists(PERSIST_DIRECTORY):
#             os.makedirs(PERSIST_DIRECTORY)
#         self.db = Chroma(
#             persist_directory=PERSIST_DIRECTORY,
#             embedding_function=self.embeddingFunction,
#             collection_name=COLLECTION_NAME
#         )
    
#     # def ClearDatabase(self):
#     #     try:
#     #         # Create a new collection name instead of deleting the old one
#     #         global COLLECTION_NAME
#     #         import uuid
#     #         COLLECTION_NAME = f"multi_source_docs_{uuid.uuid4().hex[:8]}"
        
#     #         # Initialize a new database with the new collection name
#     #         self.InitializeDatabase()
#     #         st.success('Database Cleared Successfully (New collection created).')
#     #     except Exception as e:
#     #         st.error(f'Error Clearing Database: {str(e)}')

#     def ClearDatabase(self):
#         try:
#             # Use Chroma's methods to delete the collection
#             import chromadb
        
#         # Release the current database reference
#             if hasattr(self, 'db'):
#                 del self.db
        
#             # Force garbage collection
#             import gc
#             gc.collect()
        
#             # Create a direct client to ChromaDB
#             client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
#             # Delete the collection if it exists
#             try:
#                 client.delete_collection(COLLECTION_NAME)
#             except Exception:
#                 # Collection might not exist, which is fine
#                 pass
        
#             # Reinitialize our database
#             self.InitializeDatabase()
        
#             st.success('Database Cleared Successfully.')
#         except Exception as e:
#             st.error(f'Error Clearing Database: {str(e)}')


    
#     def Search(self, query, num_results=4):
#         try:
#             results = self.db.similarity_search(query, k=num_results)
#             return results
#         except Exception as e:
#             st.error(f'Error during search: {str(e)}')
#             return None

# def ProcessAndAddToDB(docs, app):
#     try:
#         if docs:
#             app.db.add_documents(docs)
#             st.success(f'{len(docs)} document chunks added to the database.')
#         else:
#             st.error('No document to add to the database.')
#     except Exception as e:
#         st.error(f'Error adding documents to the database: {str(e)}')

# def main():
#     st.title('Multi-Source RAG App!')
#     app = RAGApp()
#     app.InitializeDatabase()
    
#     st.sidebar.title('Data Sources')
#     sourceType = st.sidebar.selectbox(
#         'Select Source Type:',
#         ['Text File', 'PDF Document', 'Youtube Video', 'Website']
#     )
    
#     if sourceType == 'Text File':
#         file = st.file_uploader('Upload a text file:', type=['txt'])
#         if file:
#             with st.spinner('Processing...'):
#                 docs = app.processor.ProcessTextFile(file)
#                 ProcessAndAddToDB(docs, app)
#     elif sourceType == 'PDF Document':
#         file = st.file_uploader('Upload a PDF file:', type=['pdf'])
#         if file:
#             docs = app.processor.ProcessPDFFile(file)
#             ProcessAndAddToDB(docs, app)
#     elif sourceType == 'Youtube Video':
#         url = st.text_input('Enter Youtube URL:')
#         if url and st.button('Process Video'):
#             if 'youtube.com' in url or 'youtu.be' in url:
#                 docs = app.processor.ProcessYTFile(url)
#                 ProcessAndAddToDB(docs, app)
#             else:
#                 st.error('Please enter a valid YouTube link.')
#     elif sourceType == 'Website':
#         url = st.text_input('Enter Website URL:')
#         if url and st.button('Process Website'):
#             if validators.url(url):
#                 docs = app.processor.ProcessWebsiteFile(url)
#                 ProcessAndAddToDB(docs, app)
#             else:
#                 st.error('Please enter a valid website link.')
    
#     st.sidebar.subheader('Database Management')
#     if st.sidebar.button('Clear Database'):
#         app.ClearDatabase()
    
#     try:
#         collection = app.db.get()
#         st.sidebar.subheader('Database Statistics')
#         st.sidebar.write(f'Number of Documents: {len(collection.get("documents", []))}')
#     except Exception as e:
#         st.sidebar.error('Error Loading the Database Statistics.')
    
#     st.subheader('Search')
#     query = st.text_input('Enter Your Search Query')
#     numResults = st.slider('Number of results', min_value=1, max_value=10, value=4)
    
#     if query:
#         results = app.Search(query, numResults)
#         if results:
#             st.subheader('Search Results')
#             for i, doc in enumerate(results):
#                 with st.expander(f'Result {i+1}'):
#                     st.write(doc.page_content)

# main()
##############################################################################
import streamlit as st
from langchain.document_loaders import(
    TextLoader,
    PyPDFLoader,
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import shutil
import validators

# Import LLM-related modules with updated imports
from langchain_openai import ChatOpenAI
###Debug
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
###Debug
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Define constant values used throughout the application
PERSIST_DIRECTORY = 'chroma_db'
COLLECTION_NAME = 'multi_source_docs' # Name of the collection in the Chroma DB

# C1: Document Processing Class
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def ProcessTextFile(self, file):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            try:
                temp_file.write(file.getbuffer())
                temp_filepath = temp_file.name
            finally:
                temp_file.close()
            loader = TextLoader(temp_filepath, encoding='utf-8')
            docs = loader.load()
            os.unlink(temp_filepath)
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            st.error(f'Error processing the file: {str(e)}')
            return None
    
    def ProcessPDFFile(self, file):
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            try:
                temp_file.write(file.getbuffer())
                temp_filepath = temp_file.name
            finally:
                temp_file.close()
            loader = PyPDFLoader(temp_filepath)
            docs = loader.load()
            os.unlink(temp_filepath)
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            st.error(f'Error processing the file: {str(e)}')
            return None
    
    def ProcessYTFile(self, url):
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=['en'])
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            st.error(f'Error processing the file: {str(e)}')
            return None
    
    def ProcessWebsiteFile(self, url):
        try:
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            return self.text_splitter.split_documents(docs)
        except Exception as e:
            st.error(f'Error processing the file: {str(e)}')
            return None

# C2: RAG App Class
class RAGApp:
    def __init__(self):
        self.embeddingFunction = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        self.processor = DocumentProcessor()
        ###Debug
        self.qa_chain = None
    
    def InitializeDatabase(self):
        if not os.path.exists(PERSIST_DIRECTORY):
            os.makedirs(PERSIST_DIRECTORY)
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddingFunction,
            collection_name=COLLECTION_NAME
        )
        # Initialize QA chain if we have an API key
        self.InitializeQAChain()
    
    def InitializeLLM(self):
        """Initialize the LLM based on selected provider with fixed parameter names"""
        llm_provider = st.session_state.get('llm_provider', 'OpenAI')
        
        if llm_provider == 'OpenAI':
            # Check if API key is in environment or session state
            api_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('openai_api_key')
            if not api_key:
                st.warning("OpenAI API key not found. Please enter it in the sidebar.")
                return None
            
            model_name = st.session_state.get('openai_model', 'gpt-3.5-turbo')
            try:
                # Updated initialization with correct parameter names
                return ChatOpenAI(
                    temperature=0.3,
                    ###Debug
                    model_name=model_name,
                    api_key=api_key
                )
            except Exception as e:
                st.error(f"Error initializing OpenAI: {str(e)}")
                st.info("Trying alternative initialization...")
                # Fallback to alternative parameter names if needed
                return ChatOpenAI(
                    temperature=0.3,
                    ###Debug
                    model=model_name,
                    openai_api_key=api_key
                )
            
        elif llm_provider == 'Anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY') or st.session_state.get('anthropic_api_key')
            if not api_key:
                st.warning("Anthropic API key not found. Please enter it in the sidebar.")
                return None
                
            model_name = st.session_state.get('anthropic_model', 'claude-3-opus-20240229')
            try:
                # Updated initialization with correct parameter names
                return ChatAnthropic(
                    temperature=0.3,
                    model_name=model_name,
                    api_key=api_key
                )
            except Exception as e:
                st.error(f"Error initializing Anthropic: {str(e)}")
                st.info("Trying alternative initialization...")
                # Fallback to alternative parameter names if needed
                return ChatAnthropic(
                    temperature=0.3,
                    model=model_name,
                    anthropic_api_key=api_key
                )
            
        elif llm_provider == 'HuggingFace':
            # Using Hugging Face inference API
            api_key = os.getenv('HUGGINGFACE_API_KEY') or st.session_state.get('huggingface_api_key')
            if not api_key:
                st.warning("HuggingFace API key not found. Please enter it in the sidebar.")
                return None
            
            try:
                # Try importing the updated version
                from langchain_huggingface import HuggingFaceHub
            except ImportError:
                # Fallback to legacy import
                from langchain.llms import HuggingFaceHub
                
            model_name = st.session_state.get('huggingface_model', 'google/flan-t5-xxl')
            try:
                return HuggingFaceHub(
                    repo_id=model_name,
                    huggingfacehub_api_token=api_key
                )
            except Exception as e:
                st.error(f"Error initializing HuggingFace: {str(e)}")
                # Try alternative parameter name
                return HuggingFaceHub(
                    repo_id=model_name,
                    api_key=api_key
                )
        
        return None
    
    def InitializeQAChain(self):
        """Initialize the QA chain with the LLM and retriever"""
        # Get LLM from selected provider
        llm = self.InitializeLLM()
        
        if llm is None:
            self.qa_chain = None
            return
            
        # Create retriever from the vector store
        try:
            retriever = self.db.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            st.error(f"Error creating retriever: {str(e)}")
            # Try alternative method
            retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Custom prompt template
        template = """
        You are a helpful assistant that answers questions based on provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer based only on the context provided. 
        If the context doesn't contain relevant information to answer the question, 
        just say "I don't have enough information to answer this question."
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            ###Debug
        except Exception as e:
            st.error(f"Error creating QA chain: {str(e)}")
            # Try alternative with fewer parameters
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )

    def ClearDatabase(self):
        try:
            # Use Chroma's methods to delete the collection
            import chromadb
        
            # Release the current database reference
            if hasattr(self, 'db'):
                del self.db
        
            # Force garbage collection
            import gc
            gc.collect()
        
            # Create a direct client to ChromaDB
            client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        
            # Delete the collection if it exists
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                # Collection might not exist, which is fine
                pass
        
            # Reinitialize our database
            self.InitializeDatabase()
        
            st.success('Database Cleared Successfully.')
        except Exception as e:
            st.error(f'Error Clearing Database: {str(e)}')
    
    def Search(self, query, num_results=4):
        try:
            results = self.db.similarity_search(query, k=num_results)
            return results
        except Exception as e:
            st.error(f'Error during search: {str(e)}')
            return None
            
    def AnswerQuestion(self, query):
        """Generate an answer using the QA chain"""
        if not self.qa_chain:
            st.error("QA chain not initialized. Please check your API key.")
            return None, []
            
        try:
            ###Debug
            # Updated to handle both return formats
            result = self.qa_chain({"query": query})
            if "result" in result:
                return result["result"], result.get("source_documents", [])
            elif "answer" in result:
                return result["answer"], result.get("source_documents", [])
            else:
                st.error("Unexpected response format from QA chain")
                return str(result), []
        except Exception as e:
            st.error(f'Error generating answer: {str(e)}')
            return None, []

def ProcessAndAddToDB(docs, app):
    try:
        if docs:
            app.db.add_documents(docs)
            st.success(f'{len(docs)} document chunks added to the database.')
        else:
            st.error('No document to add to the database.')
    except Exception as e:
        st.error(f'Error adding documents to the database: {str(e)}')

def initialize_session_state():
    """Initialize session state variables"""
    if 'llm_provider' not in st.session_state:
        st.session_state['llm_provider'] = 'OpenAI'
    if 'openai_model' not in st.session_state:
        st.session_state['openai_model'] = 'gpt-3.5-turbo'
    if 'anthropic_model' not in st.session_state:
        st.session_state['anthropic_model'] = 'claude-3-opus-20240229'
    if 'huggingface_model' not in st.session_state:
        st.session_state['huggingface_model'] = 'google/flan-t5-xxl'

def main():
    st.title('Multi-Source RAG App!')
    
    # Initialize session state
    initialize_session_state()
    
    app = RAGApp()
    app.InitializeDatabase()
    ###Debug
    # ===== SIDEBAR =====
    st.sidebar.title('Configuration')
    
    # LLM Provider Selection
    st.sidebar.subheader('LLM Provider')
    llm_provider = st.sidebar.selectbox(
        'Select LLM Provider:',
        ['OpenAI', 'Anthropic', 'HuggingFace'],
        index=0,
        key='llm_provider'
    )
    
    # Provider-specific settings
    if llm_provider == 'OpenAI':
        api_key = st.sidebar.text_input('OpenAI API Key:', type='password', key='openai_api_key')
        model = st.sidebar.selectbox(
            'Select Model:',
            ['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4-1106-preview'],
            index=0,
            key='openai_model'
        )
    elif llm_provider == 'Anthropic':
        api_key = st.sidebar.text_input('Anthropic API Key:', type='password', key='anthropic_api_key')
        model = st.sidebar.selectbox(
            'Select Model:',
            ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            index=0,
            key='anthropic_model'
        )
    elif llm_provider == 'HuggingFace':
        api_key = st.sidebar.text_input('HuggingFace API Key:', type='password', key='huggingface_api_key')
        model = st.sidebar.selectbox(
            'Select Model:',
            ['google/flan-t5-xxl', 'tiiuae/falcon-7b-instruct', 'meta-llama/Llama-2-7b-chat-hf'],
            index=0,
            key='huggingface_model'
        )
    
    # Reinitialize QA chain if settings changed
    if st.sidebar.button('Apply LLM Settings'):
        app.InitializeQAChain()
        st.sidebar.success('LLM settings applied!')
    
    # Data Sources Section
    st.sidebar.title('Data Sources')
    sourceType = st.sidebar.selectbox(
        'Select Source Type:',
        ['Text File', 'PDF Document', 'Youtube Video', 'Website']
    )
    
    if sourceType == 'Text File':
        file = st.sidebar.file_uploader('Upload a text file:', type=['txt'])
        if file:
            with st.spinner('Processing...'):
                docs = app.processor.ProcessTextFile(file)
                ProcessAndAddToDB(docs, app)
    elif sourceType == 'PDF Document':
        file = st.sidebar.file_uploader('Upload a PDF file:', type=['pdf'])
        if file:
            with st.spinner('Processing...'):
                docs = app.processor.ProcessPDFFile(file)
                ProcessAndAddToDB(docs, app)
    elif sourceType == 'Youtube Video':
        url = st.sidebar.text_input('Enter Youtube URL:')
        if url and st.sidebar.button('Process Video'):
            if 'youtube.com' in url or 'youtu.be' in url:
                with st.spinner('Processing YouTube video...'):
                    docs = app.processor.ProcessYTFile(url)
                    ProcessAndAddToDB(docs, app)
            else:
                st.sidebar.error('Please enter a valid YouTube link.')
    elif sourceType == 'Website':
        url = st.sidebar.text_input('Enter Website URL:')
        if url and st.sidebar.button('Process Website'):
            if validators.url(url):
                with st.spinner('Processing website...'):
                    docs = app.processor.ProcessWebsiteFile(url)
                    ProcessAndAddToDB(docs, app)
            else:
                st.sidebar.error('Please enter a valid website link.')
    
    st.sidebar.subheader('Database Management')
    if st.sidebar.button('Clear Database'):
        app.ClearDatabase()
    
    try:
        collection = app.db.get()
        st.sidebar.subheader('Database Statistics')
        st.sidebar.write(f'Number of Documents: {len(collection.get("documents", []))}')
    except Exception as e:
        st.sidebar.error(f'Error Loading the Database Statistics: {str(e)}')
    
    # ===== MAIN CONTENT =====
    # Query interface
    st.subheader('Ask Questions')
    
    query = st.text_input('Enter Your Question:')
    col1, col2 = st.columns(2)
    
    with col1:
        search_only = st.checkbox('Show retrieved documents only', value=False)
    
    with col2:
        numResults = st.slider('Number of results', min_value=1, max_value=10, value=4)
    
    if query:
        # Get retrieved documents
        results = app.Search(query, numResults)
        
        if search_only:
            # Just show the retrieved documents
            if results:
                st.subheader('Retrieved Documents')
                for i, doc in enumerate(results):
                    with st.expander(f'Document {i+1}'):
                        st.write(doc.page_content)
        else:
            # Generate an answer using the LLM
            with st.spinner('Generating answer...'):
                answer, source_docs = app.AnswerQuestion(query)
                
                if answer:
                    st.subheader('Answer')
                    st.markdown(answer)
                    
                    # Show sources
                    st.subheader('Sources')
                    for i, doc in enumerate(source_docs):
                        with st.expander(f'Source {i+1}'):
                            st.write(doc.page_content)

if __name__ == "__main__":
    main()
