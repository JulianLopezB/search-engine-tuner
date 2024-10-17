import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
import logging
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import fasttext
from langchain_core.messages import HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

class FastTextEmbeddings:
    def __init__(self, model_path):
        try:
            self.model = fasttext.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load FastText model from {model_path}: {str(e)}")
            raise ValueError(f"Failed to load FastText model: {str(e)}")

    def embed_query(self, text):
        return self.model.get_sentence_vector(text).tolist()

class SearchService:
    def __init__(self):
        logger.info("Initializing SearchService")
        self.client = QdrantClient(path=os.getenv("VECTOR_DB_PATH"))
        
        # Initialize OpenAI embeddings
        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize OpenAI large embeddings
        self.openai_large_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize FastText embeddings
        fasttext_model_path = os.getenv("FASTTEXT_MODEL_PATH")
        if not fasttext_model_path:
            raise ValueError("FASTTEXT_MODEL_PATH must be set for FastText embeddings")
        self.fasttext_embeddings = FastTextEmbeddings(fasttext_model_path)
        
        # Initialize ChatOpenAI LLM
        self.llm = ChatOpenAI(
            model_name=os.getenv("OPENAI_LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _get_collection_name(self, embedding_type: str):
        if embedding_type == "openai":
            return "articles"
        elif embedding_type == "openai-large":
            return "articles_openai"
        elif embedding_type == "fasttext":
            return "articles_fasttext"
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def _get_embeddings(self, embedding_type: str):
        if embedding_type == "openai":
            return self.openai_embeddings
        elif embedding_type == "openai-large":
            return self.openai_large_embeddings
        elif embedding_type == "fasttext":
            return self.fasttext_embeddings
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def search(self, query: str, category: str = None, limit: int = 15, embedding_type: str = "openai", threshold: float = 0):
        logger.info(f"Performing search - query: {query}, category: {category}, limit: {limit}, embedding_type: {embedding_type}, threshold: {threshold}")
        collection_name = self._get_collection_name(embedding_type)
        embeddings = self._get_embeddings(embedding_type)
        
        filter_conditions = None
        if category:
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="metadata.grupo",
                        match=MatchValue(value=int(category))
                    )
                ]
            )
        
        query_vector = embeddings.embed_query(query)
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter_conditions,
            limit=limit * 2,  # Increase the limit to account for filtering
            search_params=models.SearchParams(hnsw_ef=int(os.getenv("HNSW_EF", 128)), exact=False),
        )
        
        # Filter results based on the threshold
        filtered_results = [result for result in search_result if result.score >= threshold]
        
        # Limit the results after filtering
        limited_results = filtered_results[:limit]
        
        logger.info(f"Search completed, found {len(search_result)} results, {len(filtered_results)} above threshold, returning {len(limited_results)}")
        return limited_results

    def get_article(self, article_id: int, embedding_type: str = "openai"):
        logger.info(f"Fetching article with id: {article_id}, embedding_type: {embedding_type}")
        collection_name = self._get_collection_name(embedding_type)
        search_result = self.client.retrieve(
            collection_name=collection_name,
            ids=[article_id]
        )
        if not search_result:
            logger.warning(f"Article not found: {article_id}")
            return None
        logger.info(f"Article found: {article_id}")
        return search_result[0]

    def get_categories(self):
        logger.info("Fetching categories")
        groups = self.client.scroll(
            collection_name=self._get_collection_name('fasttext'),
            # scroll_filter=Filter(),
            limit=int(os.getenv("CATEGORY_SCROLL_LIMIT", 10000)),
            with_payload=True,
            with_vectors=False
        )[0]
        
        unique_groups = set(point.payload.get("metadata", {}).get("grupo") for point in groups if "grupo" in point.payload.get("metadata", {}))
        logger.info(f"Retrieved {len(unique_groups)} categories")
        return sorted(list(unique_groups))

    def _get_embedding(self, text: str):
        if isinstance(self.embeddings, OpenAIEmbeddings):
            return self.embeddings.embed_query(text)
        elif isinstance(self.embeddings, FastTextEmbeddings):
            return self.embeddings.embed_query(text)
        else:
            raise ValueError(f"Unsupported embedding type: {type(self.embeddings)}")

    def rag_search(self, query: str, category: str = None):
        logger.info(f"Performing RAG search - query: {query}, category: {category}")
        
        # Prepare the prompt
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Update the qa_chain with the new prompt
        self.qa_chain.combine_documents_chain.llm_chain.prompt = PROMPT
        
        # Perform the RAG search
        result = self.qa_chain({"query": query})
        
        # Extract the answer and source documents
        answer = result['result']
        source_docs = result['source_documents']
        
        # Format the response
        response = {
            "answer": answer,
            "sources": []
        }
        
        for doc in source_docs:
            source = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (int, str)):
                    source[key] = value
                elif isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        source[key] = str(value)
                    else:
                        source[key] = value
                else:
                    source[key] = str(value)
            response["sources"].append(source)
        
        logger.info(f"RAG search completed, answer generated with {len(source_docs)} source documents")
        return response

    def search_with_ai_validation(self, query: str, category: str = None, limit: int = 15, threshold: float = 0.0, embedding_type: str = "openai"):
        logger.info(f"Performing AI-validated search - query: {query}, category: {category}, limit: {limit}, threshold: {threshold}, embedding_type: {embedding_type}")
        
        # Perform the initial search
        search_results = self.search(query, category, limit, embedding_type, threshold)

        validated_results = []
        for result in search_results:
            # Extract relevant information from the search result
            article_text = result.payload.get('metadata', {}).get('respuesta', '')
            article_title = result.payload.get('metadata', {}).get('pregunta', '')

            # Prepare the prompt for the LLM
            prompt = f"Query: {query}\n\nArticle Title: {article_title}\n\nArticle Text: {article_text}\n\nIs this article relevant to the query? Answer with 'Yes' or 'No' and a brief explanation."
            
            # Create a HumanMessage object
            message = HumanMessage(content=prompt)

            # Get the LLM's response
            llm_response = self.llm([message])
            llm_answer = llm_response.content

            # Check if the LLM considers the article relevant
            if llm_answer.lower().startswith('yes'):
                validated_results.append({'result': result, 'reason': llm_answer})

        return validated_results