import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.vectorstore import get_faiss_index

logger = setup_logger(__name__)


class AgenticRAG:
    """Wrapper class containing RAG generation logic and LLM context loading."""
    def __init__(self):
        logger.info(f"Initializing Agentic RAG Pipeline with top-to-bottom instantiation.")

        # Step 1: Pre-initialize and cache the FAISS Retriever
        try:
            self.vectorstore = get_faiss_index()
            logger.info("Successfully bound FAISS Datastore.")
        except Exception as e:
            logger.error(f"Failed to bind FAISS DataStore. System cannot query properly. Root Cause: {e}")
            self.vectorstore = None

        # Step 2: Load configured Model
        self.model_name = settings.MODEL_NAME
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Loaded tokenizer mappings for '{self.model_name}'")

            # Determine precision scaling based on GPU runtime constraints
            if torch.cuda.is_available():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16,
                    device_map="auto"
                )
                logger.info(f"Model `{self.model_name}` mapped to FP16 GPU context ({torch.cuda.get_device_name(0)})")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                logger.warning(f"No GPU Detected. Model `{self.model_name}` mapped to CPU FP32. Expect sub-optimal inference times.")

            generator = pipeline(
                "text-generation",  # Modern transformers auto-detects seq2seq vs causal
                model=model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
            )
            self.llm = HuggingFacePipeline(pipeline=generator)
            logger.info("HuggingFace Inference Pipeline initialized successfully.")

        except Exception as e:
            logger.critical(f"Failed to fetch HuggingFace model `{self.model_name}`: {e}")
            raise e

        # Step 3: Hardened instruction boundaries & Prompt Template
        template = """Read the exact context provided and concisely respond to the user's specific question. If the requested information is absent from the provided context, gracefully admit "I don't know".

        Context: {context}
        Question: {question}
        Answer:"""
        self.prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    def _get_context(self, docs, question: str, max_input_tokens: int = 400) -> str:
        """Dynamically construct input context stopping when internal LLM FLAN-T5 constraints are hit."""
        shell = self.prompt.format(question=question, context="")
        shell_tokens = len(self.tokenizer.encode(shell))
        remaining_budget = max_input_tokens - shell_tokens

        context_blocks = []
        used = 0
        
        for doc in docs:
            chunk_tokens = len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
            if used + chunk_tokens > remaining_budget:
                remaining = remaining_budget - used
                if remaining > 25:
                    tokens = self.tokenizer.encode(doc.page_content, add_special_tokens=False)[:remaining]
                    context_blocks.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
                break
            context_blocks.append(doc.page_content)
            used += chunk_tokens

        logger.debug(f"Assembled dynamic context size: {used} input tokens from {len(docs)} segments.")
        return "\n\n".join(context_blocks)

    def query(self, question: str, k: int = 3):
        """Invoke extraction-style Q&A context generation against local knowledge."""
        logger.info(f"Incoming external query received: '{question}' (K={k})")
        
        if not self.vectorstore:
            raise ValueError("FAISS Database failed to load. Ensure 'data/' is populated.")

        # 1. FAISS Embedded Lookup
        logger.debug("Sinking into FAISS search nodes...")
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(question)

        # 2. Dynamic Token Packing
        context = self._get_context(docs, question)

        # 3. Model Request
        prompt_text = self.prompt.format(question=question, context=context)
        logger.debug("Prompt synthesized. Injecting into GPU Pipeline execution thread.")
        
        answer = self.llm.invoke(prompt_text)
        
        return answer, docs
