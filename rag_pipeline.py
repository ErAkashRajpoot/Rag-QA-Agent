import torch
from langchain_huggingface import HuggingFacePipeline
from vectorstore import build_faiss_index
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate


class AgenticRAG:
    def __init__(self, data, chunk_size=500, chunk_overlap=100):
        """
        Initialize the Agentic RAG pipeline.

        Args:
            data: Either a folder path (loads all PDFs/TXT) or a list of text strings.
            chunk_size: Size of text chunks for splitting.
            chunk_overlap: Overlap between chunks.
        """
        # Step 1: Build FAISS retriever
        self.vectorstore = build_faiss_index(data, chunk_size, chunk_overlap)

        # Step 2: Load FLAN-T5 Large model (780M params, FP16 for GPU)
        self.model_name = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Use GPU with FP16 if available, otherwise CPU
        if torch.cuda.is_available():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto"
            )
            print(f"Model loaded on GPU ({torch.cuda.get_device_name(0)}) in FP16")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print("Model loaded on CPU in FP32")

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
        )
        self.llm = HuggingFacePipeline(pipeline=generator)

        # Step 3: Few-shot prompt for extraction-style QA
        template = """Read the context and answer the question. If the context does not contain the answer, output "I don't know".

        Context: Word2Vec is a neural embedding technique that represents words in a dense vector space. It includes two main architectures: Continuous Bag of Words (CBOW) and Skip-Gram.
        Question: What are the two main architectures of Word2Vec?
        Answer: Continuous Bag of Words (CBOW) and Skip-Gram.

        Context: {context}
        Question: {question}
        Answer:"""
        self.prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    def _get_context(self, docs, question, max_input_tokens=450):
        """
        Build context from retrieved docs, ensuring total input fits within
        FLAN-T5's 512 token limit.
        """
        # Calculate how many tokens the prompt + question use
        shell = self.prompt.format(question=question, context="")
        shell_tokens = len(self.tokenizer.encode(shell))
        budget = max_input_tokens - shell_tokens

        # Add doc chunks one by one until we hit the budget
        context_parts = []
        used = 0
        for doc in docs:
            chunk_tokens = len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
            if used + chunk_tokens > budget:
                # Add partial chunk if we have space
                remaining = budget - used
                if remaining > 20:
                    tokens = self.tokenizer.encode(doc.page_content, add_special_tokens=False)[:remaining]
                    context_parts.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
                break
            context_parts.append(doc.page_content)
            used += chunk_tokens

        return "\n\n".join(context_parts)

    def query(self, question, k=3):
        """
        Query the RAG pipeline.

        Args:
            question: The question to answer.
            k: Number of relevant documents to retrieve.
        """
        # 1. Retrieve relevant docs
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        docs = retriever.invoke(question)

        # 2. Build context that fits within token limit
        context = self._get_context(docs, question)

        # 3. Format the prompt and generate
        prompt_text = self.prompt.format(question=question, context=context)
        answer = self.llm.invoke(prompt_text)

        return answer, docs
