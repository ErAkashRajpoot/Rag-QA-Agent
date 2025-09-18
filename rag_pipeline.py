from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from vectorstore import build_faiss_index
from transformers import pipeline
from langchain.prompts import PromptTemplate

class AgenticRAG:
    def __init__(self, docs):
        # Step 1: FAISS retriever
        self.vectorstore = build_faiss_index(docs)

        # Step 2: Load local FLAN-T5 model
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )
        self.llm = HuggingFacePipeline(pipeline=generator)

        # Step 3: Define an "agentic" prompt
        template = """
        You are an intelligent research assistant.
        Follow these steps:
        1. Analyze the question carefully.
        2. Retrieve and summarize the most relevant knowledge from context.
        3. Generate a clear, precise final answer.
        
        Question: {question}
        Context: {context}
        
        Step-by-step reasoning + Final Answer:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    def query(self, question):
        # 1. Retrieve docs
        retriever = self.vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([d.page_content for d in docs])

        # 2. Format the agentic prompt
        prompt_text = self.prompt.format(question=question, context=context)

        # 3. Generate answer
        answer = self.llm(prompt_text)

        return answer, docs
