import asyncio
from typing import Any, Dict, List
from datasets import Dataset

from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings

from app.core.config import AppConfig
from app.services.providers.base import Message, BaseLLM, BaseEmbedder
from app.services.providers.factory import create_llm, create_embedder
from app.core.logging import get_logger

logger = get_logger(__name__)

class CustomRagasLLM(BaseRagasLLM):
    def __init__(self, base_llm: BaseLLM):
        self.base_llm = base_llm

    def generate(self, messages: list[list[BaseMessage]], **kwargs: Any) -> LLMResult:
        our_messages = []
        for m in messages[0]:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            our_messages.append(Message(role=role, content=m.content))
        
        response = self.base_llm.generate(our_messages, **kwargs)
        return LLMResult(generations=[[Generation(text=response.text)]])

    async def agenerate(self, messages: list[list[BaseMessage]], **kwargs: Any) -> LLMResult:
        return await asyncio.to_thread(self.generate, messages, **kwargs)


class CustomRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, base_embedder: BaseEmbedder):
        self.base_embedder = base_embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.base_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.base_embedder.embed_query(text)
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


class RagasRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = create_llm(config)
        self.embedder = create_embedder(config)
        self.llm_wrapper = CustomRagasLLM(self.llm)
        self.embeddings_wrapper = CustomRagasEmbeddings(self.embedder)

    def evaluate_question(
        self, 
        question: str, 
        answer: str, 
        retrieved_contexts: list[str], 
        ground_truth: str
    ) -> dict[str, float]:
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [retrieved_contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)
        
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=self.llm_wrapper,
                embeddings=self.embeddings_wrapper
            )
            return dict(result)
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }


def compute_refusal_accuracy(answer: str, is_out_of_scope: bool) -> float:
    """
    Computes accuracy of refusal.
    Returns 1.0 if correct behavior, 0.0 otherwise.
    """
    refusal_phrase = "I don't have enough information in the available documents to answer this question."
    did_refuse = refusal_phrase in answer
    
    if is_out_of_scope:
        return 1.0 if did_refuse else 0.0
    else:
        return 0.0 if did_refuse else 1.0
