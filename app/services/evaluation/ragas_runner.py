import asyncio
import sys
import types
from dataclasses import dataclass
from typing import Any

from datasets import Dataset
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import Generation, LLMResult

# Monkey patch for Ragas 0.4.3 bug with langchain-community
try:
    import langchain_community # noqa
    sys.modules['langchain_community.chat_models'] = types.ModuleType('chat_models')
    sys.modules['langchain_community.chat_models.vertexai'] = types.ModuleType('vertexai')
    setattr(sys.modules['langchain_community.chat_models.vertexai'], 'ChatVertexAI', None)
except ImportError:
    pass

from ragas import evaluate
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from app.core.config import AppConfig
from app.core.logging import get_logger
from app.services.providers.base import Message
from app.services.providers.factory import create_embedder, create_llm

logger = get_logger(__name__)


@dataclass
class CustomRagasLLM(BaseRagasLLM):
    base_llm: Any = None

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None
    ) -> LLMResult:
        messages = prompt.to_messages()
        our_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"

            content = m.content
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        text_parts.append(str(c["text"]))
                    else:
                        text_parts.append(str(c))
                content_str = " ".join(text_parts)
            else:
                content_str = str(content)

            our_messages.append(Message(role=role, content=content_str))

        # We pass the constructed messages to our provider
        response = self.base_llm.generate(our_messages)
        return LLMResult(generations=[[Generation(text=response.text)]])

    async def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None
    ) -> LLMResult:
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float | None = 0.01,
        stop: list[str] | None = None,
        callbacks: Any = None
    ) -> LLMResult:
        return await self.generate(prompt, n, temperature, stop, callbacks)


@dataclass
class CustomRagasEmbeddings(BaseRagasEmbeddings):
    base_embedder: Any = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.base_embedder.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


class RagasRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = create_llm(config)
        self.embedder = create_embedder(config)
        self.llm_wrapper = CustomRagasLLM(base_llm=self.llm)
        self.embeddings_wrapper = CustomRagasEmbeddings(base_embedder=self.embedder)

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

            # In Ragas 0.4.x, evaluate returns EvaluationResult which does not have .items()
            # We can use its __getitem__ which returns a list of scores for that metric
            if hasattr(result, "_repr_dict"):
                # Use the internal representation dict which has the mean scores
                return {str(k): float(v) for k, v in getattr(result, "_repr_dict").items()}

            # Fallback for strict typing if _repr_dict is hidden
            metrics_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            return {name: float(getattr(result, "__getitem__")(name)[0]) for name in metrics_names}

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
