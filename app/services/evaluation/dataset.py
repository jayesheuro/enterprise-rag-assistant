import json
from enum import Enum
from pydantic import BaseModel, Field

class QuestionType(str, Enum):
    FACTUAL = "factual"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS = "ambiguous"

class CreatedBy(str, Enum):
    SYNTHETIC = "synthetic"
    MANUAL = "manual"

class GoldenQuestion(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    expected_source_docs: list[str]
    question_type: QuestionType
    created_by: CreatedBy

class GoldenDataset(BaseModel):
    version: str
    questions: list[GoldenQuestion]
    
    @property
    def path(self) -> str:
        return f"eval/golden/golden_{self.version}.jsonl"

    @property
    def size(self) -> int:
        return len(self.questions)
        
    def filter_by_type(self, qtype: QuestionType) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.question_type == qtype]
        
    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                questions.append(GoldenQuestion(**data))
        
        # Extract version from filename assuming format golden_v1.jsonl
        import os
        filename = os.path.basename(path)
        version = filename.replace("golden_", "").replace(".jsonl", "")
        return cls(version=version, questions=questions)
