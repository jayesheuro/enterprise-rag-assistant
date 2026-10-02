"""Loads and renders prompt templates from YAML files."""

import yaml
from pathlib import Path
from pydantic import BaseModel
from app.core.config import PROJECT_ROOT

class PromptTemplate(BaseModel):
    name: str
    version: str
    template: str
    template_vars: list[str]
    changelog: list[dict]

def load_prompt(name: str) -> PromptTemplate:
    prompt_path = PROJECT_ROOT / "prompts" / f"{name}.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template {name} not found at {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return PromptTemplate(**data)

def render_prompt(template: PromptTemplate, **kwargs) -> str:
    # Verify all variables are provided
    for var in template.template_vars:
        if var not in kwargs:
            raise ValueError(f"Missing variable '{var}' for prompt template '{template.name}'")
            
    return template.template.format(**kwargs)
