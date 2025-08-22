from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from dataclasses import dataclass
import json

class BaseChatPrompt:
    """
    Base Chat Prompt Class for inheriting different templates.
    
    Overridable:
      - system_template_str
      - user_template_str
      - input_variables
    """

    system_template_str: str = "You're a helpful AI Assistant."
    user_template_str: str = "Answer the following: {input}"
    input_variables = ["input"]
    
    @classmethod
    def get_prompt(cls) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(cls.system_template_str)
        user_prompt = HumanMessagePromptTemplate.from_template(cls.user_template_str)
        return ChatPromptTemplate.from_messages([system_prompt, user_prompt])
    

def pretty_json(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)