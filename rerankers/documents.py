from dataclasses import dataclass
from typing import Optional, Union
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    id: Optional[Union[str, int]] = None
