from dataclasses import dataclass
from typing import Optional, Union
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    id: Optional[Union[str, int]] = None
    def __init__(self, text: str, id: Optional[Union[str, int]] = None, **data):
        super().__init__(text=text, id=id, **data)
