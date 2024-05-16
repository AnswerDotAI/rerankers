from typing import Optional, Union
from pydantic import BaseModel


class Document(BaseModel):
    text: str
    doc_id: Optional[Union[str, int]] = None
    metadata: Optional[dict] = None

    def __init__(
        self,
        text: str,
        doc_id: Optional[Union[str, int]] = None,
        metadata: Optional[dict] = None,
    ):
        super().__init__(text=text, doc_id=doc_id, metadata=metadata)
