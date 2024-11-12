from typing import Optional, Union, Literal
from pydantic import BaseModel, validator


class Document(BaseModel):
    document_type: Literal["text", "image"] = "text"
    text: Optional[str] = None
    base64: Optional[str] = None
    image_path: Optional[str] = None
    doc_id: Optional[Union[str, int]] = None
    metadata: Optional[dict] = None

    @validator("text")
    def validate_text(cls, v, values):
        if values.get("document_type") == "text" and v is None:
            raise ValueError("text field is required when document_type is 'text'")
        return v

    def __init__(
        self,
        text: Optional[str] = None,
        doc_id: Optional[Union[str, int]] = None,
        metadata: Optional[dict] = None,
        document_type: Literal["text", "image"] = "text",
        image_path: Optional[str] = None,
        base64: Optional[str] = None,
    ):
        super().__init__(
            text=text,
            doc_id=doc_id,
            metadata=metadata,
            document_type=document_type,
            base64=base64,
            image_path=image_path,
        )
