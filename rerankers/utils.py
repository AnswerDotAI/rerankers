import base64
import binascii
from typing import Union, Optional, List, Iterable
try:
    import io
    from PIL import Image
except ImportError:
    pass
from rerankers.documents import Document

def vprint(txt: str, verbose: int) -> None:
    if verbose > 0:
        print(txt)


try:
    import torch

    def get_dtype(
        dtype: Optional[Union[str, torch.dtype]],
        device: Optional[Union[str, torch.device]],
        verbose: int = 1,
    ) -> torch.dtype:
        if dtype is None:
            vprint("No dtype set", verbose)
        # if device == "cpu":
        #     vprint("Device set to `cpu`, setting dtype to `float32`", verbose)
        #     dtype = torch.float32
        if not isinstance(dtype, torch.dtype):
            if dtype == "fp16" or dtype == "float16":
                dtype = torch.float16
            elif dtype == "bf16" or dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        vprint(f"Using dtype {dtype}", verbose)
        return dtype

    def get_device(
        device: Optional[Union[str, torch.device]],
        verbose: int = 1,
        no_mps: bool = False,
    ) -> Union[str, torch.device]:
        if not device:
            vprint("No device set", verbose)
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available() and not no_mps:
                device = "mps"
            else:
                device = "cpu"
            vprint(f"Using device {device}", verbose)
        return device

except ImportError:
    pass


def make_documents(
    docs: List[str],
    doc_ids: Optional[Union[List[str], List[int]]] = None,
):
    if doc_ids is None:
        doc_ids = list(range(len(docs)))
    return [Document(doc, doc_id=doc_ids[i]) for i, doc in enumerate(docs)]


def prep_docs(
    docs: Union[str, List[str], Document, List[Document]],
    doc_ids: Optional[Union[List[str], List[int]]] = None,
    metadata: Optional[List[dict]] = None,
):
    if isinstance(docs, Document) or (
        isinstance(docs, List) and isinstance(docs[0], Document)
    ):
        if isinstance(docs, Document):
            docs = [docs]
        if doc_ids is not None:
            if docs[0].doc_id is not None:
                print(
                    "Overriding doc_ids passed within the Document objects with explicitly passed doc_ids!"
                )
                print(
                    "This is not the preferred way of doing so, please double-check your code."
                )
            for i, doc in enumerate(docs):
                doc.doc_id = doc_ids[i]

        elif doc_ids is None:
            doc_ids = [doc.doc_id for doc in docs]
            if doc_ids[0] is None:
                print(
                    "'None' doc_ids detected, reverting to auto-generated integer ids..."
                )
                doc_ids = list(range(len(docs)))

        if metadata is not None:
            if docs[0].meatadata is not None:
                print(
                    "Overriding doc_ids passed within the Document objects with explicitly passed doc_ids!"
                )
                print(
                    "This is not the preferred way of doing so, please double-check your code."
                )
            for i, doc in enumerate(docs):
                doc.metadata = metadata[i]

        return docs

    if isinstance(docs, str):
        docs = [docs]
    if doc_ids is None:
        doc_ids = list(range(len(docs)))
    if metadata is None:
        metadata = [{} for _ in docs]

    return [
        Document(doc, doc_id=doc_ids[i], metadata=metadata[i])
        for i, doc in enumerate(docs)
    ]


def prep_image_docs(
    docs: Union[str, List[str], Document, List[Document]],
    doc_ids: Optional[Union[List[str], List[int]]] = None,
    metadata: Optional[List[dict]] = None,
) -> List[Document]:
    """
    Prepare image documents for processing. Can handle base64 encoded images or file paths.
    Similar to prep_docs but specialized for image documents.
    """
    # If already Document objects, handle similarly to prep_docs
    if isinstance(docs, Document) or (
        isinstance(docs, List) and isinstance(docs[0], Document)
    ):
        if isinstance(docs, Document):
            docs = [docs]
        # Validate all docs are image type
        for doc in docs:
            if doc.document_type != "image":
                raise ValueError("All documents must be of type 'image'")
        return prep_docs(docs, doc_ids, metadata)

    # Handle string inputs (paths or base64)
    if isinstance(docs, str):
        docs = [docs]

    processed_docs = []
    for doc in docs:
        # Check if input is base64 by attempting to decode
        try:
            # Try to decode and verify it's an image
            decoded = base64.b64decode(doc)
            try:
                Image.open(io.BytesIO(decoded)).verify()
                b64 = doc
                image_path = None
            except:
                raise binascii.Error("Invalid image data")
        except binascii.Error:
            # If decode fails, treat as file path
            try:
                image_path = doc
                with open(doc, 'rb') as img_file:
                    b64 = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                raise ValueError(f"Could not process image input {doc}: {str(e)}")
        
        processed_docs.append(
            Document(
                document_type="image",
                base64=b64,
                image_path=image_path
            )
        )

    # Handle doc_ids and metadata
    if doc_ids is None:
        doc_ids = list(range(len(processed_docs)))
    if metadata is None:
        metadata = [{} for _ in processed_docs]

    # Set doc_ids and metadata
    for i, doc in enumerate(processed_docs):
        doc.doc_id = doc_ids[i]
        doc.metadata = metadata[i]


    return processed_docs




def get_chunks(iterable: Iterable, chunk_size: int):  # noqa: E741
    """
    Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/base.py with extra typing and more descriptive names.
    This method is used to split a list l into chunks of batch size n.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]
