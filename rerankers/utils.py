from typing import Union, Optional, List, Iterable


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
        if device == "cpu":
            vprint("Device set to `cpu`, setting dtype to `float32`", verbose)
            dtype = torch.float32
        if not isinstance(dtype, torch.dtype):
            if dtype == "fp16" or "float16":
                dtype = torch.float16
            elif dtype == "bf16" or "bfloat16":
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
    print("Torch not installed...")


def ensure_docids(
    doc_ids: Optional[Union[List[str], List[int]]], len_docs: int
) -> Union[List[str], List[int]]:
    if doc_ids is None:
        return list(range(len_docs))
    return doc_ids


def ensure_docs_list(docs: Union[str, List[str]]) -> List[str]:
    if isinstance(docs, str):
        return [docs]
    elif isinstance(docs, List) and all(isinstance(doc, str) for doc in docs):
        return docs
    else:
        raise ValueError(
            f"docs must be a string or a list of strings, not {type(docs)}"
        )


def get_chunks(iterable: Iterable, chunk_size: int):  # noqa: E741
    """
    Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/base.py with extra typing and more descriptive names.
    This method is used to split a list l into chunks of batch size n.
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]
