import json
import os
from transformers import AutoConfig, AutoTokenizer, AutoModel
import datasets
import yaml
from fastapi import Request
from fastapi.responses import JSONResponse

debug_flg = bool(os.environ.get("AGENTENV_DEBUG", False))

if debug_flg:
    print("Debug mode")

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size




def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def load_config(config_file):
    with open(config_file) as reader:
        config = yaml.safe_load(reader)
    return config


class EnvError(Exception):
    """Base class for all environment errors."""
    code: str = "INTERNAL_ERROR"
    status: int = 500
    retryable: bool = False

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EnvNotReadyError(EnvError):
    code = "ENV_NOT_READY"
    status = 503
    retryable = True


class EnvClosedError(EnvError):
    code = "ENV_CLOSED"
    status = 409


class EpisodeFinishedError(EnvError):
    code = "EPISODE_FINISHED"
    status = 409


class TaskOutOfRangeError(EnvError):
    code = "TASK_OUT_OF_RANGE"
    status = 400


class InvalidActionError(EnvError):
    code = "INVALID_ACTION"
    status = 400


class ConfigMissingError(EnvError):
    code = "CONFIG_MISSING"
    status = 503


class EnvNotFoundError(EnvError):
    code = "ENV_NOT_FOUND"
    status = 404


async def env_error_handler(request: Request, exc: EnvError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "details": {},
            }
        },
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "retryable": False,
                "details": {},
            }
        },
    )


def register_error_handlers(app):
    app.add_exception_handler(EnvError, env_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
