import modal
from pathlib import Path


MODELS_DIR = "/llm"
MODEL_REPO_ID = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16" # quantized to 4-bit
MODEL_REVISION = None #"a7c09948d9a632c2c840722f519672cd94af885d"

try:
    volume = modal.Volume.from_name("llm", create_if_missing=False)
except modal.exception.NotFoundError:
    raise RuntimeError("Volume not found. Please deploy [download-model-service] first")


# RAG index volume (created by rag_ingest.py)
INDEX_VOLUME_DIR = "/index"
INDEX_VOLUME_NAME = "rag-index"
INDEX_FAISS_PATH = f"{INDEX_VOLUME_DIR}/index.faiss"
META_JSON_PATH = f"{INDEX_VOLUME_DIR}/meta.json"
try:
    index_volume = modal.Volume.from_name(INDEX_VOLUME_NAME, create_if_missing=False)
except modal.exception.NotFoundError:
    raise RuntimeError(
        "Volume 'rag-index' not found. Run rag_ingest.build_index first to create the index."
    )


app = modal.App("vllm-chat-rag")



N_GPU = 1  # first upgrade to more powerful GPUs, and only then increase GPU count
MINUTE = 60
HOUR = 60 * MINUTE

secrets = [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"]),
           modal.Secret.from_name("openai-secret", required_keys=["OPENAI_API_KEY"])   
        ]

# Image used for serving container (add faiss / numpy / openai)
vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.3post1",
    "fastapi[standard]==0.115.4",
    "faiss-cpu",
    "numpy",
    "openai>=1.40.0",
)

@app.cls(
    image=vllm_image,
    gpu="T4", # upgrade to "A100" or "A10G" for larger models
    scaledown_window=5 * MINUTE, # how long the container will wait for new requests before shutting down
    volumes={MODELS_DIR: volume, INDEX_VOLUME_DIR: index_volume},
    timeout=24 * HOUR, # how long the function will run before being terminated
    secrets=secrets,
    #allow_concurrent_inputs=2 # how many requests can be processed concurrently
)
@modal.concurrent(max_inputs=2)
class RAGLLMConfig:


    @modal.enter()
    def load(self):
        from pathlib import Path
        import time
        import asyncio
        from vllm.entrypoints.openai.serving_engine import BaseModelPath
        from vllm.usage.usage_lib import UsageContext
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.logger import RequestLogger
        import sys, types
        import json
        import faiss
        from openai import OpenAI        

        # ---- pyairports stub ----
        if "pyairports" not in sys.modules:
            airports_mod = types.ModuleType("pyairports.airports")
            airports_mod.AIRPORT_LIST = []
            pkg_mod = types.ModuleType("pyairports")
            pkg_mod.airports = airports_mod
            sys.modules["pyairports"] = pkg_mod
            sys.modules["pyairports.airports"] = airports_mod
        # -------------------------
        self.engine = None
        self.model_config = None
        self.request_logger = None
        self.base_model_paths = None

        # RAG state
        self.faiss_index = None
        self.rag_chunks = None
        self.embedding_model = None
        self.rag_dim = None
        self.embed_client = None

        volume.reload() 

        model_path = Path(MODELS_DIR) / MODEL_REPO_ID
        if not model_path.exists() or not any(model_path.glob("*")): 
            print(f"Model not found at {model_path}, downloading...")

            download_model_function = modal.Function.from_name("download-model-service", "download_model")
            download_model_function.remote(MODEL_REPO_ID, MODEL_REVISION)

            while not self.is_model_downloaded(model_path):
                print(f"Model not downloaded yet, waiting...")
                time.sleep(5)
                volume.reload() 

            print(f"Model downloaded to {model_path}")
        else:
            print(f"Model already exists at {model_path}")


        engine_args = AsyncEngineArgs(
            model=MODELS_DIR + "/" + MODEL_REPO_ID,
            tensor_parallel_size=N_GPU, # allows model computations to be distributed across multiple GPUs for better performance
            gpu_memory_utilization=0.90, #  allowing the engine to use up to 90% of available GPU memory to balance between performance and resource allocation
            max_model_len=8096, # maximum number of tokens the model can process in a single request
            enforce_eager=False, 
        )

        self.engine = AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.OPENAI_API_SERVER
        )
        
        
        self.model_config = asyncio.run(self.engine.get_model_config())

       #self.model_config = self.get_model_config(self.engine)

        self.request_logger = RequestLogger(max_log_len=2048)

        self.base_model_paths = [
            BaseModelPath(name=MODEL_REPO_ID.split("/")[1], model_path=MODEL_REPO_ID)
        ]

        # ---------- RAG: index + metadata ----------
        index_volume.reload()

        print(f"Loading FAISS index from {INDEX_FAISS_PATH} ...")
        self.faiss_index = faiss.read_index(INDEX_FAISS_PATH)

        with open(META_JSON_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.rag_chunks = meta["chunks"]
        self.embedding_model = meta["embedding_model"]
        self.rag_dim = meta["dim"]

        print(
            f"Loaded RAG index: {meta['num_vectors']} vectors, dim={self.rag_dim}, "
            f"embedding_model={self.embedding_model}"
        )

        # OpenAI client for embeddings (needs OPENAI_API_KEY in env / secret)
        self.embed_client = OpenAI()

    # ------------------ RAG helper ------------------

    def retrieve_chunks(self, query: str, k: int = 3):
        """
        Embed the query with the same model used at ingestion,
        search the FAISS index, and return top-k chunk dicts.
        """
        import numpy as np
        import faiss

        query = (query or "").strip()
        if not query or self.faiss_index is None or self.rag_chunks is None:
            return []

        resp = self.embed_client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        qvec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)

        # cosine-style search (embeddings were L2-normalized at index build)
        faiss.normalize_L2(qvec)
        D, I = self.faiss_index.search(qvec, k=k)

        results = []
        for score, idx in zip(D[0], I[0]):
            chunk = self.rag_chunks[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "doc_path": chunk["doc_path"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )
        return results


    @modal.asgi_app()
    def serve(self):
        import fastapi
        import vllm.entrypoints.openai.api_server as api_server
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest


        web_app = fastapi.FastAPI(
            title=f"OpenAI-compatible {MODEL_REPO_ID} server",
            description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
            version="0.0.1",
            docs_url="/docs",
        )

        web_app.add_middleware(
            fastapi.middleware.cors.CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        router = fastapi.APIRouter()

        router.include_router(api_server.router)
        web_app.include_router(router)


        # ----- Custom chat factory with RAG -----

        class RAGChat(OpenAIServingChat):
            def __init__(self, rag_cfg: "RAGLLMConfig", *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._rag_cfg = rag_cfg

            def _add_rag_context_to_messages(self, messages: list) -> list:
                """
                Private helper:
                - Finds last user message.
                - Uses backend RAG to get chunks.
                - Appends context directly into that user message.
                - Returns the updated messages list.
                """
                if not messages:
                    return messages

                # 1. Find last user message
                last_user_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        last_user_idx = i
                        break

                if last_user_idx is None:
                    return messages

                content = messages[last_user_idx].get("content", "")
                if not isinstance(content, str) or not content.strip():
                    return messages

                # 2. Retrieve chunks with relevance threshold
                chunks = self._rag_cfg.retrieve_chunks(content, k=3)
                if not chunks:
                    # No relevant docs → leave request unchanged
                    return messages

                # 3. Build context string (you can drop the 800-char cut if you want)
                blocks = []
                for r in chunks:
                    text = r["text"]
                    if len(text) > 800:         # just a safety cap on characters
                        text = text[:800]
                    blocks.append(f"[{r['doc_path']}#{r['chunk_id']}]:\n{text}")

                context_str = "\n\n".join(blocks)

                # 4. Append context directly to the user message    (NO system msg)
                new_content = (
                    content
                    + "\n\n[Retrieved context from your documents:]\n"
                    + context_str
                )

                messages[last_user_idx]["content"] = new_content
                return messages

            async def create_chat_completion(self, request, raw_request=None):
                # Turn into dict just to access messages easily
                data = (
                    request.model_dump()
                    if hasattr(request, "model_dump")
                    else request.dict()
                )
                messages = data.get("messages", []) or []

                # Let the helper decide whether/how to inject RAG context
                updated_messages = self._add_rag_context_to_messages(messages)

                # Mutate the existing pydantic model in-place
                request.messages = updated_messages

                # Now call the original implementation
                return await super().create_chat_completion(request, raw_request)


        def chat_factory(settings=None):
            return RAGChat(
                self,
                self.engine,
                model_config=self.model_config,
                base_model_paths=self.base_model_paths,
                chat_template=None,
                response_role="assistant",
                lora_modules=[],
                prompt_adapters=[],
                request_logger=self.request_logger,
            )

        api_server.chat = chat_factory

        api_server.completion = lambda s: OpenAIServingCompletion(
            self.engine,
            model_config=self.model_config,
            base_model_paths=self.base_model_paths,
            lora_modules=[],
            prompt_adapters=[],
            request_logger=self.request_logger,
        )

        return web_app
    
    @staticmethod
    def is_model_downloaded(model_path: Path) -> bool:
        completion_file = model_path / "download_complete"
        return completion_file.exists()