"""
Singleton model loader — prevents loading the same model multiple times.
All services import from here to share a single instance in memory.
Imports are intentionally lazy (inside the function) so that torch and
sentence-transformers are NOT loaded at startup — only on first use.
"""

_models: dict = {}


def get_model(model_name: str):
    """Return cached SentenceTransformer instance, loading it only once."""
    if model_name not in _models:
        # Lazy import — keeps torch/transformers out of RAM until first query
        from sentence_transformers import SentenceTransformer
        print(f"[ModelLoader] Loading model: {model_name}")
        _models[model_name] = SentenceTransformer(model_name)
        print(f"[ModelLoader] Model ready: {model_name}")
    return _models[model_name]
