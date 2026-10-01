from rag.config import DEFAULT_CONFIG, PipelineConfig, load_config


def test_baseline_config_loads():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.name == "baseline"
    assert cfg.collection_name.startswith("fb_focused_")


def test_index_key_ignores_query_time_settings():
    base = PipelineConfig(name="a")
    other = base.model_copy(
        update={
            "name": "b",
            "retriever": base.retriever.model_copy(update={"top_k": 20}),
            "generator": base.generator.model_copy(update={"provider": "anthropic"}),
        }
    )
    assert base.collection_name == other.collection_name


def test_index_key_changes_with_chunking():
    base = PipelineConfig(name="a")
    bigger = base.model_copy(update={"chunker": base.chunker.model_copy(update={"size": 800})})
    assert base.collection_name != bigger.collection_name
