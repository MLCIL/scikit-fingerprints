from huggingface_hub import hf_hub_download


def _get_weights_path(repo_id: str, filename: str) -> str:
    """Download CLAMP compound encoder weights from HuggingFace Hub if needed,
    returning the local cached path.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )
