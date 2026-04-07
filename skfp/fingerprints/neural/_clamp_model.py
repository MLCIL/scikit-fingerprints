import functools

import torch
from torch import nn

_KEY_MAP = {
    "compound_encoder.linear_input.weight": "net.0.weight",
    "compound_encoder.linear_input.bias": "net.0.bias",
    "compound_encoder.linear_hidden_l.0.weight": "net.4.weight",
    "compound_encoder.linear_hidden_l.0.bias": "net.4.bias",
    "compound_encoder.linear_output.weight": "net.8.weight",
    "compound_encoder.linear_output.bias": "net.8.bias",
}


class CLAMPCompoundEncoder(nn.Module):
    """
    Two-layer MLP compound encoder from CLAMP [1]_.

    Architecture::

        Input (8192) -> Linear(8192, 4096) -> LayerNorm -> ReLU -> Dropout(0.1)
                     -> Linear(4096, 2048) -> LayerNorm -> ReLU -> Dropout(0.2)
                     -> Linear(2048, 768)
        Output (768)

    Dropout values match the pretrained checkpoint's hp.json (dropout_input=0.1,
    dropout_hidden=0.2). They have no effect at inference time since the model
    is always run in eval() mode, but are kept faithful to the original.

    References
    ----------
    .. [1] `Seidl et al.
        "Enhancing Activity Prediction Models in Drug Discovery with the
        Ability to Understand Human Language"
        ICML 2023
        <https://arxiv.org/abs/2303.03363>`_
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 768),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_clamp_compound_encoder(checkpoint_path: str) -> CLAMPCompoundEncoder:
    """Load pretrained CLAMP compound encoder from a checkpoint file."""
    model = CLAMPCompoundEncoder()

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    full_state_dict = checkpoint["model_state_dict"]
    encoder_state_dict = {
        _KEY_MAP[k]: v for k, v in full_state_dict.items() if k in _KEY_MAP
    }
    model.load_state_dict(encoder_state_dict)
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def get_clamp_model(checkpoint_path: str) -> CLAMPCompoundEncoder:
    """Load and cache the CLAMP compound encoder model."""
    return _load_clamp_compound_encoder(checkpoint_path)
