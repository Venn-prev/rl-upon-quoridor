import pytest
import torch

from quoridor_ai.core.encoding import PLANES_BY_VERSION, encode_batch, version_for_planes
from quoridor_ai.core.engine import ACTION_SIZE, State
from quoridor_ai.model import IncompatibleCheckpoint, PolicyValueNet, net_from_checkpoint, planes_of


def test_policy_value_net_forward_shapes_for_supported_encodings():
    for version, planes in PLANES_BY_VERSION.items():
        net = PolicyValueNet(8, 1, planes).eval()
        states = torch.from_numpy(encode_batch([State(), State(ply=3)], version)).float()
        logits, values = net(states)
        assert logits.shape == (2, ACTION_SIZE)
        assert values.shape == (2,)
        assert torch.all(values.abs() <= 1)
        assert net.planes == planes


def test_net_from_checkpoint_restores_architecture_and_weights():
    for version, planes in PLANES_BY_VERSION.items():
        source = PolicyValueNet(16, 2, planes).eval()
        checkpoint = {"model": source.state_dict()}
        restored = net_from_checkpoint(checkpoint)
        assert restored.channels == 16
        assert restored.blocks == 2
        assert restored.planes == source.planes
        assert version_for_planes(restored.planes) == version
        states = torch.from_numpy(encode_batch([State()], version)).float()
        with torch.inference_mode():
            assert torch.allclose(restored(states)[0], source(states)[0])


def test_planes_of_reads_the_stem_shape():
    for planes in PLANES_BY_VERSION.values():
        net = PolicyValueNet(8, 1, planes)
        assert planes_of(net.state_dict()) == planes


def test_net_from_checkpoint_rejects_old_architecture():
    checkpoint = {
        "model": {
            "stem.0.weight": torch.zeros(96, 7, 3, 3),
            "body.0.weight": torch.zeros(1),
        }
    }
    with pytest.raises(IncompatibleCheckpoint):
        net_from_checkpoint(checkpoint)
