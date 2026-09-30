import pytest

from f8studio_server.runtime_identity import StudioRuntimeIdentity


@pytest.mark.parametrize("node_id", ["studio", "operator"])
def test_endpoint_round_trip(node_id: str) -> None:
    identity = StudioRuntimeIdentity("studio_current")
    assert identity.to_public(identity.to_runtime("studio"), identity.to_runtime(node_id)) == ("studio", node_id)
    assert identity.to_public("engine", "operator") == ("engine", "operator")


def test_epoch_isolation_and_unbound_identity() -> None:
    identity = StudioRuntimeIdentity("studio_current")
    assert identity.to_public("studio_other", "operator") is None
    assert identity.to_public("studio", "studio") is None
    unbound = StudioRuntimeIdentity()
    assert unbound.to_public("studio", "studio") == ("studio", "studio")
    assert unbound.to_public("studio_other", "studio_other") is None
