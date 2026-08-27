"""The `step` control message (spec §12.3.5): one frame at an arbitrary t.

Drives ``_receive_loop`` directly rather than ``run()``. The paced send loop
would otherwise be racing this one for the same encoder, and the whole point
of the assertion is that the bytes are exactly what a frame at that t is.
"""

import asyncio
import json

import pytest

from luminary.comms.codec import CodecConfig
from luminary.drivers.websocket_driver import WebSocketSession, _Disconnect
from luminary.engine.engine import Engine
from luminary.geometry.capture import CaptureParams, capture
from luminary.geometry.scaffold import Scaffold
from luminary.patterns.registry import default_registry

SCAFFOLD = {
    "schema": "luminary.scaffold/1",
    "space": {"authoritative": ["xy"]},
    "lines": [
        {"p1": [0, 0], "p2": [200, 0]},
        {"p1": [200, 0], "p2": [200, 200]},
        {"p1": [200, 200], "p2": [0, 200]},
        {"p1": [0, 200], "p2": [0, 0]},
    ],
    "meta": {"name": "step-test"},
}


class FakeSocket:
    """The Starlette WebSocket shape, as much of it as the driver touches."""

    def __init__(self, controls):
        self.sent = []
        self._inbox = [{"text": json.dumps(c)} for c in controls]
        self._inbox.append({"type": "websocket.disconnect"})

    async def send_bytes(self, data):
        self.sent.append(data)

    async def receive(self):
        return self._inbox.pop(0)


def _engine():
    lights = capture(Scaffold.load(SCAFFOLD), CaptureParams(count_per_line=16))
    return Engine(lights, default_registry().get("simple"), codec_config=CodecConfig())


def _controls(*controls):
    """Run just the control-message half of a session; return it."""
    session = WebSocketSession(_engine(), FakeSocket(controls))
    with pytest.raises(_Disconnect):
        asyncio.run(session._receive_loop())
    return session


def test_step_emits_exactly_that_frame_and_pauses():
    t = 12.5
    session = _controls({"type": "step", "t": t})
    assert session.paused
    assert session.websocket.sent == _engine().frame(t)


def test_step_seeks_backwards_too():
    """t is a coordinate, not a cursor: the paced loop can only go forward."""
    session = _controls({"type": "step", "t": 40.0}, {"type": "step", "t": 1.0})
    replay = _engine()
    assert session.websocket.sent == replay.frame(40.0) + replay.frame(1.0)


@pytest.mark.parametrize("control", [{"type": "step"}, {"type": "step", "t": "soon"}])
def test_step_without_a_usable_t_is_ignored(control):
    session = _controls(control)
    assert not session.paused
    assert session.websocket.sent == []
