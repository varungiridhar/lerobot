#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("libero")

from robosuite.utils import binding_utils  # noqa: E402

from lerobot.envs.libero import (  # noqa: E402
    LiberoEnv,
    _install_robosuite_render_context_guard,
)


def test_robosuite_read_pixels_rebinds_context_at_render_entry(monkeypatch):
    events: list[str] = []

    class FakeGLContext:
        def make_current(self):
            events.append("make_current")

    def fake_read_pixels(*, rgb, depth, viewport, con):
        del depth, viewport, con
        events.append("read_pixels")
        rgb.fill(7)

    monkeypatch.setattr(binding_utils.mujoco, "mjr_readPixels", fake_read_pixels)
    render_context = SimpleNamespace(gl_ctx=FakeGLContext(), con=object())

    pixels = binding_utils.MjRenderContext.read_pixels(render_context, width=4, height=3)

    assert events == ["make_current", "read_pixels"]
    assert pixels.shape == (3, 4, 3)
    assert pixels.dtype == np.uint8
    assert np.all(pixels == 7)


def test_robosuite_render_context_guard_is_installed_once():
    guarded_render = binding_utils.MjRenderContext.render
    guarded_read_pixels = binding_utils.MjRenderContext.read_pixels
    assert guarded_render._lerobot_makes_context_current
    assert guarded_read_pixels._lerobot_makes_context_current

    _install_robosuite_render_context_guard()

    assert binding_utils.MjRenderContext.render is guarded_render
    assert binding_utils.MjRenderContext.read_pixels is guarded_read_pixels


def test_libero_step_makes_its_offscreen_context_current_first():
    events: list[str] = []

    class FakeGLContext:
        def make_current(self):
            events.append("make_current")

    class FakeBackend:
        sim = SimpleNamespace(
            _render_context_offscreen=SimpleNamespace(gl_ctx=FakeGLContext())
        )

        def step(self, _action):
            events.append("step")
            return {}, 0.0, False, {}

        def check_success(self):
            return False

    env = object.__new__(LiberoEnv)
    env._env = FakeBackend()
    env.task = "test task"
    env.task_id = 0
    env._format_raw_obs = lambda raw_obs: raw_obs

    env.step(np.zeros(7, dtype=np.float32))

    assert events == ["make_current", "step"]


def test_libero_reports_a_missing_offscreen_context_cleanly():
    env = object.__new__(LiberoEnv)
    env._env = SimpleNamespace(sim=SimpleNamespace(_render_context_offscreen=None))

    with pytest.raises(RuntimeError, match="offscreen GL context"):
        env._make_render_context_current()


def test_libero_backend_uses_persistent_render_context(monkeypatch):
    captured_kwargs = {}

    class FakeOffscreenRenderEnv:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def reset(self):
            return {}

    task = SimpleNamespace(
        name="test task",
        language="test instruction",
        problem_folder="test_problem",
        bddl_file="test.bddl",
    )
    suite = SimpleNamespace(get_task=lambda _task_id: task)
    env = object.__new__(LiberoEnv)
    env.observation_height = 224
    env.observation_width = 224

    monkeypatch.setattr("lerobot.envs.libero.OffScreenRenderEnv", FakeOffscreenRenderEnv)
    monkeypatch.setattr("lerobot.envs.libero.get_libero_path", lambda _name: "/fake/bddl")

    env._make_envs_task(suite, task_id=0)

    assert captured_kwargs["hard_reset"] is False


def test_libero_step_does_not_reset_on_termination():
    events: list[str] = []

    class FakeGLContext:
        def make_current(self):
            events.append("make_current")

    class FakeBackend:
        sim = SimpleNamespace(
            _render_context_offscreen=SimpleNamespace(gl_ctx=FakeGLContext())
        )

        def step(self, _action):
            events.append("step")
            return {}, 1.0, False, {}

        def check_success(self):
            events.append("check_success")
            return True

    env = object.__new__(LiberoEnv)
    env._env = FakeBackend()
    env.task = "test task"
    env.task_id = 0
    env._format_raw_obs = lambda raw_obs: raw_obs
    env.reset = lambda *args, **kwargs: events.append("reset")

    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))

    assert terminated is True
    assert info["final_info"]["is_success"] is True
    assert events == ["make_current", "step", "check_success"]


def test_libero_close_rebinds_context_and_is_idempotent():
    events: list[str] = []

    class FakeGLContext:
        def make_current(self):
            events.append("make_current")

    class FakeBackend:
        sim = SimpleNamespace(
            _render_context_offscreen=SimpleNamespace(gl_ctx=FakeGLContext())
        )

        def close(self):
            events.append("close")

    env = object.__new__(LiberoEnv)
    env._env = FakeBackend()

    env.close()
    env.close()

    assert events == ["make_current", "close"]
    assert env._env is None
