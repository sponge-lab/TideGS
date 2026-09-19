from types import SimpleNamespace
from unittest.mock import Mock
import io

import pytest

import scene
import utils.camera_utils as camera_utils
import utils.general_utils as utils
from storage.tide_storage_adapter import TideStorageAdapter


def test_predecode_uses_explicit_args(monkeypatch):
    explicit_args = SimpleNamespace(
        multiprocesses_decode_dataset_to_disk=True,
        decode_dataset_path="explicit",
    )
    global_args = SimpleNamespace(
        multiprocesses_decode_dataset_to_disk=False,
        decode_dataset_path="global",
    )
    called = {}

    monkeypatch.setattr(utils, "ARGS", global_args)

    def fake_predecode(cam_infos, args, num_workers=None):
        called["args"] = args

    monkeypatch.setattr(
        camera_utils,
        "predecode_dataset_to_disk_multiprocess",
        fake_predecode,
    )

    camera_utils.predecode_dataset_to_disk([], explicit_args)

    assert called["args"] is explicit_args


def test_storage_shutdown_is_idempotent():
    class FakeQueue:
        def put(self, value):
            pass

        def join(self):
            pass

    adapter = object.__new__(TideStorageAdapter)
    adapter._closed = False
    adapter._log = lambda message: None
    adapter.flush_resident_dirty = Mock()
    adapter.drain_cache_writebacks = Mock()
    adapter.wait_for_bounds_refresh = Mock()
    adapter._cache_commit_queue = FakeQueue()
    adapter._cache_commit_thread = None
    adapter._bounds_refresh_queue = FakeQueue()
    adapter._bounds_refresh_thread = None
    adapter.pipeline = SimpleNamespace(shutdown=Mock())
    adapter.cache = SimpleNamespace(shutdown=Mock())
    adapter.storage = SimpleNamespace(maybe_compact=Mock(), close=Mock())

    adapter.shutdown()
    adapter.shutdown()

    adapter.flush_resident_dirty.assert_called_once_with()
    adapter.drain_cache_writebacks.assert_called_once_with()
    adapter.wait_for_bounds_refresh.assert_called_once_with()
    adapter.pipeline.shutdown.assert_called_once_with()
    adapter.cache.shutdown.assert_called_once_with()
    adapter.storage.maybe_compact.assert_called_once_with(min_patches=2, force=True)
    adapter.storage.close.assert_called_once_with()


def test_training_cleanup_runs_once_and_surfaces_writeback_failure(monkeypatch):
    import train_tidegs

    resources = {
        "mem_mon": SimpleNamespace(close=Mock()),
        "storage_adapter": SimpleNamespace(shutdown=Mock(side_effect=RuntimeError("missing block 7"))),
        "scene": SimpleNamespace(clean_up=Mock()),
    }
    monkeypatch.setattr(train_tidegs, "_shutdown_double_buffer_gpu", Mock())

    def train(*args):
        args[-1].update(resources)

    monkeypatch.setattr(train_tidegs, "_training_impl", train)
    with pytest.raises(RuntimeError, match="missing block 7"):
        train_tidegs.training(None, None, None, None, io.StringIO())

    resources["mem_mon"].close.assert_called_once_with()
    resources["storage_adapter"].shutdown.assert_called_once_with()
    resources["scene"].clean_up.assert_called_once_with()
    train_tidegs._shutdown_double_buffer_gpu.assert_called_once_with()


def test_cleanup_preserves_original_training_exception(monkeypatch):
    import train_tidegs

    def train(*args):
        args[-1]["mem_mon"] = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failed")))
        raise ValueError("optimizer failed")

    monkeypatch.setattr(train_tidegs, "_training_impl", train)
    log = io.StringIO()
    with pytest.raises(ValueError, match="optimizer failed"):
        train_tidegs.training(None, None, None, None, log)
    assert "close failed" in log.getvalue()
