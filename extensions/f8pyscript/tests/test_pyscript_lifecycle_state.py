from f8pyscript.lifecycle_state import PyScriptLifecycleState


def test_lifecycle_state_exposes_runtime_gates() -> None:
    state = PyScriptLifecycleState()

    assert state.can_handle_data is True
    assert state.can_read_video_latest is True
    assert state.can_tick is False
    assert state.should_stop_in_destructor is False

    state.mark_started()

    assert state.started is True
    assert state.can_tick is True
    assert state.should_stop_in_destructor is True

    state.set_active(False)
    state.mark_paused()

    assert state.can_handle_data is False
    assert state.can_read_video_latest is False
    assert state.can_tick is False

    state.set_active(True)
    state.mark_resumed()

    assert state.can_handle_data is True
    assert state.can_read_video_latest is True
    assert state.can_tick is True

    assert state.begin_close() is True
    assert state.begin_close() is False
    assert state.should_stop_in_destructor is False

    state.mark_stopped()

    assert state.started is False
    assert state.paused is False


def test_lifecycle_compile_failure_keeps_pause_state_unchanged() -> None:
    state = PyScriptLifecycleState(started=True, paused=True, active=False)

    state.mark_compile_failed()

    assert state.started is False
    assert state.paused is True
    assert state.active is False
