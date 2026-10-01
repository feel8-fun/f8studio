#pragma once

namespace f8::implayer {

enum class PlaybackIntent {
  Playing,
  Paused,
  Stopped,
};

enum class MediaSyncAction {
  LoadAndPlay,
  LoadAndPause,
  PreserveStopped,
};

constexpr MediaSyncAction media_sync_action_for_intent(PlaybackIntent intent) {
  switch (intent) {
    case PlaybackIntent::Playing:
      return MediaSyncAction::LoadAndPlay;
    case PlaybackIntent::Paused:
      return MediaSyncAction::LoadAndPause;
    case PlaybackIntent::Stopped:
      return MediaSyncAction::PreserveStopped;
  }
  return MediaSyncAction::LoadAndPlay;
}

constexpr bool playback_intent_wants_loaded_media(PlaybackIntent intent) {
  return media_sync_action_for_intent(intent) != MediaSyncAction::PreserveStopped;
}

constexpr bool playback_intent_wants_playback(PlaybackIntent intent) {
  return media_sync_action_for_intent(intent) == MediaSyncAction::LoadAndPlay;
}

constexpr bool playback_intent_should_reload_stopped_media(PlaybackIntent intent) {
  return playback_intent_wants_loaded_media(intent);
}

}  // namespace f8::implayer
