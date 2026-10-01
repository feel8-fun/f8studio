#include <gtest/gtest.h>

#include "implayer_playback_state.h"

namespace {

using f8::implayer::MediaSyncAction;
using f8::implayer::PlaybackIntent;
using f8::implayer::media_sync_action_for_intent;
using f8::implayer::playback_intent_should_reload_stopped_media;
using f8::implayer::playback_intent_wants_loaded_media;
using f8::implayer::playback_intent_wants_playback;

TEST(ImPlayerPlaybackState, PlayingIntentLoadsAndStartsPlayback) {
  EXPECT_EQ(media_sync_action_for_intent(PlaybackIntent::Playing), MediaSyncAction::LoadAndPlay);
  EXPECT_TRUE(playback_intent_wants_loaded_media(PlaybackIntent::Playing));
  EXPECT_TRUE(playback_intent_should_reload_stopped_media(PlaybackIntent::Playing));
  EXPECT_TRUE(playback_intent_wants_playback(PlaybackIntent::Playing));
}

TEST(ImPlayerPlaybackState, PausedIntentLoadsWithoutResumingPlayback) {
  EXPECT_EQ(media_sync_action_for_intent(PlaybackIntent::Paused), MediaSyncAction::LoadAndPause);
  EXPECT_TRUE(playback_intent_wants_loaded_media(PlaybackIntent::Paused));
  EXPECT_TRUE(playback_intent_should_reload_stopped_media(PlaybackIntent::Paused));
  EXPECT_FALSE(playback_intent_wants_playback(PlaybackIntent::Paused));
}

TEST(ImPlayerPlaybackState, StoppedIntentPreservesStoppedStateOnSync) {
  EXPECT_EQ(media_sync_action_for_intent(PlaybackIntent::Stopped), MediaSyncAction::PreserveStopped);
  EXPECT_FALSE(playback_intent_wants_loaded_media(PlaybackIntent::Stopped));
  EXPECT_FALSE(playback_intent_should_reload_stopped_media(PlaybackIntent::Stopped));
  EXPECT_FALSE(playback_intent_wants_playback(PlaybackIntent::Stopped));
}

}  // namespace
