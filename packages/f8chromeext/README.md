# Feel The Dance Bridge (Chrome Extension)

This Chrome extension discovers playable `<video>` elements across tabs and streams them to the Feel The Dance React app via WebRTC. It now covers:

- Manifest v3 scaffolding with service worker background logic, tab registry, and toolbar context menu.
- Popup UI that lists all detected videos on the current page, highlights them on hover, and initiates streaming.
- WebRTC sender (source tab) + receiver (Feel tab) peers with background signaling relay, session override, and cleanup.
- Feel tab integration that forwards remote `MediaStream`s to the React player and stops any previous playback/capture sessions automatically.

## Structure

```
packages/feel-the-dance-extension/
├── manifest.json
├── README.md
├── assets/...
└── src
    ├── background.js
    ├── content
    │   ├── feelTab.js
    │   ├── highlight.css
    │   └── videoDetector.js
    └── popup
        ├── index.html
        ├── popup.css
        └── popup.js
```

## Usage

1. Ensure the Feel The Dance web app includes `<meta name="feel-bridge-target" content="feel8-player">` (already added to `packages/feel-the-dance-react/index.html`). That lets the extension recognize the destination tab.
2. In Chrome, open `chrome://extensions`, enable **Developer mode**, click **Load unpacked**, and select `packages/feel-the-dance-bridge`.
3. Open/refresh the Feel The Dance tab so it registers with the bridge. The extension badge will display how many active player tabs are available.
4. Click the extension icon on any page with HTML5 video. The popup lists every detected source with resolution/duration info. Hovering an entry highlights it in the page, and clicking **Stream** captures the video via `captureStream()` and forwards it to the selected Feel tab over WebRTC.
5. When multiple Feel tabs are open, right-click the extension icon to pick the default streaming target (the currently selected tab shows a checkmark). The top menu item opens https://dance.feel8.fun if you need a new player tab.
6. The Feel app automatically stops existing playback/capture sessions when a new remote stream arrives. When the remote feed ends (or the source tab closes), the app reverts to the idle state.

## Notes

- WebRTC uses Google’s public STUN server by default. Update `ICE_CONFIG` in both sender/receiver if you need TURN.
- Only one stream is allowed per Feel tab; new sources automatically override older sessions.
- The popup currently surfaces basic status text. Future diagnostics (stats, bitrate controls, etc.) can hook into the sender/receiver events exposed in `videoDetector.js` and `feelTab.js`.
