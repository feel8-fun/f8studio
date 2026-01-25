# Store lising

## Product details (EN)
Feel Bridge: Empower Your Online Video Experience with Advanced Streaming and Mirroring

Feel Bridge is a sophisticated browser extension designed to enhance how you interact with online videos across the web. It automatically detects and catalogs all playable video, livestream, and screen capture elements embedded within any webpage. This extension serves as a seamless “bridge” that connects your browser’s media content directly to external platforms, synchronized viewing environments, or specialized services such as the Feel web studio. Whether you're a casual viewer or an advanced user wanting greater control, Feel Bridge streamlines media management and delivers a powerful, real-time streaming experience right from your browser.

At its core, Feel Bridge continuously scans web pages for video elements and dynamically compiles an interactive list displayed within a sleek popup interface. From here, you can easily identify, select, and highlight videos on the page. By utilizing cutting-edge WebRTC technology, the extension enables peer-to-peer streaming from your current tab directly to a paired target tab or external platform. This allows you to mirror any playable video, livestream, or even your screen capture in real time, all with just one click. It also gracefully manages switching between tabs during streaming sessions and remembers your preferred stream target for a smooth, consistent experience.

Feel Bridge offers a rich set of features that facilitate advanced media interaction. Videos are assigned unique identifiers and tracked as part of an interactive catalog that remains updated as you navigate or interact with complex pages. The extension supports highlighting specific media sources, making it easier to spot and control videos even amid multiple embedded elements. You can start bridging video streams to external players or synchronized watch parties through a secure, browser-native WebRTC connection, ensuring fast, reliable streaming without routing data through third-party servers. Context menu integration further streamlines your workflow by allowing instant video management actions from anywhere in your browsing session.

Designed with privacy and versatility in mind, Feel Bridge operates entirely locally within your browser. It never collects personal data or uploads browsing history. All video detection and streaming processes run on your device, with communication secured through browser messaging protocols and WebRTC ICE servers for peer-to-peer connectivity. Additionally, the extension supports localization and multilingual interfaces through flexible translation utilities, making its powerful features accessible worldwide.

Key Features:
- Automatic real-time detection and cataloging of all playable video, livestream, and screen capture elements on any webpage.
- Interactive popup interface displaying every media source with additional metadata, previews, and position highlighting.
- Seamless one-click WebRTC streaming (“bridging”) from your browser directly to a paired target tab or external platform like the Feel web studio.
- Intuitive video highlighting for effortless identification and control amid multiple embedded videos.
- Smart tracking of your preferred streaming target and smooth handling of tab switching during active sessions.
- Context menu integration enabling quick access to video management and streaming commands.
- Full privacy assurance with zero personal data collection and local processing of all detected content.
- Multilingual support powered by dynamic browser and custom translation tools for a global user experience.

How It Works:
1. Install Feel Bridge from your browser’s extension store and pin the icon to your toolbar.
2. Browse any webpage containing video content—Feel Bridge automatically detects and catalogs all playable media elements in real time.
3. Click the extension icon to open the popup interface, which lists every detected video, livestream, or screen capture, complete with metadata and visual highlights.
4. Hover over or select any media source to trigger dynamic highlighting directly on the page for easy identification.
5. Initiate streaming by clicking the “stream” button, which instantly begins WebRTC mirroring to a paired target tab or external platform such as the Feel web studio.
6. Enjoy seamless switching between browser tabs during streaming sessions; the extension remembers your chosen target for future streams.
7. Use context menu options (right-click) anywhere in your browsing session to quickly manage, capture, or stream media with minimal effort.

Privacy: - No personal data collected. All detection and streaming are handled locally within the extension using secure, direct WebRTC connections, ensuring your browsing history and personal information remain private and never leave your device.

## Product details (ZH)
Feel Bridge：让浏览器里的任意视频、直播或屏幕捕获一键镜像到 Feel Web Studio

Feel Bridge 是一款浏览器扩展，用于自动发现并管理网页中的可播放视频、直播流和屏幕捕获源。它会实时扫描页面上的 video 元素，并在弹窗中列出可用媒体，支持高亮定位、筛选和一键启动 WebRTC 串流。你可以将当前标签页中的任意视频镜像到指定的 Feel 播放标签页或其他配对端，无需上传、无需第三方服务器，中途切换标签页也能平滑继续。

主要特性：
- 自动检测网页上的所有可播放视频、直播或屏幕捕获源，并展示详细元数据
- 弹窗里一键高亮定位页面视频，方便在复杂页面中找到目标
- 通过安全的 WebRTC 直连，将视频从当前标签镜像到配对的 Feel 标签页或外部平台
- 记住上次使用的目标标签，流畅处理标签切换和会话恢复
- 右键菜单快速调用检测/串流指令，提升工作流效率
- 全流程本地处理，不收集个人数据；通信仅用于浏览器内消息和 WebRTC ICE 连接
- 多语言界面，便于全球用户使用

使用步骤：
1) 安装并固定 Feel Bridge 图标；  
2) 打开包含视频的网页，扩展会自动检测并列出所有可播放媒体；  
3) 点击扩展弹窗，悬停或选择视频可在页面高亮对应元素；  
4) 点击“串流”按钮，通过 WebRTC 将当前媒体镜像到选定的 Feel 标签页；  
5) 如需切换标签或来源，扩展会记住你的目标并平滑恢复；  
6) 右键菜单同样可快速执行检测和串流操作。

## Category: Entertainment


# Privacy
To facilitate the compliance of your extension with the Chrome Web Store Developer Program Policies, you are required to provide the information listed below. The information provided in this form will be shared with the Chrome Web Store team. Please ensure that the information provided is accurate, as it will improve the review time of your extension and decrease the risk of this version being rejected.

## Single purpose
An extension must have a single purpose that is narrow and easy-to-understand. Learn more

Feel The Dance by Feel8.Fun lets you mirror any playable video, livestream, or screen capture from Chrome straight into the Feel web studio. The companion popup lists every media element on the page, highlights sources you hover, and starts WebRTC streaming to a paired Feel tab with a single click. It remembers your preferred target, gracefully handles switching tabs, and never collects or uploads your browsing history.

## Permission justification
A permission is either one of a list of known strings, such as "activeTab", or a match pattern giving access to one or more hosts. Remove any permission that is not needed to fulfill the single purpose of your extension. Requesting an unnecessary permission will result in this version being rejected.

 - tabs: Required to enumerate Feel tabs, track preferred targets, open the Feel Studio tab when none is available, and clean up/stop sessions when tabs close.
 - activeTab: Needed so the popup can message the currently focused page to collect video metadata, highlight elements, and start/stop streaming from that page.
 - contextMenus justification: Provides action menu entries to pick the target Feel tab and quickly open the studio from the extension icon.
 - webNavigation justification: Used in the popup to enumerate frames (`getAllFrames`) so videos inside iframes can be discovered and streamed; also helps handle tab/frame navigation events cleanly.
 - Host permission justification: Content scripts must run on all http/https pages to detect playable video elements, highlight them, and start WebRTC streaming from any site the user chooses.
 - Are you using remote code? No

# Privacy policy
An extension must have a privacy policy if it collects user data.

Privacy policy URL: https://dance.feel8.fun/privacy
