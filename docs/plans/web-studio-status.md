# Web Studio 迁移状态

最后更新：2026-09-25。目标方案：[web-studio-migration.md](web-studio-migration.md)。

## 当前结论

迁移期间的 Video/Audio/3D Media Lab 页面已移除，包括独立测试播放器、演示骨架、专用样式和未再使用的前端像素采样接口。正式入口统一为节点预览和 Outputs；节点放大查看均使用 `?view=outputs&node=...`，3D 聚焦视图保留完整画布和交互帧率，无数据时等待真实骨架。媒体 E2E 已改走正式渲染组件，合成音视频源仅继续用于测试。

P7 后的图编辑精修新增 Node Library 分组、手动 command 调用、节点内 3D 预览、按节点定位的同页输出视图、可固定排序的 Live Outputs，以及前端本地扩展注册边界。已修复同一保存图再次运行时只命中历史成功记录、未检查现场服务的部署错误；日志中心展示当前服务进程生命周期内有界保留的服务输出、部署结果、运行时/API 错误和媒体网关请求错误。页面标题与产品名已合并到顶栏。第三方扩展包安装和通用 Zenoh 数据桥接仍待实施；跨浏览器标签共享同一个 WebRTC peer 未实现。细节与验收见 [web-studio-refinement.md](web-studio-refinement.md)。

2026-09-23 对已保存的 `Untitled 10` 图（revision 89）实测：不改动 Screen Capture → Video Viz 连线，重启 Studio 后再次部署从 `queued` 进入 `succeeded`，Screen Capture 的 `captureRunning=true`、输出 1920×1080，浏览器节点预览解码出 640×360 视频帧。日志中心可回看服务启动与部署过程。

P0、P1、P2、P3、P3.5、P4、P5、P6 和 P7 已在 Linux 主机完成验证。P7 已删除旧 Qt Studio、两个旧扩展包及 Unity 子模块中的 PySide setup UI，移除 Qt/NodeGraphQt/PyQtGraph 依赖和旧 `studio` feature，并将 OpenCV 切换到 headless 发行包。正式启动器现在启动 Web Studio、等待 HTTP health 后打开浏览器；生产 wheel 内嵌本地 Web bundle。Windows 初步迁移与实机验证见下节；全新构建树和完整发布安装仍保留门禁。

P6 新增 SQLite 持久化的 Agent 会话/消息/工具进度/产物/审批，确定性建图与诊断闭环，服务端 provider 注册，Web Agent 工作区，以及复用同一 HTTP/application service 的 CLI 和 MCP。OpenAI、Anthropic、Gemini 与 Ollama 的凭据只从服务端环境读取；当前主机没有模型凭据，因此真实模型 smoke 明确跳过，未宣称供应商在线请求通过。P5 新增本地资产/不可变版本/项目快照、component 捕获与插入、variant 状态应用、节点 schema 写回、从本地 bundle 加载的 Monaco、每会话 `basedpyright-langserver --stdio` completion/hover 和确定性诊断、音频频谱、曲线/track/TCode presentation renderer、节点内 Video Viz 实时预览、Template Match 浏览器截图裁剪、Unity detect/preview/confirm-apply、完整 UDP 骨架帧验证、串口枚举、Web 局部快捷键，以及 SQLite 持久化的无 Qt Win32/X11 全局快捷键后端。P4 使用 React Flow 接入权威项目文档和 catalog，已具备项目创建/选择、可持久化缩放的 service canvas 与 operator 嵌套、动态端口、typed data/state/exec/command 连接规则、data edge policy、约束拖放、级联删除、子图复制、多选、撤销/重做、自动持久化重开、schema 驱动的 Inspector 与 inline state controls、上游 state 只读联动、部署/停止，以及 draft/layout/deployed revision 和逐服务部署错误展示。P3.5 将 Zenoh 订阅、媒体转换、aiortc peer 和编码执行迁入独立 `f8media_gateway` 进程。多 peer 软件编码在组合负载下的 main 为 26.86-27.52 FPS，低于 28 FPS 初始预算；硬件/共享编码仍是发布前优化项。`f8studio_core`、`f8media_protocol`、`f8media_gateway`、`f8studio_server` 和 `f8studio_web` 均为独立包。

## Windows 初步验证（2026-09-24）

Windows 11 10.0.22621 上安装 `web-studio-test` 与 `web-studio-runtime`；Python 3.14.7、FastAPI 0.141.1、Uvicorn 0.53.0、aiortc 1.15.0、PyAV 17.1.0 和 Zenoh 1.9.0 可用，H.264/VP8/VP9 软件编码器可见。`studio_python_typecheck` 为 0 errors，`studio_no_qt_check` 通过；core/server/media gateway 测试分别为 21/70/24 passed，Web 单测为 33 passed，Windows Google Chrome E2E 为 42 passed、4 skipped（desktop/mobile）。Win32 `RegisterHotKey` 实机注册与注销通过；尚未验证键盘在 Studio 失焦时实际触发。

正式服务上的 `studio_runtime_probe` 启动 PyEngine、部署图并确认 `processed=6`；`studio_real_video_probe` 启动新编译的 `f8screencap_service.exe`，经 Zenoh/WebRTC 解码出 1920×540 的真实采集帧并停止受管服务。Windows 完整 `dist_ci --archive` 生成 `build/dist/f8studio-windows-x86_64.zip`；新 Nuitka `f8studio.exe --dry-run` 返回 0。`studio_release_smoke` 在临时 Windows venv 非 editable 安装六个 wheel，确认嵌入 Web 首页与 health。此轮使用已有 Conan/CMake 构建树，尚未从空构建树重新 bootstrap；zip 的 `install_env.bat` 未在全新目录执行。实体 Unity/VaM/Unreal 安装、真实串口、焦点外热键触发和 Windows 媒体性能预算仍待目标环境验证。

## 基线

| 项目 | 记录 |
| --- | --- |
| 基线 commit | `cd400f7d8f83c97eddb6730df06942c9b7488a3f` |
| 系统 | Linux 6.8.0-139-generic x86_64，glibc 2.39 |
| CPU | Intel Core i7-8700，12 logical CPUs |
| GPU | NVIDIA GeForce GTX 1080；Intel UHD Graphics 630 |
| Python | 3.14.7 |
| Node | 24.21.0（Pixi `web-studio` 环境） |
| FastAPI / Uvicorn | 0.141.1 / 0.53.0 |
| aiortc / PyAV | 1.15.0 / 17.1.0 |
| eclipse-zenoh | 1.9.0（与当前原生 Zenoh 运行时保持兼容） |
| 首发浏览器范围 | Windows/Linux Chromium；Linux 与 Windows Google Chrome E2E 已通过 |

`studio_dependency_probe` 已成功导入 FastAPI、Zenoh、aiortc、PyAV，当前 Linux 构建报告 H.264、VP8、VP9 软件编码器可见。这只证明依赖和 codec 注册可用，不是 WebRTC 性能通过。

## 阶段状态

| 阶段 | 状态 | 证据 / 下一步 |
| --- | --- | --- |
| P0 基线与环境 | 完成（Linux） | 独立环境可求解；严格 Python/TS 类型检查、单元测试、生产构建、真实 HTTP 探测、npm audit、无 Qt 审计通过 |
| P1 图模型与编译器 | 完成 | 纯数据文档、显式目录/端口、原子 typed patch、双 revision、history、校验、确定性编译与新格式样例均有测试 |
| P2 无界面运行时 | 完成（Linux） | typed HTTP/WS、13 个 Studio 算子、真实 PyEngine、内置 Studio graph/presentation、部分部署、断连和关闭均已验证 |
| P3 视频与 3D 原型 | 完成（Linux，性能降级已记录） | 视频/数值/精确 overlay/音频/Three.js/重启/延迟/5 分钟组合与 30 分钟稳定性均有真实证据；组合 main 未达 28 FPS，见下方限制 |
| P3.5 媒体网关进程隔离 | 完成（Linux） | `f8media-api/1`、远程代理、独立 PID、真实 screencap 链路和父进程关闭回收均已验证；共享/native 编码待后续优化 |
| P4 Web 图编辑 | 完成（Linux） | 空图到内置 Studio runtime 的创建、配置、部署、monitor、再次修改和重开闭环通过；service/operator 容器、typed 连线、edge policy、动态状态控件、历史与冲突恢复均有测试；300/600 预算通过并记录 1000/2000 压力曲线 |
| P5 本地业务能力 | 完成（Linux；平台/硬件门禁保留） | 本地资产、schema、Monaco/LSP、renderer、两个扩展、Unity/UDP/串口及原生快捷键实现完成；X11 焦点外触发通过，Windows 热键、真实游戏安装和真实串口需目标机验证 |
| P6 AI / CLI / MCP | 完成（Linux；真实模型按凭据门禁） | 确定性 build/diagnose、精确审批、会话/产物、CLI、MCP 与跨页面 `graph.committed` 闭环通过；本机无 provider key，真实请求明确跳过 |
| P7 移除 Qt | 完成（Linux；Windows 发布验证待完成） | 旧 Qt 包/UI/依赖和 feature 已删除；headless OpenCV、Web 启动器、嵌入式前端 wheel、仓库/运行时 Qt 审计和离线安装 smoke 通过 |

## 功能迁移台账

状态含义：`待迁移` 表示已纳入范围但新链路不可用；`基础已建` 表示只有公共入口或契约；`核心完成` 表示无界面领域层已经完成，但还未接入 P2 API 或运行时。

| 领域 | 现有能力 | 新目标 | 状态 |
| --- | --- | --- | --- |
| 图文档 | 节点、服务容器、data/state/exec/command 边、布局 | 纯数据文档、显式端口类型、graph/layout revision | 核心完成 |
| 图操作 | 创建、删除、连接、动态 spec、undo/redo | 原子 typed patch、幂等、冲突和历史 | Web 基础编辑、端口方向/kind/payload/基数/cycle/exec service 约束、data queue/latest policy、service 级联删除、约束重绑定、原子容器复制与显式 service binding 完成 |
| 编译部署 | NodeGraphQt 对象编译为 runtime graph | 纯文档确定性编译、异步部署 job | 核心与 API 完成；真实 PyEngine 部署通过 |
| 服务运行时 | 发现、启停、状态、命令、监控、受管进程 | 无 Qt application service | 基础完成；真实 Zenoh、状态、exec 与关闭通过 |
| Studio 算子 | text/track/wave/video/audio/3D、control、expr、patch hub 等 | 后端算子与前端 renderer 分离 | 14 个后端算子静态注册，新增 `f8.viz.tcode`；不走旧插件加载器 |
| 视频 | BGRA、FLOW2_F16、SCALAR1_F32 latest-frame | WebRTC 分级预览、精确 overlay、数值查询 | 垂直原型完成；传输 v2 携带 producer epoch，overlay 按 source/stream/epoch/frame/timestamp 精确匹配，超时隐藏并计数 |
| 3D | Three.js 骨架、world-up、多人 | 本地打包 Three.js + 有界数据通道 | 垂直原型完成：本地 Three.js、world-up、多人协议、事件重连和显式资源释放 |
| 音频与曲线 | 播放、波形、频谱、track | WebRTC 音轨和降采样绘制 | 完成：波形/频谱切换、RAF 曲线、track canvas 与有界 presentation 最新值 |
| 编辑器 | Monaco、Python stubs/LSP | 浏览器 Monaco + 后端 LSP 会话 | 完成：本地 Monaco worker、临时 workspace/support files、持久 LSP completion/hover、结构化 diagnostics 与清理 |
| 项目与资产 | 项目、版本、组件、变体、导入导出、云同步 | 后端持久化和类型化 API | 本地范围完成：不可变资产版本、项目快照恢复、导入导出、component 捕获/插入、variant 应用；远端 Asset Cloud 是外部系统，按“无外部依赖”前提不纳入 P5 |
| 本地扩展 | template_match、viz_tcode | 显式后端注册与 TS renderer 注册 | 完成：Template Match 原图坐标 ROI/canvas PNG 写回；TCode 算子静态注册并由本地 TS renderer 渲染，无 CDN |
| 游戏/设备 | Unity/VaM、UDP、串口、外部进程 | 原生能力留后端，Web 配置与观察 | Linux 实现完成：Unity preview/显式确认、SDK 完整帧 UDP 验证、串口枚举、catalog allowlist 进程；真实游戏/串口待硬件验证，Unreal/VaM 无仓库自有 installer，保留显式 capability 门禁 |
| 快捷键 | Qt/OS 全局快捷键 | Web 局部快捷键 + 后端原生全局适配 | 完成：Web 局部快捷键；Inspector 字段绑定；SQLite 持久化；无 Qt Win32 `RegisterHotKey` worker 与 X11 grab/event backend；图提交后校验刷新；触发时原子提交 state、同步 runtime 并推送 graph event。当前 X11 主机经 XTEST 验证焦点外 grab/event；Windows 实机仍为发布门禁，无 DISPLAY 时明确 unavailable |
| Agent | provider、会话、工具、审批、图构建 | 统一 application service，无 Qt bridge | 完成：确定性/OpenAI/Anthropic/Gemini/Ollama 服务端注册、持久会话、工具/产物、取消、revision/hash/expiry 审批和 build/diagnose 闭环；真实供应商连接依凭据验证 |
| Web/API 壳 | 无独立产品入口 | loopback FastAPI、health/capabilities、React 工作区 | typed 图/项目/job/runtime/monitor API 与 WS 事件已建；React Flow 主工作区已接入 |

## 当前验证

```text
pixi run -e web-studio-test studio_core_test            21 passed
pixi run -e web-studio-test studio_media_gateway_test   23 passed
pixi run -e web-studio-test studio_server_test          65 passed
pixi run -e web-studio-test studio_web_test             30 passed
pixi run -e web-studio-test studio_web_e2e              34 passed, 4 skipped（desktop/mobile；service 鼠标缩放、runtime Inspector、全局快捷键配置与 1080p 延迟仅 desktop）
pixi run -e web-studio-test studio_graph_bench           1 passed（300/600 预算 + 1000/2000 压力曲线）
pixi run -e web-studio-test studio_python_typecheck     0 errors
pixi run -e web-studio-test studio_web_typecheck        passed
pixi run -e web-studio-test studio_no_qt_check          passed
pixi run -e web-studio-test studio_release_smoke        passed（6 个非 editable wheel；内嵌 Web bundle）
pixi run -e ci dist_ci                                  passed（Linux 原生服务、wheels、Web bundle、单文件启动器）
pixi run -e web-studio-test studio_dependency_probe     passed
pixi run -e web-studio studio_web_build                 passed
pixi run -e web-studio-test studio_media_bench          passed
pixi run pytest_sdk                                     262 passed
pixi run -e cpp cpp_test_release                         27 passed
pixi run -e web-studio-test studio_stream_restart_probe passed（0.372 秒恢复）
pixi run -e web-studio-test studio_real_video_probe     passed（C++ screencap，1920x1080）
pixi run -e web-studio studio_agent_model_smoke         skipped（未配置 `OPENAI_API_KEY`）
P3 SDK / detection targeted tests                       40 passed
pixi run -e web-studio npm --prefix packages/f8studio_web audit --json
                                                        0 vulnerabilities
```

尚未验证：Windows 全新构建树与 zip 解压安装、硬件编码、非 Chromium 浏览器和远端跨主机时钟。它们属于发布平台补充，不能据 Linux 软件编码结果宣称通过。

P7 当前 Linux 证据：旧 `packages/f8pystudio`、`f8pystudio_ext_template_match`、`f8pystudio_ext_viz_tcode` 及其 Qt 测试和脚本已删除；扩展能力已由 P5 的静态后端/前端注册实现。Unity 子模块删除独立 PySide/PyInstaller setup UI，保留 typed setup 包供 Web Studio 和 CLI 使用。Pixi 根清单、各 Python 包清单、子模块清单和 lockfile 均不再声明 Qt，视觉依赖使用 `opencv-contrib-python-headless`。`studio_no_qt_check` 会扫描保留 Python 源码的禁止 import、TOML 依赖、已安装 distribution、已加载 module、环境中的 Qt 动态库、旧包路径和应用构造边界。

`studio-runtime` 不包含 Node 工具链或旧 `studio` feature；启动器执行 `studio_server` 并以 `/api/health` 判断就绪。发布构建先生成本地 Web bundle，再将其装入 `f8studio-server` wheel。`studio_release_smoke` 在临时 venv 中非 editable 安装六个本地 wheel，确认本地包来自 `site-packages`、没有 Qt distribution、内嵌首页和 health 可访问。Windows 上启动器 dry run、wheel smoke 和 Win32 热键注册已通过；全新目录执行 zip 安装脚本及焦点外热键触发仍待验证。

P6 当前 Linux 证据：`StudioAutomationTools` 是浏览器、内置 Agent、HTTP、CLI 和 MCP 的统一业务边界；所有 patch 继续使用相同 revision、幂等和事务规则，并发布同一种 `graph.committed`。Playwright 在两个同时打开的页面中由 Agent 提交 patch，Graph 页面无需刷新即出现 service/operator；外部 API 的 graph-only 和 layout-only 新 revision 也按序同步，不重新应用旧事件。实际无浏览器 CLI 在独立数据目录完成建图、两次审批、校验、真实 `f8.pystudio` 部署和 monitor 读取；MCP 测试覆盖公开工具注册及 patch/approval 参数原样透传。新入口不导入 `f8pystudio`、Qt 或 GUI graph adapter。

Agent 审批记录绑定 `approvalId`、`toolCallId`、规范 JSON 的 SHA-256 参数 hash、目标 graph revision 和五分钟有效期；错误 hash、过期或并发 revision 变化不会执行工具。取消会同步关闭 pending approval/tool call，并明确说明已完成副作用不回滚；部署失败使 tool 与 run 失败并分别记录 traceback ID。catalog/graph/tool result 保存摘要，patch、最终 deployment 与当前项目 monitor 保存为可展开产物，避免把无关服务 telemetry 或重复 catalog 塞入会话。OpenAI、Anthropic、Gemini 和 Ollama 通过服务端环境变量配置，API 只返回 provider/model/configured；密钥响应测试和前端 bundle 审查通过。真实模型探针使用 `store=False` 的 OpenAI Responses 调用，但本机没有 `OPENAI_API_KEY`，因此仅记录 skip；确定性 provider 已覆盖完整工具闭环。

Provider 配置分别使用 `OPENAI_API_KEY` / `F8STUDIO_OPENAI_MODEL`、`ANTHROPIC_API_KEY` / `F8STUDIO_ANTHROPIC_MODEL`、`GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）/ `F8STUDIO_GEMINI_MODEL`，以及 `F8STUDIO_OLLAMA_MODEL`；对应 endpoint 均可用同名前缀的 `*_ENDPOINT` 覆盖。无头入口为 `pixi run -e web-studio studio_cli --url http://127.0.0.1:8260 ...` 和 `pixi run -e web-studio studio_mcp --studio-url http://127.0.0.1:8260`。

P5 当前 Linux 证据：Studio Server 新表与 API 在同一 SQLite 事务边界保存 local asset/current version、不可变 asset version 和 project snapshot；component/variant payload 在写入和导入时按 typed schema 校验，导出再导入保留完整版本历史，project restore 作为新 revision 提交。Assets 工作区可捕获整个项目 fragment、重映射 node/edge/service id 后原子插入，也可将 variant stateValues 应用到明确目标节点。Graph Inspector 的 `spec + ports` 编辑通过 `replaceNode` 进入后端图校验，不直接篡改浏览器投影。

Monaco、editor/json worker 和所有 renderer 均来自本地 Vite bundle。Python 会话创建隔离 workspace 和 support files，持有独立 `basedpyright-langserver --stdio` 生命周期并提供 completion/hover；保存使用严格递增 document version，显式 Analyze 返回结构化 diagnostics，关闭应用会回收进程和临时目录。服务测试实际请求了 completion、hover 和错误/正确代码 diagnostics。生产构建将 Monaco 保持为按需加载 chunk；当前压缩前约 2.26 MiB，属于后续加载体积优化项，不影响首屏主 bundle。

App 级 PresentationStore 统一持有 presentation WebSocket、快照和 32-node 上限，Graph、Live Outputs 与 3D 视图按节点订阅同一事实源。Video Viz 在 React Flow 节点中保留固定 16:9 区域，并与独立 Live Outputs 视图通过 source/quality 引用计数池共享 WebRTC peer/session；最后一个消费者释放后关闭浏览器 peer 和 Gateway session，视频帧不进入 React state。真实 C++ screencap 验证得到 640x360、`readyState=4` 的节点内播放；移动节点前后 `<video>` DOM 与 `MediaStream` 对象保持相同且只有一次媒体协商，切换到无预览项目后 Gateway session 计数恢复基线。Web renderer 另覆盖 text、wave、track 与 TCode；音频页增加 AnalyserNode 频谱。`f8.viz.tcode` 是第 14 个静态内置算子，旧插件 loader 和 CDN `osr-emu` 均不参与。Template Match 直接调用 runtime command，在浏览器按实际 image bounds 将 pointer ROI 换算到原图像素并用 canvas 编码 PNG，再通过 runtime state API 写回 `templateImagePngB64`。

本机集成遵守 detect -> preview exact writes -> `confirm=true` -> apply；没有在自动测试中对真实游戏目录执行写入。UDP verifier 绑定默认端口 39540，并以 `f8pysdk.motion.decode_skeleton_datagram` 成功解码完整 `modelName + bones` 帧为通过条件，随机 datagram 不计成功。TCode/serial 物理输出没有由 P5 UI 自动 arm，图模板仍要求 250 ms watchdog。全局快捷键保留旧 Control Panel 的 button 递增与 select 下一项语义：binding 指向明确的 project/node/field，OS 线程经主 asyncio loop 提交权威 graph revision，并尝试同步已部署 runtime；原生后端 unavailable 或字段删除/被上游 state 驱动时显示 disabled/error，不注册无效按键。伪 Win32/native backend 测试覆盖规范化、键码、注册、注销、持久化和真实 graph/runtime 触发，Playwright 覆盖 Inspector 保存及重开。当前 `DISPLAY=:0` X11 主机还以 XTEST 向失焦窗口注入 `Ctrl+Alt+Shift+P`，实际 root passive grab 收到事件；该探针发现并修复了 `owner_events` 与 `array.array` modifier mapping/Num Lock 两个问题。Windows 桌面、真实 Unity/VaM/Unreal 目标和串口设备仍必须在相应发布机上验证，不能用 Linux capability 状态替代。

P3.5 当前 Linux 证据：正式 `studio_server` CLI 在 PID 2132966 启动受管 `f8media_gateway` PID 2132996，分别监听 `127.0.0.1:8260` 与 `127.0.0.1:8261`。`/api/media/gateway` 返回 `f8media-api/1`、独立 gateway epoch 和子进程 PID。经 Studio 原 URL 代理，synthetic WebRTC 解码出变化的 640x360 帧；真实 C++ screencap -> Zenoh v2 -> Gateway -> WebRTC 解码出变化的 1920x1080 main 帧。中断 Studio 后两个 PID 与两个监听端口均消失。单元/集成测试另覆盖网关资源清理、422/404 透传、断连 503、协议错配、端口占用时 PID 所有权校验和 managed process 回收。

里程碑审查进一步补齐了异常关闭与协商竞态：Studio 与受管 Gateway 之间保留 stdin ownership pipe，父进程异常退出时 Gateway 自动终止；快速 Stop 使用 operation token 阻止过期协商回写 UI 或遗留 session；处于 ICE checking 的 peer 在逻辑 session/source 立即释放后延迟到底层 STUN transaction 结束再关闭，避免 aioice 0.10.2 的 retry-after-close 未处理异常。该取消场景在 desktop/mobile 连续 6 次压力重复和完整 E2E 中均通过，资源计数回到零。Studio 构造路径也已验证不导入 `f8media_gateway`、aiortc 或 PyAV。

Zenoh envelope 的通用二进制解码函数仍属于 SDK 的 `f8pysdk/video_transport.py`，便于 Python publisher/subscriber 共用协议；调用它的 subscriber、latest-frame hub、BGRA/FLOW/SCALAR 转换、overlay、WebRTC track、peer/session 和 aiortc 编码执行均位于 `f8media_gateway`。一个网关进程管理所有 source/viewer；同 source 共享一次 Zenoh subscriber 和部分预处理，但每个 `RTCRtpSender` 当前仍创建自己的编码器。因此该拆分隔离了 Studio 的 GIL、事件循环和故障域，并提供未来 Rust/C++ 网关替换边界，没有消除同源多 viewer 的重复编码。

P3 当前 Linux 证据：真实 Uvicorn HTTP 信令可建立 aiortc peer，`synthetic://bars-1080p` 解码为变化的 1920x1080 帧；受管 `f8.screencap` 经项目部署 arm 后，真实链路 `screencap -> Zenoh v2 latest-frame -> WebRTC` 同样解码为变化的 1920x1080 帧，探针结束后服务进程被回收。BGRA padded stride、动态分辨率、源时间戳回退、同源多视图共享、不同源隔离、关闭重开和 application shutdown 均有测试。FLOW2_F16 使用 HSV 方向/20.0 magnitude scale，SCALAR1_F32 使用 2/98 百分位和 turbo 色图；NaN/Inf 在预览中转黑，精确查询返回 `null + finite=false`，并保留 source frameId、stream epoch 和时间戳。

视频帧协议 v2 由 Python/C++ publisher 在每次 producer 启动时生成 128 位 `streamEpoch`。Python/C++ describe metadata、payload schema version 和本地 catalog 均已核对为 v2，`streamEpoch` 为必填字段；音频协议保持 v1。检测输出携带 `streamId + streamEpoch + frameId + tsMs`；overlay store 以完整身份和 capture timestamp 为 key，只在 50 ms 等待预算内合成匹配的 BGRA 帧。未匹配结果隐藏，过期/拒绝/超时通过 `/api/media/metrics` 监控，store 上限 256 条/2 秒，raw history 上限 8 帧/300 ms。专项测试覆盖精确绘制、frame/epoch/timestamp 不匹配、超时、过期和有界内存。

音频使用独立 typed WebRTC 会话，支持 `synthetic://tone`、`synthetic://tone-stereo` 和真实 `f8/` Zenoh F32LE source；当前严格接收 48 kHz mono/stereo 并显式转换为 WebRTC PCM。浏览器必须由 Play 手势启动 AudioContext，提供音量、静音、停止和 RAF/AnalyserNode 波形。Chrome E2E 证明收到非静音音轨且波形非平坦。Zenoh transport 是 latest-chunk：慢消费者可能丢块，sequence gap 会计数；当前不承诺 gapless 播放或音视频同步。

`studio_media_bench` 最新结果为 60.00 秒的 1 路 1920x1080 main + 4 路 thumbnail：main 28.80 FPS，thumbnail 7.88-7.95 FPS，进程 CPU 177.8%。负载 RSS 889.9 MiB；释放后的热基线 725.0 MiB，随后 50 次 SDP 连接/关闭为 704.5 MiB，session/source 每轮回到零。单独媒体场景达到 main ≥28 FPS；thumbnail 受每 peer 软件编码限制，低于 10 FPS 档位上限。

组合 5 分钟场景额外包含一条音频：main 26.86 FPS、四路 thumbnail 7.69-7.73 FPS、CPU 191.1%、控制响应 p95 47.38 ms、错误 0、资源计数归零。30 分钟稳定性场景按退出条件运行 main + 4 thumbnails + 10 Hz 3D 骨架事件 + 2 Hz graph/monitor HTTP 控制：main 27.52 FPS、thumbnail 7.66-7.69 FPS、CPU 177.5%、3,474 次控制请求 p95 43.18 ms、16,877 条 presentation 事件、队列峰值 1、错误 0。任务数负载态 42、峰值 58、清理后 2；RSS 活跃样本约 500-693 MiB，最后 5 分钟净增 19.9 MiB（门槛 50 MiB），清理后 522.4 MiB；video/audio session/source 全部归零。原始证据位于 `docs/plans/evidence/p3-combined-5m.json` 与 `docs/plans/evidence/p3-stability-30m.json`。组合 main 未达 28 FPS，已定位为多 peer 软件编码瓶颈；不宣称媒体性能完全对等。

Playwright 使用本机 Google Chrome 在 1440x900 与 390x844 视口验证：视频元素像素随帧变化、点击画面可查询带 frameId 的 latest raw 数值、Three.js framebuffer 非空、轨道拖拽改变画面、WebRTC 音频波形非平坦、布局无横向/纵向溢出，且页面无未处理错误。Graph Editor 用例验证 operator 以 React Flow `parentId` 嵌套在 service 中、拖入兼容容器会原子更新 binding/layout、移动 service 会对其 operator 做严格相同的绝对位移、选中 service 后可从四边和四角缩放且宽高会持久化、缩放后子节点仍在容器内且 viewport 不重置、删除 service 会级联删除 operator、inline slider 会持久化 typed state，以及 exec/data 连线和 data queue policy 会写入项目文档。RW state 行只显示一次名称并将释放的宽度用于 editor；接入上游 state 后保持原 input/select/checkbox DOM 与对齐，仅切换为锁定状态。内置 Value Stepper 用例从空项目开始，原子创建隐藏的 `studio` service 和 operator，完成 state 配置、部署、Ready monitor、再次修改、draft/deployed revision 分离及刷新恢复。无 operator 子节点的 service 固定按自身可见端口收缩到 280 px，容器 service 默认 524 px，operator 为 240 px；标题/端口行压缩到 34/24 px，同时保留端口 24 px 交互热区。compact code/wrapline/JSON 控件使用单行输入和内部截断，长文本不再参与节点 intrinsic width。desktop/mobile 截图显示 state input/control/output 同排、长表达式不撑宽节点且节点没有重叠。无效连接规则由前端纯函数测试与后端 mutation 校验覆盖；移动端 typed connection 连续 5 次压力重复通过；空白区拖放回滚当前没有稳定的浏览器手势用例。1080p main 使用帧内 16 位 capture clock marker 和 `requestVideoFrameCallback` 测得 60 个显示帧最新 p95 75 ms；localhost 浏览器和 producer 共用系统时钟，未使用 HTTP ping 代替视频延迟。截图保存在测试产物目录；Three.js 和图标均从本地 bundle 加载，无 CDN。

P4 复杂图基准从真实 catalog 创建当前可见行数最多的 `f8.pyengine/f8.handy_out`（每节点 18 行），通过真实 HTTP patch 构图，再由正式 GraphWorkspace 加载和交互。300 节点/600 边场景：批量提交 755.1 ms、首次可交互 1.680 s、拖拽 60.0 FPS、帧时间 p95 16.8 ms、UI 交互 p95 26.2 ms、GC 后 JS heap 24.6 MiB；1000 节点/2000 边压力场景：提交 3.276 s、首次可交互 4.589 s、拖拽 59.9 FPS、UI 交互 p95 26.5 ms、heap 46.6 MiB。两种场景均只挂载视口内 25 个节点和 52 条边。1000 节点拖拽后的完整文档持久化确认耗时 3.371 s，作为压力曲线记录，不属于即时拖拽反馈预算。原始证据位于 `docs/plans/evidence/p4-graph-performance.json`。

真实 Zenoh publisher 关闭并重开后，producer epoch 改变，现有 WebRTC session/source 保持单实例并在 0.372 秒恢复出帧；结束后资源计数归零。该结果满足 5 秒恢复预算，且没有重新执行图操作。

真实 P2 探针已在本机通过 Zenoh 启动独立 `web-studio-runtime` 环境中的 `f8.pyengine`，经 HTTP 创建 `f8.tick -> exec -> f8.print` 图并部署，确认 runtime graph revision、状态写入、deactivate/activate 与 monitor `processed=15`。探针停止服务后无残留进程；另在服务仍运行时中断 Web Studio，应用约 1.5 秒完成关闭且 supervisor/PyEngine 均被回收。该证据覆盖当前 Linux 主机，不替代 Windows 验证。

后端现在内置独立的 `f8.pystudio` ServiceRuntime，catalog 暴露 13 个已迁移算子：text、wave、video、audio、track、3D、data/state expr、control panel、backdrop、note、patch hub 和 value stepper。表达式求值使用 AST allowlist；精简环境未安装 NumPy 时 `allowNumpy` 返回明确 monitor 错误。Track 和 3D 输入通过显式 `msgspec.Struct` 边界规范化，不复制旧 Qt renderer。每个应用实例创建独立 registry 与 `EventPresentationOutlet`，不使用旧全局 UI command sink；presentation 与 monitor 都作为不可靠实时事件，队列满时丢弃，不挤占可靠 graph/job 事件的 replay 语义。新运行链路未导入 `f8pystudio`。

`probe_studio_presentation.py` 已通过真实 Uvicorn、Zenoh 和 WebSocket：经 HTTP 创建并部署仅包含内置 `studio` 服务与 `f8.viz.three_d` 的项目，写入 `worldUp=-z` 后收到 `presentation.command / viz.three_d.world_up`。精简 server 现在明确依赖 `websockets`，因此生产 `/api/events` 不再依赖环境中偶然存在的 Uvicorn extra。部分部署测试固定了一个服务成功、另一个 Zenoh 断连时的 `partially_failed` 和逐服务结果；runtime 请求断连固定映射为 HTTP 503。

P1 测试覆盖文档编解码与新格式样例、真实 `services/f8/engine/describe.json` 目录、data/state/exec/command、动态 spec、patch hub、跨服务 half-edge、自动采样、组件 fragment、禁用节点、payload 不匹配、状态环、原子回滚、revision 冲突、幂等及 undo/redo。state mutation 现在按权威 value schema 递归校验类型、enum、数值边界、multipleOf、数组 items、对象 required/properties/additionalProperties，前端 JSON 可序列化不再是唯一约束。语义 hash 不含布局和事务 revision，并对节点、边及 JSON mapping 的插入顺序稳定。

项目 mutation 与 request id 幂等记录已在同一 SQLite 事务提交，默认每项目保留 2048 条，并已验证进程重启后重放与冲突。幂等重放不重复发布 `graph.committed`。undo/redo 历史仍属于当前进程中的项目 session；持久版本历史属于后续项目/资产工作。每次提交都要求精确 graph/layout revision，因此过期客户端不能覆盖新提交。

## 已发现的仓库前置问题

旧 `default` 环境因 `external/f8unitymods` 缺少 `pyproject.toml/setup.py` 无法求解。新 `web-studio` 环境不引用该 feature，因此可以独立工作。modding 功能仍在迁移范围内；不得通过空包掩盖缺失源码。

## 标准命令

`studio_server` 对同一操作系统用户实行单实例锁，与监听端口和 `F8STUDIO_DATA_DIR` 无关。第二个服务端启动会明确失败；媒体网关可作为独立进程运行。旧版本服务端需重启后才参与此锁。

```bash
pixi run -e web-studio studio_server
# 同时支持 SSH loopback 转发与受信任 VPN/LAN 直连：
pixi run -e web-studio studio_server --host 0.0.0.0 --allowed-host 164.107.57.10 --port 8260
# 默认受管网关为 127.0.0.1:8211；也可单独启动并让 Studio 连接：
pixi run -e web-studio studio_media_gateway --port 8211
pixi run -e web-studio studio_server --media-gateway-url http://127.0.0.1:8211 --external-media-gateway
pixi run -e web-studio studio_web_dev
pixi run -e web-studio studio_web_build
pixi run -e web-studio-test studio_core_test
pixi run -e web-studio-test studio_media_gateway_test
pixi run -e web-studio-test studio_server_test
pixi run -e web-studio-test studio_web_test
pixi run -e web-studio-test studio_web_e2e
pixi run -e web-studio-test studio_graph_bench
pixi run -e web-studio-test studio_python_typecheck
pixi run -e web-studio-test studio_web_typecheck
pixi run -e web-studio-test studio_no_qt_check
pixi run -e web-studio-test studio_dependency_probe
pixi run pytest_sdk
pixi run -e web-studio-test studio_media_bench
pixi run -e web-studio-test studio_stream_restart_probe
pixi run -e web-studio-test studio_p3_combined_bench
# 30 分钟稳定性与证据文件：
pixi run -e web-studio-test python scripts/web_studio/benchmark_p3_combined.py --duration 1800 --output docs/plans/evidence/p3-stability-30m.json
# 另一个终端先运行：pixi run -e web-studio studio_server --port 8231
pixi run -e web-studio-test studio_runtime_probe
# 对已启动的 Web Studio 验证确定性与真实采集视频：
pixi run -e web-studio-test studio_media_probe --base-url http://127.0.0.1:8231
pixi run -e web-studio-test studio_real_video_probe --base-url http://127.0.0.1:8231
# 同一服务也可验证内置 Studio graph 与 presentation WebSocket：
pixi run -e web-studio-test python scripts/web_studio/probe_studio_presentation.py --base-url http://127.0.0.1:8231
```

VPN/LAN 直连模式必须用 `--allowed-host` 显式列出可信地址；默认仍只监听 loopback。通配 bind 不会通配 HTTP Host/Origin。Gateway 控制端口保持 loopback。WebRTC 媒体不经过 Studio HTTP 端口，客户端还必须能访问 Gateway SDP 中公布的动态 UDP ICE candidate；若 VPN 或防火墙不允许该流量，仍需 TURN/TCP/TLS 或明确的 UDP 端口策略。

### 严格 VPN / SSH 模式

当 VPN 只允许 SSH 且丢弃 TCP 8260 和动态 UDP 时，使用 loopback TURN/TCP。浏览器通过第二条 SSH local forward 连接 TURN；TURN 在服务器本机用 UDP relay 与 Media Gateway 通信，因此公网和 VPN 都不需要开放媒体端口。

服务器侧 coturn 必须只监听 loopback，并允许 loopback peer。当前验证配置监听 `127.0.0.1:3478/TCP`，relay 端口为 `49160-49200`：

```bash
docker run --rm -d --name f8studio-turn --network host coturn/coturn:4.6.3 \
  -n --log-file=stdout \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 --listening-port=3478 \
  --min-port=49160 --max-port=49200 \
  --lt-cred-mech --user=studio:<credential> --realm=f8studio.local \
  --no-cli --no-tls --no-dtls --allow-loopback-peers --no-multicast-peers

pixi run -e web-studio studio_server \
  --host 127.0.0.1 --port 8260 \
  --media-gateway-url http://127.0.0.1:8261 \
  --turn-url 'turn:localhost:3478?transport=tcp' \
  --turn-username studio --turn-credential '<credential>' --force-turn
```

Windows 客户端通过同一个 SSH 连接转发两个 TCP 端口：

```powershell
ssh -N `
  -L 8260:127.0.0.1:8260 `
  -L 3478:127.0.0.1:3478 `
  <user>@164.107.57.10
```

浏览器从 `/api/media/rtc-configuration` 读取运行时 ICE 配置。启用 `--force-turn` 后使用 `iceTransportPolicy=relay`，并在发送 offer 前等待 ICE gathering 完成；这使非 trickle TURN candidate 确实进入 SDP。2026-09-22 使用真实 Chrome 强制 TURN/TCP 验证 synthetic 视频、音频和波形均通过，coturn allocation 计数证明媒体走 relay。loopback TURN 模式依赖 SSH 身份验证，只适合受控远程开发。Studio 在此模式下也保持 loopback bind；Host allowlist 不是身份验证，不应为了 SSH 转发绑定公网接口。面向多用户部署应使用 TURN/TLS 443、短期凭据、正式证书和应用层身份验证。
