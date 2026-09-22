# Web Studio 全量迁移实施方案

日期：2026-09-21。执行对象：GPT 5.6 Sol。状态：实施前设计，尚未完成性能验证。

## 1. 任务与授权边界

将 Studio 前端从 Python Qt 迁移到 Web，并最终移除 Studio 对 PySide6、QtPy、NodeGraphQt、PyQtGraph、QtWebEngine 的依赖。保留现有 Python/C++ 计算、采集、设备、游戏接入能力，通过无界面本机后端提供给浏览器。

用户明确允许重构全部本地代码，不需要照顾外部框架、第三方插件消费者或旧接口兼容。可以修改包名、API、项目格式、插件注册方式和内部协议，并同步更新所有仓库内调用者。不要构建长期兼容层、伪 Qt 对象、旧类名别名或双轨业务实现。

这里的“不受外部依赖约束”指没有必须维持兼容的外部消费者，不是禁止使用 React、Three.js 等依赖。正常使用并锁定必要依赖；资源打包到本地，核心工作流不依赖 CDN。

本次交付是 Web Studio；不是把 Python/C++ 引擎翻译为 JavaScript，也不是立即建设云端多租户产品。浏览器、CLI、MCP、内置 Agent 最终调用同一套业务能力。

数据格式允许破坏性升级，但这不等于可以删除用户现有文件、数据库、凭据或游戏安装。开发使用独立数据目录；旧数据给出明确“不支持此版本”诊断，确有需要时编写一次性离线转换工具，不保留运行时旧格式分支。同步重建仓库内样例和资源。

## 2. 已核对的仓库事实

以下是设计依据，不应把旧 README 中不存在的目录当作实现入口。

| 当前入口 | 已确认事实 | 处理方向 |
| --- | --- | --- |
| `schemas/protocol.yml` | 现有 OpenAPI 3.0.3 协议源 | 复用服务与运行时概念，新增 Studio 文档/API 契约 |
| `packages/f8pysdk/f8pysdk/video_transport.py` | Zenoh latest-frame；BGRA32、FLOW2_F16、SCALAR1_F32；frame_id、ts_ms | 保留计算数据，增加媒体出口 |
| `packages/f8pystudio/f8pystudio/render_nodes/video_preview.py` | 内嵌预览限频约 10 FPS、最大 640×360 | Web 继续分级订阅 |
| `render_nodes/viz_three_d.py`、`render_nodes/web_assets/viz_three_d/index.html` | QtWebEngine 包装 Three.js | 提取场景逻辑，打包本地模块 |
| `ui/dialogs/monaco_editor_dialog.py`、`ui/support/monaco_editor_page_html.py` | Monaco 经 QtWebChannel 通信 | 前端组件 + 后端 LSP 会话 |
| `bridge/studio_bridge.py`、`bridge/facade_qt.py` | QObject、Signal 与服务编排耦合 | 显式应用服务与类型化事件 |
| `bridge/studio_service.py`、`operators/` | Studio 还承载运行时算子及 UI command sink | 算子迁入后端，呈现事件独立建模 |
| `nodegraph/runtime_compiler.py` | 编译器读取 NodeGraphQt 对象 | 改为编译纯数据文档 |
| `automation/graph_adapter.py` | 读取 GUI 图、Qt undo 命令 | GraphStore + 原子 patch + 历史记录 |
| `agents/tools/graph.py`、`agents/qt_bridge.py` | Agent 工具路径依赖 GUI/Qt | Agent 直接调用应用服务 |
| `plugins/`、`packages/f8pystudio_ext_*` | 本地插件区分 renderer/operator | 改为显式注册，本地扩展一起迁移 |
| `assets/`、`editor_assist/`、`modding/`、`global_hotkeys/` | UI 以外仍有大量本机业务 | 保留能力、移除 UI 依赖 |
| `pixi.toml` | Python 3.14；win-64/linux-64；已有 aiortc、websockets 依赖 | 先验证可安装与运行，不假定已有媒体实现 |
| `packages/f8assetcloud_worker/console_web/package.json` | 已有 React/Vite Web 项目 | 可参考工程习惯，不将 Studio 塞进资产云控制台 |

除首行明确给出全路径的独立包外，表中的相对 Python 路径位于 `packages/f8pystudio/f8pystudio/`。已搜索的代码没有实际使用 RTCPeerConnection/VideoStreamTrack，Pixi 声明依赖不代表 WebRTC 已实现。

撰写方案时实际执行 `pixi run -e default python scripts/check_docs_links.py`，在运行脚本前即因 `external/f8unitymods` 缺少 `pyproject.toml/setup.py` 而依赖求解失败。这是当前检出状态的前置问题，不是迁移代码回归。P0 必须检查该目录的来源和完整性：恢复仓库所需源码，或在不丢失 modding 能力的前提下将其拆为独立 feature。不要创建空包伪装依赖已满足；新环境也要验证 workspace 求解是否仍受它影响。

## 3. 已确定的技术方向

| 层 | 默认选择 | 约束 |
| --- | --- | --- |
| Web 前端 | React + TypeScript strict + Vite | 独立包；固定锁文件；核心资源离线可用 |
| 节点画布 | React Flow | 先用真实复杂节点验证；不是业务模型或执行引擎 |
| 3D | Three.js + WebGL | 先迁移已有能力，WebGPU 后续可选 |
| 代码编辑 | Monaco | Worker 本地打包；Python 运行和 LSP 留在后端 |
| HTTP/WS | Python FastAPI + Uvicorn | 验证 Python 3.14 依赖；应用逻辑不绑定框架 |
| 主预览媒体 | WebRTC；初版 aiortc/PyAV | 不承诺自动硬件编码；按实际测量决定 C++/FFmpeg 媒体进程 |
| 内部通信 | 现有 Zenoh | 浏览器不直接订阅完整运行时总线 |
| 测试 | pytest、Vitest、Playwright | 真实浏览器/真实 Zenoh 验证不能被 mock 替代 |
| 发布形态 | 本机后端 + 浏览器 | 首轮不引入 Electron/Tauri；后续另做桌面壳 ADR |

建议新建：

```text
packages/f8studio_core/              # 纯领域模型、图变更、编译器、目录契约
  f8studio_core/{graph,catalog,compiler,contracts}/
packages/f8media_protocol/           # 无 aiortc/PyAV 的版本化模型、Protocol 与 typed HTTP client
  f8media_protocol/{client,contracts,models}.py
packages/f8media_gateway/            # 独立 Zenoh -> WebRTC 媒体进程
  f8media_gateway/{app,service,media,audio_media,overlay}.py
packages/f8studio_server/            # 应用服务、存储、运行时、HTTP/WS、Agent 与媒体代理
  f8studio_server/{api,application,persistence,runtime,agents}/
packages/f8studio_web/               # React/TS
  src/{app,api,graph,panels,media,three,editor,agents}/
schemas/studio/                     # 新文档/API/事件 schema；明确引用运行时 schema
scripts/web_studio/                 # 契约生成、基准、验证、发布辅助
docs/plans/web-studio-status.md      # 实施时创建：阶段、证据、限制、下一步
```

按领域需要增加子目录，不预先创建空壳体系。已有模块在明确归属后直接移动/重构，不复制整个 `f8pystudio`。最小依赖方向：

```text
web -> HTTP/WS signaling -> server -> core -> SDK 契约/验证
  |                           |
  |                           +-> storage / process / LSP / Agent providers
  |
  +-> WebRTC media <-> media gateway <-> SDK/Zenoh <-> Python/C++ services
```

`core` 不导入 server、Qt、GUI、HTTP 框架，也不启动进程或写文件。server 不导入旧 Studio UI。浏览器不负责执行实时算子、设备控制时序或持有模型提供方密钥。

媒体网关使用版本化的 `f8media-api/1` typed HTTP 边界；模型、Protocol 和 async client 位于不依赖 aiortc/PyAV 的 `f8media_protocol`。正式 Studio CLI 默认管理一个独立网关子进程，HTTP 只代理信令、overlay、精确采样和监控；RTP/ICE 直接在浏览器与网关之间流动。测试和基准可显式注入进程内实现。一个网关管理多个 source 和 peer，不按窗口启动进程；同 source 共享 Zenoh subscriber 和可复用帧预处理，但当前 aiortc sender 仍为每个 peer 独立编码，进程隔离本身不等于共享压缩码流。

## 4. 图模型：优先解决的核心问题

### 4.1 统一文档与状态所有权

定义 `StudioDocument`，具有明确版本、projectId、graph、layout。graph 至少包含：

- 服务实例、算子节点、稳定 ID、所属服务实例；服务容器显示与服务身份分开。
- 显式端口 ID、端口种类、方向、值 schema；不从 `[D]`、`[E]` 等显示字符串猜语义。
- 边 kind、源/目标、策略、队列和超时配置；命令与 state 的运行时适配放编译阶段。
- 可编辑 spec 覆盖、配置值、组件/变体引用或展开结果及其明确规则。
- layout 中保存位置、尺寸、折叠、分组、注释等；选择、hover、拖拽中间态是客户端临时状态。

明确区分：持久配置、图草稿、已部署图、远端权威运行状态、监控采样。不要把服务读回状态和 UI 初始配置混写成一个字段。布局变动不触发部署；使用 graphRevision/layoutRevision 或等价的语义分类。

后端 GraphStore 是业务图权威来源。前端可有乐观投影，但服务端拒绝后必须正确回滚/重同步；不能把 React Flow 序列化结果直接用作运行时图。

### 4.2 Patch、事务、历史

定义判别联合类型 `GraphOperation`，至少支持 create/delete node、connect/disconnect、set config、edit ports/spec、rename、bind service、component insertion。布局操作单独分类。Python 使用显式 Struct/dataclass，TS 使用判别联合；仅真正动态用户数据允许 JSON value。

请求含 `requestId`、`expectedRevision`、操作列表。后端流程：

1. 校验 envelope/schema、目标 ID、权限/可写性和版本。
2. 在临时候选文档应用全部操作；对最终结果做完整图校验。
3. 全部成功才原子提交、递增 revision、生成撤销记录和持久化结果。
4. 返回新 revision 与变更；广播同一提交事件。
5. 同 requestId 同内容重试返回已记录结果；同 ID 不同内容明确拒绝。幂等记录作用域、保留时间和重启语义必须写进契约。

错误至少区分 revision conflict、invalid edge、missing node、invalid schema、read-only field、unsupported document version。失败不得留下半个图或半段历史。

Undo/redo 作用于文档事务，不假装撤销已经发出的硬件命令或进程启动。多人/多标签首版采用后端串行提交和版本冲突，暂不实现 CRDT。远端提交使旧撤销前提失效时拒绝并解释，不能覆盖别人更新。

### 4.3 编译与部署

将 compiler 改为 `compile_document(document, catalog) -> CompiledRuntimeGraphs` 一类纯入口。不传 NodeGraphQt、QWidget 或模仿其方法的对象。

保留并验证 data/state/exec 连接规则、服务归属、命令端口映射、默认值、自动采样需求、patch hub 等实际语义。可删除旧序列化细节；不能因“不兼容”而丢掉运行能力。

使用规范化语义 hash 确定部署内容；同文档编译稳定，无随机 ID 漂移。部署记录 source revision/hash、jobId、各服务结果；部分失败明确表示，不能伪装成全局原子成功。重复部署须去重；部署期间草稿继续修改时，完成结果仍对应原 revision。

## 5. 后端与浏览器契约

以下是实现接口清单，不要求机械照搬 URL，但语义必须具备并在生成契约中固定。

| 接口族 | 最小职责 |
| --- | --- |
| `/api/health`、`/api/capabilities` | 存活、就绪、协议版本、实际媒体/设备能力 |
| `/api/catalog` | 服务/算子/spec/renderers 静态目录 |
| `/api/projects`、项目 document | 创建、加载、保存、列表；隔离测试数据目录 |
| 项目 patch/validate/undo/redo | 权威图变更与检查 |
| 项目 deploy、`/api/jobs/{id}` | 异步部署、进度、结果与可取消阶段 |
| runtime services/commands/state | 启停、命令、状态写入；区分接受与真正完成 |
| `/api/events` WebSocket | 图提交、服务状态、日志、监控、呈现数据 |
| media sessions/signaling | 订阅授权、SDP/ICE、关闭、质量档位、能力查询 |
| editor sessions | LSP 消息、虚拟文件、诊断、会话关闭 |
| agent sessions/runs | 消息、工具进度、取消、审批、产物与历史 |

所有接口与事件有版本和明确类型。生成 TS 类型及运行时校验器，Python 侧有相同契约测试。选定一种生成流程，不能手工维护三份重复接口。复用 `schemas/protocol.yml` 的模型时显式引用/生成，不修改现有 SDK codegen 使其悄悄改变运行时协议。

事件 envelope 至少包含 eventId、serverEpoch、sequence、type、scope、timestamp、payload。可靠事件与瞬时数据区别处理：

- 图提交/作业终态：可恢复；短期重放，超出保留窗口明确要求重新拉取 snapshot。
- 日志：有界保留并声明缺口；监控与骨架：合并最新值，不堆积。
- 订阅必须明确 topic/scope；每客户端有限队列，慢客户端不能阻塞其他客户端或 Zenoh 回调。
- 重连采用 snapshot + 游标衔接，处理取 snapshot 期间发生的新事件；epoch 变化清空旧序号预期。
- 控制队列不与视频帧共用无界队列；视频不进入 JSON 事件通道。

Python 一个明确的 asyncio 生命周期管理任务；同步/CPU 密集工作进入受控线程或进程。Zenoh 回调通过安全队列转入事件循环，禁止阻塞总线线程。

默认同源、仅监听 loopback；Vite 开发代理保持统一接口。受信任 VPN/LAN 调试可绑定具体接口，或使用通配 bind 同时服务 loopback 和 VPN；所有非 loopback Host 必须通过 `--allowed-host` 显式列出，通配 bind 不得转化为通配 Origin/Host。由于 API 能启动进程和改文件，必须验证 Origin/Host、使用本机会话鉴权并避免通配 CORS。凭据留在系统 keyring/后端。外网模式另行显式配置 TLS、身份认证、访问范围；不把 localhost 安全假设用于远程。HTTP 可达不代表 WebRTC 可达：VPN、防火墙或 TURN 还必须承载 ICE candidate 的动态 UDP/TCP 媒体路径。

服务生命周期默认不绑定标签页：刷新、关闭页面不停止运行图。只有明确 stop 或后端退出策略才终止受管服务；非受管外部服务不能误杀。后端退出关闭自有媒体、LSP、订阅、任务和受管子进程，记录失败。重启时重新发现/核对所有权，不盲目重复启动。

## 6. 视频、音频与数值可视化

### 6.1 最小媒体链路

第一版使用真实 SDK latest-frame reader -> 有界最新帧槽 -> aiortc/PyAV -> 浏览器 `<video>`。HTTP 提供信令，落实 ICE 候选交换或明确的非 trickle 协议。能力探测实际测试编解码器，支持失败必须返回可见错误。

适配 BGRA 的 width/height/pitch、非连续行、buffer 生命周期、颜色转换。明确 WebRTC frame pts/time_base 的单调映射；丢帧不使时间倒退。消费完释放原始帧；没有订阅者时停止额外预览工作。禁止每来一帧就创建 task、无界 queue 或累积完整视频。

订阅以 source + 质量档位组织。相同来源读帧、缩放能共享就共享，但不能声称 aiortc 默认跨 peer 共享编码器。测量多 peer CPU 后再决定是否升级媒体架构。

质量档位初始为 thumbnail（≤640×360、≤10 FPS）和 main（目标 1080p30）；源不足时准确报告实际尺寸/FPS。不可见缩略图取消或降低订阅；tab 隐藏时降低呈现负载，后端计算继续。

主视频不用 HLS 作为低延迟默认方案。JPEG/MJPEG 或 WebSocket 图像可用于诊断，但不能用它们的成功替代 WebRTC 验收。WebCodecs 仅在明确需要自行控制帧映射/解码时引入，它本身不是传输协议。

### 6.2 同步与 overlay

不能直接按“最新骨架/框”叠加视频。新增源标识 streamId、streamEpoch、frameId、captureTimestamp；处理结果保留来源帧身份。流重启不能仅靠重复 frameId 判断新旧。

现有 ts_ms 不等于端到端统一时钟。localhost 基准采用可校准时钟/帧内标记；远端延迟须计入时钟误差。WebRTC RTP/媒体时间和 SDK frameId 必须定义映射，不能假设浏览器可直接拿到自定义帧 ID。

分两步实现：

1. 第一版需要精确叠加时，在媒体网关按同源帧匹配检测结果后合成视频；定义等待预算和超时丢弃策略。独立 3D 仍通过 data channel/WS 更新。
2. 需要客户端可编辑 overlay 时，用 `requestVideoFrameCallback`、可用 RTP/媒体时间信息与网关映射做目标浏览器验证。API 信息不足时继续使用服务端合成，或另立 WebCodecs 方案，不承诺未经验证的逐帧精确叠加。

UI 明确区分实时近似显示、精确配对显示和暂停帧检查。没有匹配结果时隐藏/标注过期，不能悄悄误配。

### 6.3 光流、标量、音频

FLOW2_F16 与 SCALAR1_F32 不是普通图像。首版在后端按显式 range/colormap/NaN 规则生成预览；精确采样通过独立二进制数值接口获取并标注 frameId。不得对有损编码视频取像素后宣称是原始数值。

如需前端 GPU 数值渲染，定义 dtype、端序、shape、stride、范围、有效值与 frameId，验证已知数值 fixture；不要同时支持两套无人使用的实现。

迁移音频播放、波形和频谱：播放可走 WebRTC 音轨；波形/频谱传降采样数据，绘图与 React 状态分离。处理浏览器自动播放限制（用户手势启用）、静音、设备选择与断流；明确音视频时钟和同步策略。

### 6.4 性能升级决策

aiortc 是验证起点，不是预定最终性能答案。若 1080p30 或多路目标未达标，记录解码/拷贝/转换/编码/传输/显示分段时间后，只替换瓶颈：预先降采样、降低复制、独立媒体进程、C++/FFmpeg/GStreamer 硬件编码等。选硬件路径需验证 Windows/Linux 的实际编码器和回退行为，不能仅以机器有 GPU 判定通过。

## 7. 前端功能与本地扩展迁移

前端 store 分开管理持久图、编辑瞬态、运行状态、监控缓存。视频帧不进 React store；高频骨架/曲线通过有界缓冲和 RAF 消费。规范化节点/边数据、细粒度订阅、memo、视口裁剪；不要为每节点创建永久 WebGL context。

React Flow 阶段必须覆盖：服务容器与子节点移动/绑定、data/state/exec/command 端口、动态 spec、多选/复制/删除、搜索插入、分组/注释、撤销、缩放和键盘操作。视觉父容器与执行服务归属必须显式处理，不因拖拽位置偶然改变算子归属。

迁移清单：

| 能力 | 实施要求 |
| --- | --- |
| 属性与 schema 编辑 | 从 typed spec 构建，校验与后端一致，支持动态端口/状态/命令 |
| Studio 算子 | state/data expr、control panel、value stepper、patch hub 等分类迁移；纯 UI 注释无需运行时节点 |
| 3D | skeleton 协议、world-up、多人、相机、断流/重连；几何/材质显式释放 |
| 2D 可视化 | video、track、text、wave、audio、scalar/flow；逐一列行为测试 |
| Monaco/LSP | 虚拟工作区、动态类型/stubs、诊断、补全、会话清理；Python 在后端运行 |
| 项目与资产 | 本地 DB、版本、组件/变体、导入导出、同步、发布预览与认证 |
| 监控/日志 | 保持 monitor/data 边界，受限历史、过滤、导出、错误 tracebackId |
| 游戏/设备 | 原生 UDP/串口/进程/文件操作留后端，浏览器提供配置和结果 |
| 全局快捷键 | 网页快捷键与 OS 全局快捷键区分；后端原生适配或后续桌面壳，验证焦点外行为 |
| 文件操作 | 浏览器上传下载 + 后端明确路径策略；不能把浏览器 File 当任意 OS 路径 |
| 本地扩展 | template_match、viz_tcode 的运行逻辑和 renderer 分别迁入后端/前端 |

不维护旧 entry-point 插件 ABI。以明确的 Python 注册函数和 TS renderer 注册表替代字符串反射；目录中未知 renderer 给出诊断视图，不静默显示空白。不要新造通用插件平台。

## 8. AI、CLI 与 MCP

先复用现有 provider、对话和工具逻辑，解除 Qt 线程/信号依赖；不要同时更换所有 Agent 框架。

所有 graph/read/patch/validate/deploy/observe/project 工具调用 application service。内置 Agent 可进程内调用，CLI/MCP 经 API 调用，不绕过 revision、事务和部署规则。GUI 不在线时工具仍能工作。

第一组 AI 验收闭环：读取 catalog 与图 -> 生成 typed patch -> 展示 diff -> 校验 -> 按工具策略执行 -> 部署 -> 读取 monitor -> 返回有证据的结果。

审批绑定 toolCallId、精确参数 hash、目标 revision 和有效期；参数/图变化后失效，拒绝/超时不执行。审批只用于产品现有策略要求的副作用，不把所有只读操作做成人工确认。工具取消不等于已完成副作用回滚，UI 必须准确呈现。

模型没有 key 时可以跳过真实供应商请求，但必须通过确定性 fake provider 的完整工具闭环；跳过项明确记录，不能报告“AI 全部通过”。提供方秘密不发送到前端。

多模态第二步增加后端生成的帧/片段引用、来源与时间戳、ROI、有限采样率和上下文预算。不把每一帧自动上传模型；前端 UI Web 化不等于实时视频理解已经实现。

## 9. 阶段任务与硬性退出条件

按阶段提交可运行结果。除表中明确可后续完成的内容外，验证未通过不得宣称阶段完成。不需要在每阶段向用户重新请求开始下一步的许可。

### P0：基线、范围台账与可复现环境

- 建立 `web-studio-status.md`，记录基线 commit、操作系统、CPU/GPU/驱动、Pixi/Python/浏览器版本、当前测试失败。
- 枚举 Studio 功能及算子/扩展清单，每项有目标模块、验证 fixture、状态；功能可重构但不可无声丢弃。
- 记录现有真实图规模、视频预览和 3D 行为，选取轻量 deterministic 场景与一个完整真实服务场景。
- 新增独立 Pixi feature/environment：`web-studio`、`web-studio-test`，初始即不依赖旧 `studio` feature；不要从安装了 Qt 的 default 环境证明“无 Qt”。
- Pin Node/package manager 和 lockfile；验证 FastAPI、aiortc/PyAV、Zenoh 在 Python 3.14 与目标系统的安装/最小运行。若失败，记录 ADR 调整独立环境，不随意改变所有引擎的 Python。

退出条件：一条命令能启动最小无 Qt 服务并返回 health；浏览器空壳可启动；基线和依赖结果可重复。未实现功能不得返回虚假成功。

### P1：独立图领域模型和编译器

- 实现文档契约、目录、GraphStore、原子 patch、revision、history、校验与纯编译器。
- 将旧 compiler 中真实执行规则迁入新模型；以实际服务目录构造 fixture。
- 为 data/state/exec/command、容器归属、动态端口、自动采样、组件展开建立有效/无效样本。
- 迁移仓库样例到新文档格式；旧图只作参考行为证据，不把旧对象适配器作为正式接口。

退出条件：无 Qt 进程可创建/修改/保存/加载/编译完整图；失败 patch 无副作用；冲突和幂等行为正确；编译确定性通过。

### P2：无界面运行时与最小 HTTP/WS

- 移动 Studio runtime operators、注册表、服务部署/发现/状态/命令到新后端。
- 替换全局 UI command sink 为显式注入的呈现事件出口；同项目实例隔离，生命周期明确。
- 实现项目持久化、catalog、patch、deploy job、service control、事件 snapshot/resume。
- 启停一个真实 pyengine 或 C++ 服务，验证 rungraph、state、data、exec；内存 bus 测试只是单元测试。
- 对进程退出、Zenoh 断连、部署部分失败、重复请求、后端关闭添加集成验证。

退出条件：浏览器未打开也能通过 HTTP 完成图运行；刷新不重复部署、不停止算子；不产生孤儿进程；事件重连后与后端一致。

### P3：媒体与 3D 垂直原型，提前解决性能风险

- 不等完整画布：做专门验证页，接真实 SDK 视频 reader、WebRTC main/thumbnail、Three.js 骨架。
- 先 deterministic 视频（帧号/颜色/运动标记），再真实 f8implayer 或 f8screencap 场景；外部模型可暂用确定性骨架源替代。
- 覆盖 stride、断流、分辨率改变、流重启、关闭重开、一个源多视图和多源。
- 实现服务端匹配 overlay、标量/光流预览及精确数值查询基础链路；音频建立最小播放与波形验证。
- 输出初次基准，与 P0 同机基线比较；瓶颈未定位不得扩大完整 UI 重写。

退出条件：单路真实 1080p30 目标、4 路缩略图、3D 和图控制同时运行；同步、资源释放、CPU/内存记录齐全。达不到预算则先修媒体架构或记录明确降级能力，不宣称功能对等完成。

### P3.5：媒体网关进程隔离

- 将 Zenoh 视频/音频订阅、帧转换、overlay、aiortc peer 和编码从 Studio Server 移入独立 `f8media_gateway` 包与进程；将轻量控制契约放入 `f8media_protocol`。
- Studio Server 通过 typed async client 代理既有 `/api/media/*` 和 `/api/audio/*`，浏览器接口保持稳定；网关健康信息包含协议版本、epoch 和 PID。
- 正式 CLI 默认管理一个网关子进程；允许 `--external-media-gateway` 连接后续 Rust/C++ 实现。启动必须校验协议版本和受管 PID，关闭必须回收会话、source、HTTP client 与子进程。
- 保留 `InProcessMediaGateway` 作为明确的测试/基准注入点，不作为正式 CLI 默认路径。高频帧和计数不进入 state API，监控仍走 media metrics/data 边界。
- 为 native 替换保留 source、quality、SDP、overlay、frame mapping、sample 和 metrics 契约；不得将 Python 对象或 aiortc 类型暴露到 Studio Server。

退出条件：Studio 与网关 PID 不同；synthetic 和真实 C++ screencap 链路经 Studio 代理完成 WebRTC；422/404/503 保持语义；协议/PID 错配拒绝启动；Studio 退出后网关 PID 和端口消失。该阶段不宣称解决同源多 viewer 的重复编码，后续共享编码器需单独实现和基准验证。

### P4：Web 图编辑主工作流

- 接入 React Flow，完成图创建、连接、属性、服务容器、动态端口、搜索、复制、多选、撤销、保存/加载、部署/停止。
- UI 展示 draft 与 deployed revision，显示局部部署失败；不把布局更新当运行时配置。
- 接入真实 video/3D 视图与监控；实现可见性订阅和可关闭资源。
- 以 Playwright 验证从空图到运行结果的用户流程，而非仅截一张空白画布截图。

退出条件：不用旧 Qt 即可完成核心“建图 -> 配置 -> 运行 -> 观察 -> 修改 -> 保存重开”闭环；复杂图预算通过。

### P5：剩余本地业务能力

- 根据 P0 清单逐项迁移资产/组件/变体、schema 编辑、Monaco/LSP、音频/曲线/track、两个本地扩展、游戏和设备操作、全局快捷键。
- 迁移相应业务测试；废弃 Qt 几何/生命周期测试，但为保留行为补前端或后端测试。
- 禁止只迁移菜单入口、背后继续调用隐藏 Qt 进程。

退出条件：功能台账全部标记已迁移或有明确范围说明；核心原有能力不能仅列“未来支持”而宣布全量完成。需硬件/Windows 的项标记待验证，保留发布门槛。

### P6：AI、CLI、MCP 统一服务入口

- 迁移 provider、Agent 生命周期、对话/产物、工具进度与审批；替换所有 GUI graph adapter。
- CLI/MCP 对同一图修改产生相同提交事件，浏览器立即同步。
- 实现至少一个确定性 AI 自建图/诊断闭环，及一个具备可用凭据时的真实模型 smoke test。

退出条件：无浏览器、无 Qt 时工具完整工作；多标签/AI 并发冲突正确；执行失败有上下文和 traceback；凭据不出现在网络响应或前端 bundle。

### P7：移除 Qt 与正式交付

- 删除旧 GUI 包/模块、旧插件加载路径、过渡代码；更新所有本地 imports、service.yml、describe、样例、任务、launcher、dist/build 脚本。
- 清理 Pixi features/dependencies/lockfile；若 OpenCV 引入 GUI Qt 依赖，验证 headless 发行包能覆盖实际 CV 功能后替换。
- 全新环境安装构建后的 wheel/Web bundle，离线启动核心功能；不要只验证 editable 源码运行。
- 更新开发/用户文档和默认启动入口，提供 build/start/test/codegen/benchmark 命令。
- Windows/Linux 分别执行发布验证；剩余平台验证不得用 Linux mock 冒充。

退出条件：第 12 节全部满足，最终报告证据和限制。删除 Qt 是最终结果，不是以删除依赖替代功能迁移。

## 10. 验证策略与初始预算

### 10.1 测试层次

| 层 | 必须验证的风险 |
| --- | --- |
| core pytest | patch 原子性、ID/边规则、revision、undo 前提、编译确定性、无效配置 |
| server pytest | 存储原子提交、job 生命周期、幂等、错误 envelope、订阅背压、进程所有权 |
| 真实集成 | Zenoh + pyengine/C++、真实采集或播放、断开恢复；不能全用 mem backend |
| Vitest | 文档投影、状态分离、冲突回滚、输入校验、资源订阅规则 |
| Playwright | 编辑/保存/重开/部署、视频有实际变化帧、3D 有正确几何、Agent 工具结果 |
| 有头 GPU smoke | 实际视频解码、WebGL、硬件编码能力、真实显示延迟和视觉正确性 |
| 构建/架构 | 生成契约无漂移、类型检查、无 Qt 安装/导入、打包后离线资源完整 |

不为纯展示文案写重复实现的测试。优先测试真实失败模式：错误端口、过期 revision、断网、编码失败、后台恢复、帧积压、服务崩溃。

### 10.2 可量化场景

以下是初始验收预算，不是已测结论。P0 固定测试机器与媒体内容；如需调整，必须记录原因和前后数据，不能为通过测试静默放宽。浏览器首发为 Windows/Linux 的 Chromium 系，其他浏览器另测后再声明支持。

| 场景 | 初始预算/断言 |
| --- | --- |
| 1080p30 主预览，localhost，60 秒 | 热身后实际显示均值 ≥28 FPS；采集到显示延迟 p95 ≤200 ms |
| 主预览 + 4 路 360p10 + 单个 3D，5 分钟 | 控制响应 p95 ≤200 ms；无持续队列增长；主预览达到上述目标或明确定位瓶颈 |
| 精确 overlay | 结果与视频源帧匹配；超时隐藏/标记，零次静默误配；不能只测 WS 到达延迟 |
| 图画布 300 节点/600 边 | 常规拖拽目标 ≥50 FPS，交互响应 p95 ≤100 ms；另测实际最重节点组合 |
| 1000 节点/2000 边压力场景 | 输出帧时间/内存/响应曲线，不把低复杂度节点结果外推到视频节点 |
| 断连 5 秒后恢复 | 网络恢复后 5 秒内恢复权威文档/状态；媒体重新协商后恢复；无重复图操作 |
| 关闭/重开预览 50 次 | peer、订阅、任务数量回到基线；热身后 RSS 增长 ≤max(50 MiB, 10%)；GPU 资源单独观察 |
| 持续运行 30 分钟 | 无未处理异常、无限历史/队列增长；CPU/GPU/内存/丢帧可导出 |
| 数值数据 | 已知 F16/F32 fixture 正确解码，NaN/Inf/范围/端序测试明确 |

延迟报告包含采集、SDK 读取、转换/编码、浏览器解码/显示的测量方法与时钟误差。软件无 GPU CI 只能验证流程，不能作为性能达标证据。使用帧内标记或已验证时间映射，不能以 HTTP ping 代替视频端到端延迟。

性能统计走 monitor/data，绝不新增 FPS、延迟、处理/丢帧计数等 service stateFields。

### 10.3 实施者应提供的标准命令

下列命令是待新增的任务名称，现在不能假定已经存在。创建任务后再在状态文档登记实际命令与结果。

```bash
pixi run -e web-studio studio_web_dev
pixi run -e web-studio studio_server
pixi run -e web-studio-test studio_core_test
pixi run -e web-studio-test studio_server_test
pixi run -e web-studio-test studio_web_test
pixi run -e web-studio-test studio_web_e2e
pixi run -e web-studio-test studio_web_typecheck
pixi run -e web-studio-test studio_contract_check
pixi run -e web-studio-test studio_no_qt_check
pixi run -e web-studio-test studio_media_bench
pixi run -e web-studio studio_web_build
```

Python 使用 Pixi；Node/npm/pnpm 等同样通过选定的受管环境执行。对 SDK/引擎有改动时运行对应现有 pytest；影响 C++ 时运行相应 configure/build/test。不要每改一行就全量跑全部平台测试。

`studio_no_qt_check` 至少包含静态 import/依赖检查、干净环境包清单、启动后的实际模块检查和关键业务执行。还应核对构建产物中的 Qt 动态库；不能仅搜索字符串 Qt（历史文档必然提到 Qt），也不能只证明没有弹出窗口。

## 11. 已知风险与决策触发条件

| 风险 | 发现阶段 | 决策 |
| --- | --- | --- |
| aiortc/PyAV 的平台/Python 版本问题 | P0 | 在独立环境解决版本，记录 ADR，不阻塞到 P3 才发现 |
| Python 编码 CPU 或多 peer 开销超预算 | P3/P3.5 | 已隔离媒体进程并稳定 `f8media-api/1`；下一步在网关内实现共享/native 编码，Studio 与浏览器信令 API 不改 |
| 浏览器无法可靠关联 SDK frameId | P3 | 精确 overlay 用服务端合成；客户端叠加必须单独证明 |
| React Flow 复杂图不达标 | P4 初期 | 先细粒度更新/裁剪；仍不达标时评估 Canvas/WebGL 画布，保留同一文档模型 |
| 编译器存在隐式 Qt 行为 | P1 | 用业务 fixture 明确规则，禁止 fake Qt 兼容对象 |
| Studio 算子消失导致图不能运行 | P2 | 本地算子逐个有归属和运行验证，再删除旧注册路径 |
| 浏览器后台节流 | P3/P4 | 后端继续运行、媒体按可见性降级；恢复时丢弃过期呈现数据 |
| OS 集成功能受浏览器沙箱限制 | P5 | 保留后端原生能力；必要时后续桌面壳，不假装网页拥有任意本机权限 |
| 旧测试大量依赖 Qt | 各阶段 | 迁移业务断言，删除已废弃 UI 机制测试，不批量 skip 保绿 |
| 全量迁移范围漂移 | 每阶段 | 按功能台账与退出条件交付，避免同时重做计算引擎/云平台/通用插件框架 |

## 12. 最终完成定义

- [ ] 新入口无需 Qt 包、Qt 动态库或隐藏 Qt 子进程。
- [ ] 所有保留运行时服务和 Studio 本地算子能通过新模型运行。
- [ ] Web 可独立完成建图、配置、运行、观察、保存、加载。
- [ ] 实时视频、3D、音频及数值可视化有行为与性能证据。
- [ ] 图编辑、AI、CLI、MCP 使用同一权威文档与应用服务。
- [ ] 旧本地扩展能力已显式迁移，无长期旧 ABI 适配层。
- [ ] 工程内调用者、样例、清单、生成文件、启动器和文档同步更新。
- [ ] 断连、重启、部分部署失败和进程清理有测试。
- [ ] Python/TS 类型检查、契约检查、针对性测试和 E2E 通过。
- [ ] Windows/Linux 干净构建安装验证完成；缺少硬件/凭据的验证明确列出，不冒充通过。
- [ ] 核心 UI、Monaco、Three.js 资源本地打包，无 CDN 依赖。
- [ ] 用户数据未被隐式删除，格式升级策略明确。

## 13. 给 GPT 5.6 Sol 的执行指令

从 P0 开始实际修改仓库，再逐步进入后续阶段。先阅读根 AGENTS.md 与所修改目录的局部约束。以本方案为目标设计，不把当前 Qt 结构视为不可改变的约束。

每阶段交付：可运行代码、与风险匹配的验证、状态文档更新、剩余限制。记录“计划/已实现未验证/已验证/受阻”，不要把设计目标写成完成事实。出现确定性失败先定位修复；缺少设备、平台或凭据时继续其他独立工作，并明确未验证门槛。

禁止通过 `getattr/setattr/hasattr` 模拟静态接口，禁止传递无约束 Any 代替领域类型，禁止静默吞异常。核心逻辑 fail fast；UI/API/worker 边界捕获时输出上下文、异常及 traceback，重复高频错误去重。

可以删除旧实现、更新全部本地调用者，不要为未知外部消费者保存兼容路径。迁移阶段可以暂留旧 Qt 应用作行为参照，但新产品的运行链路不能调用它，P7 必须清理。

首个实际交付应是：独立无 Qt Pixi 环境、最小 core/server/web 包、health 与本地 Web 启动、功能台账和依赖探测证据。随后完成纯文档编译与真实服务运行，尽早进入 P3 媒体验证。不要把第一阶段耗在完整主题、布局美化或通用插件框架上。

## 14. 参考资料

- [WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)
- [WebCodecs API](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API)
- [requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)
- [React Flow performance](https://reactflow.dev/learn/advanced-use/performance)
- [aiortc documentation](https://aiortc.readthedocs.io/en/latest/)

实现时以实际锁定版本的接口和目标平台测试为准。
