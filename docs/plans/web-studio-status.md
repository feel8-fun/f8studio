# Web Studio 迁移状态

最后更新：2026-09-22。目标方案：[web-studio-migration.md](web-studio-migration.md)。

## 当前结论

P0、P1、P2、P3 和 P3.5 已在当前 Linux 主机完成验证。P2 包含持久化项目、图编辑 API、可靠事件游标、部署 job、服务控制、运行时 monitor、真实 PyEngine 联调，以及 13 个无 Qt Studio runtime operators。P3 已完成 WebRTC main/thumbnail、真实 screencap -> Zenoh -> WebRTC、FLOW/SCALAR、服务端精确 overlay、WebRTC 音频与波形、Three.js 骨架、重启恢复、浏览器显示延迟和 30 分钟稳定性验证。P3.5 将 Zenoh 订阅、媒体转换、aiortc peer 和编码执行迁入独立 `f8media_gateway` 进程，Studio Server 只依赖轻量 `f8media_protocol` 并代理 typed HTTP 信令。多 peer 软件编码在组合负载下的 main 为 26.86-27.52 FPS，低于 28 FPS 初始预算，因此 P3 原型完成不等于媒体性能完全对等；进程隔离已完成，硬件/共享编码仍是发布前优化项。`f8studio_core`、`f8media_protocol`、`f8media_gateway`、`f8studio_server` 和 `f8studio_web` 均为独立包；Web Studio 与服务运行时环境都不组合旧 `studio` feature。

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
| eclipse-zenoh | 1.10.1 |
| 首发浏览器范围 | Windows/Linux Chromium；Linux Google Chrome E2E 已通过，Windows 待验证 |

`studio_dependency_probe` 已成功导入 FastAPI、Zenoh、aiortc、PyAV，当前 Linux 构建报告 H.264、VP8、VP9 软件编码器可见。这只证明依赖和 codec 注册可用，不是 WebRTC 性能通过。

## 阶段状态

| 阶段 | 状态 | 证据 / 下一步 |
| --- | --- | --- |
| P0 基线与环境 | 完成（Linux） | 独立环境可求解；严格 Python/TS 类型检查、单元测试、生产构建、真实 HTTP 探测、npm audit、无 Qt 审计通过 |
| P1 图模型与编译器 | 完成 | 纯数据文档、显式目录/端口、原子 typed patch、双 revision、history、校验、确定性编译与新格式样例均有测试 |
| P2 无界面运行时 | 完成（Linux） | typed HTTP/WS、13 个 Studio 算子、真实 PyEngine、内置 Studio graph/presentation、部分部署、断连和关闭均已验证 |
| P3 视频与 3D 原型 | 完成（Linux，性能降级已记录） | 视频/数值/精确 overlay/音频/Three.js/重启/延迟/5 分钟组合与 30 分钟稳定性均有真实证据；组合 main 未达 28 FPS，见下方限制 |
| P3.5 媒体网关进程隔离 | 完成（Linux） | `f8media-api/1`、远程代理、独立 PID、真实 screencap 链路和父进程关闭回收均已验证；共享/native 编码待后续优化 |
| P4 Web 图编辑 | 未开始 | 当前页面是连接到真实 health API 的工作区壳 |
| P5 本地业务能力 | 未开始 | 依据下方功能台账逐项迁移 |
| P6 AI / CLI / MCP | 未开始 | 旧实现仍依赖 Qt 图适配器 |
| P7 移除 Qt | 未开始 | 旧 Qt 应用仍作为行为参照保留 |

## 功能迁移台账

状态含义：`待迁移` 表示已纳入范围但新链路不可用；`基础已建` 表示只有公共入口或契约；`核心完成` 表示无界面领域层已经完成，但还未接入 P2 API 或运行时。

| 领域 | 现有能力 | 新目标 | 状态 |
| --- | --- | --- | --- |
| 图文档 | 节点、服务容器、data/state/exec/command 边、布局 | 纯数据文档、显式端口类型、graph/layout revision | 核心完成 |
| 图操作 | 创建、删除、连接、动态 spec、undo/redo | 原子 typed patch、幂等、冲突和历史 | 核心完成 |
| 编译部署 | NodeGraphQt 对象编译为 runtime graph | 纯文档确定性编译、异步部署 job | 核心与 API 完成；真实 PyEngine 部署通过 |
| 服务运行时 | 发现、启停、状态、命令、监控、受管进程 | 无 Qt application service | 基础完成；真实 Zenoh、状态、exec 与关闭通过 |
| Studio 算子 | text/track/wave/video/audio/3D、control、expr、patch hub 等 | 后端算子与前端 renderer 分离 | 13 个后端算子全部迁移；Web renderer 属于 P3/P4/P5 |
| 视频 | BGRA、FLOW2_F16、SCALAR1_F32 latest-frame | WebRTC 分级预览、精确 overlay、数值查询 | 垂直原型完成；传输 v2 携带 producer epoch，overlay 按 source/stream/epoch/frame/timestamp 精确匹配，超时隐藏并计数 |
| 3D | Three.js 骨架、world-up、多人 | 本地打包 Three.js + 有界数据通道 | 垂直原型完成：本地 Three.js、world-up、多人协议、事件重连和显式资源释放 |
| 音频与曲线 | 播放、波形、频谱、track | WebRTC 音轨和降采样绘制 | P3 音频播放与 RAF 波形完成；频谱与通用曲线/track renderer 留在 P5 |
| 编辑器 | Monaco、Python stubs/LSP | 浏览器 Monaco + 后端 LSP 会话 | 待迁移 |
| 项目与资产 | 项目、版本、组件、变体、导入导出、云同步 | 后端持久化和类型化 API | 项目文档与元数据完成；版本/组件/同步待迁移 |
| 本地扩展 | template_match、viz_tcode | 显式后端注册与 TS renderer 注册 | 待迁移 |
| 游戏/设备 | Unity/VaM、UDP、串口、外部进程 | 原生能力留后端，Web 配置与观察 | 待迁移 |
| 快捷键 | Qt/OS 全局快捷键 | Web 局部快捷键 + 后端原生全局适配 | 待迁移 |
| Agent | provider、会话、工具、审批、图构建 | 统一 application service，无 Qt bridge | 待迁移 |
| Web/API 壳 | 无独立产品入口 | loopback FastAPI、health/capabilities、React 工作区 | typed 图/项目/job/runtime/monitor API 与 WS 事件已建 |

## 当前验证

```text
pixi run -e web-studio-test studio_core_test            18 passed
pixi run -e web-studio-test studio_media_gateway_test   23 passed
pixi run -e web-studio-test studio_server_test          37 passed
pixi run -e web-studio-test studio_web_test             1 passed
pixi run -e web-studio-test studio_web_e2e              7 passed, 1 skipped（desktop/mobile；1080p 延迟仅 desktop）
pixi run -e web-studio-test studio_python_typecheck     0 errors
pixi run -e web-studio-test studio_web_typecheck        passed
pixi run -e web-studio-test studio_no_qt_check          passed
pixi run -e web-studio-test studio_dependency_probe     passed
pixi run -e web-studio studio_web_build                 passed
pixi run -e web-studio-test studio_media_bench          passed
pixi run -e cpp cpp_test_release                         27 passed
pixi run -e web-studio-test studio_stream_restart_probe passed（0.372 秒恢复）
pixi run -e web-studio-test studio_real_video_probe     passed（C++ screencap，1920x1080）
P3 SDK / detection targeted tests                       40 passed
pixi run -e web-studio npm --prefix packages/f8studio_web audit --json
                                                        0 vulnerabilities
```

尚未验证：Windows 求解/运行、硬件编码、非 Chromium 浏览器和远端跨主机时钟。它们属于发布平台补充，不能据当前 Linux 软件编码结果宣称通过。

P3.5 当前 Linux 证据：正式 `studio_server` CLI 在 PID 2132966 启动受管 `f8media_gateway` PID 2132996，分别监听 `127.0.0.1:8260` 与 `127.0.0.1:8261`。`/api/media/gateway` 返回 `f8media-api/1`、独立 gateway epoch 和子进程 PID。经 Studio 原 URL 代理，synthetic WebRTC 解码出变化的 640x360 帧；真实 C++ screencap -> Zenoh v2 -> Gateway -> WebRTC 解码出变化的 1920x1080 main 帧。中断 Studio 后两个 PID 与两个监听端口均消失。单元/集成测试另覆盖网关资源清理、422/404 透传、断连 503、协议错配、端口占用时 PID 所有权校验和 managed process 回收。

里程碑审查进一步补齐了异常关闭与协商竞态：Studio 与受管 Gateway 之间保留 stdin ownership pipe，父进程异常退出时 Gateway 自动终止；快速 Stop 使用 operation token 阻止过期协商回写 UI 或遗留 session；处于 ICE checking 的 peer 在逻辑 session/source 立即释放后延迟到底层 STUN transaction 结束再关闭，避免 aioice 0.10.2 的 retry-after-close 未处理异常。该取消场景在 desktop/mobile 连续 6 次压力重复和完整 E2E 中均通过，资源计数回到零。Studio 构造路径也已验证不导入 `f8media_gateway`、aiortc 或 PyAV。

Zenoh envelope 的通用二进制解码函数仍属于 SDK 的 `f8pysdk/video_transport.py`，便于 Python publisher/subscriber 共用协议；调用它的 subscriber、latest-frame hub、BGRA/FLOW/SCALAR 转换、overlay、WebRTC track、peer/session 和 aiortc 编码执行均位于 `f8media_gateway`。一个网关进程管理所有 source/viewer；同 source 共享一次 Zenoh subscriber 和部分预处理，但每个 `RTCRtpSender` 当前仍创建自己的编码器。因此该拆分隔离了 Studio 的 GIL、事件循环和故障域，并提供未来 Rust/C++ 网关替换边界，没有消除同源多 viewer 的重复编码。

P3 当前 Linux 证据：真实 Uvicorn HTTP 信令可建立 aiortc peer，`synthetic://bars-1080p` 解码为变化的 1920x1080 帧；受管 `f8.screencap` 经项目部署 arm 后，真实链路 `screencap -> Zenoh v2 latest-frame -> WebRTC` 同样解码为变化的 1920x1080 帧，探针结束后服务进程被回收。BGRA padded stride、动态分辨率、源时间戳回退、同源多视图共享、不同源隔离、关闭重开和 application shutdown 均有测试。FLOW2_F16 使用 HSV 方向/20.0 magnitude scale，SCALAR1_F32 使用 2/98 百分位和 turbo 色图；NaN/Inf 在预览中转黑，精确查询返回 `null + finite=false`，并保留 source frameId、stream epoch 和时间戳。

视频帧协议 v2 由 Python/C++ publisher 在每次 producer 启动时生成 128 位 `streamEpoch`。Python/C++ describe metadata、payload schema version 和本地 catalog 均已核对为 v2，`streamEpoch` 为必填字段；音频协议保持 v1。检测输出携带 `streamId + streamEpoch + frameId + tsMs`；overlay store 以完整身份和 capture timestamp 为 key，只在 50 ms 等待预算内合成匹配的 BGRA 帧。未匹配结果隐藏，过期/拒绝/超时通过 `/api/media/metrics` 监控，store 上限 256 条/2 秒，raw history 上限 8 帧/300 ms。专项测试覆盖精确绘制、frame/epoch/timestamp 不匹配、超时、过期和有界内存。

音频使用独立 typed WebRTC 会话，支持 `synthetic://tone`、`synthetic://tone-stereo` 和真实 `f8/` Zenoh F32LE source；当前严格接收 48 kHz mono/stereo 并显式转换为 WebRTC PCM。浏览器必须由 Play 手势启动 AudioContext，提供音量、静音、停止和 RAF/AnalyserNode 波形。Chrome E2E 证明收到非静音音轨且波形非平坦。Zenoh transport 是 latest-chunk：慢消费者可能丢块，sequence gap 会计数；当前不承诺 gapless 播放或音视频同步。

`studio_media_bench` 最新结果为 60.00 秒的 1 路 1920x1080 main + 4 路 thumbnail：main 28.80 FPS，thumbnail 7.88-7.95 FPS，进程 CPU 177.8%。负载 RSS 889.9 MiB；释放后的热基线 725.0 MiB，随后 50 次 SDP 连接/关闭为 704.5 MiB，session/source 每轮回到零。单独媒体场景达到 main ≥28 FPS；thumbnail 受每 peer 软件编码限制，低于 10 FPS 档位上限。

组合 5 分钟场景额外包含一条音频：main 26.86 FPS、四路 thumbnail 7.69-7.73 FPS、CPU 191.1%、控制响应 p95 47.38 ms、错误 0、资源计数归零。30 分钟稳定性场景按退出条件运行 main + 4 thumbnails + 10 Hz 3D 骨架事件 + 2 Hz graph/monitor HTTP 控制：main 27.52 FPS、thumbnail 7.66-7.69 FPS、CPU 177.5%、3,474 次控制请求 p95 43.18 ms、16,877 条 presentation 事件、队列峰值 1、错误 0。任务数负载态 42、峰值 58、清理后 2；RSS 活跃样本约 500-693 MiB，最后 5 分钟净增 19.9 MiB（门槛 50 MiB），清理后 522.4 MiB；video/audio session/source 全部归零。原始证据位于 `docs/plans/evidence/p3-combined-5m.json` 与 `docs/plans/evidence/p3-stability-30m.json`。组合 main 未达 28 FPS，已定位为多 peer 软件编码瓶颈；不宣称媒体性能完全对等。

Playwright 使用本机 Google Chrome 在 1440x900 与 390x844 视口验证：视频元素像素随帧变化、点击画面可查询带 frameId 的 latest raw 数值、Three.js framebuffer 非空、轨道拖拽改变画面、WebRTC 音频波形非平坦、布局无横向/纵向溢出，且页面无未处理错误。1080p main 使用帧内 16 位 capture clock marker 和 `requestVideoFrameCallback` 测得 60 个显示帧 p95 68 ms；localhost 浏览器和 producer 共用系统时钟，未使用 HTTP ping 代替视频延迟。截图保存在测试产物目录；Three.js 和图标均从本地 bundle 加载，无 CDN。

真实 Zenoh publisher 关闭并重开后，producer epoch 改变，现有 WebRTC session/source 保持单实例并在 0.372 秒恢复出帧；结束后资源计数归零。该结果满足 5 秒恢复预算，且没有重新执行图操作。

真实 P2 探针已在本机通过 Zenoh 启动独立 `web-studio-runtime` 环境中的 `f8.pyengine`，经 HTTP 创建 `f8.tick -> exec -> f8.print` 图并部署，确认 runtime graph revision、状态写入、deactivate/activate 与 monitor `processed=15`。探针停止服务后无残留进程；另在服务仍运行时中断 Web Studio，应用约 1.5 秒完成关闭且 supervisor/PyEngine 均被回收。该证据覆盖当前 Linux 主机，不替代 Windows 验证。

后端现在内置独立的 `f8.pystudio` ServiceRuntime，catalog 暴露 13 个已迁移算子：text、wave、video、audio、track、3D、data/state expr、control panel、backdrop、note、patch hub 和 value stepper。表达式求值使用 AST allowlist；精简环境未安装 NumPy 时 `allowNumpy` 返回明确 monitor 错误。Track 和 3D 输入通过显式 `msgspec.Struct` 边界规范化，不复制旧 Qt renderer。每个应用实例创建独立 registry 与 `EventPresentationOutlet`，不使用旧全局 UI command sink；presentation 与 monitor 都作为不可靠实时事件，队列满时丢弃，不挤占可靠 graph/job 事件的 replay 语义。新运行链路未导入 `f8pystudio`。

`probe_studio_presentation.py` 已通过真实 Uvicorn、Zenoh 和 WebSocket：经 HTTP 创建并部署仅包含内置 `studio` 服务与 `f8.viz.three_d` 的项目，写入 `worldUp=-z` 后收到 `presentation.command / viz.three_d.world_up`。精简 server 现在明确依赖 `websockets`，因此生产 `/api/events` 不再依赖环境中偶然存在的 Uvicorn extra。部分部署测试固定了一个服务成功、另一个 Zenoh 断连时的 `partially_failed` 和逐服务结果；runtime 请求断连固定映射为 HTTP 503。

P1 测试覆盖文档编解码与新格式样例、真实 `services/f8/engine/describe.json` 目录、data/state/exec/command、动态 spec、patch hub、跨服务 half-edge、自动采样、组件 fragment、禁用节点、payload 不匹配、状态环、原子回滚、revision 冲突、幂等及 undo/redo。语义 hash 不含布局和事务 revision，并对节点、边及 JSON mapping 的插入顺序稳定。

项目 mutation 与 request id 幂等记录已在同一 SQLite 事务提交，默认每项目保留 2048 条，并已验证进程重启后重放与冲突。幂等重放不重复发布 `graph.committed`。undo/redo 历史仍属于当前进程中的项目 session；持久版本历史属于后续项目/资产工作。每次提交都要求精确 graph/layout revision，因此过期客户端不能覆盖新提交。

## 已发现的仓库前置问题

旧 `default` 环境因 `external/f8unitymods` 缺少 `pyproject.toml/setup.py` 无法求解。新 `web-studio` 环境不引用该 feature，因此可以独立工作。modding 功能仍在迁移范围内；不得通过空包掩盖缺失源码。

## 标准命令

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
pixi run -e web-studio-test studio_python_typecheck
pixi run -e web-studio-test studio_web_typecheck
pixi run -e web-studio-test studio_no_qt_check
pixi run -e web-studio-test studio_dependency_probe
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
