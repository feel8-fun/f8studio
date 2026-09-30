全工程审阅报告（不含 docs）
7 个审阅 agent 分子系统做了只读审查。高优先级结论中我亲自抽查了 10 余条，全部属实。有一处副作用要先说明：SDK 审阅 agent 在尝试运行 codegen 时，往仓库根目录下载了 22 个 .whl 文件，我已删除。现在工作区是干净的，pixi 环境未受影响。

总体判断
前后端的字段契约本身对得比较齐：13 种 GraphOperation 和各类枚举都一致，客户端也没有调用不存在的接口。问题集中在五处：

传输层重复造轮子：推送、轮询、快照三套机制并存。
两份手写契约：TS 类型和 Python 模型各写一份，没有单一真源。
同一逻辑多份拷贝：表达式沙箱、会话池、Zenoh key 命名等。
应用层边界不清：同一类业务逻辑分散在路由、StudioAutomationTools、StudioApplication 三处。
没有 CI 质量门：测试、类型检查都不在 CI 里跑。
P0：已确认的 bug，建议马上修
#	问题	位置
1	VideoFramePacket 没有 stream_epoch 字段，f8pydl 每次发检测结果都抛 AttributeError，而且被循环边界吞掉。pixi run typecheck 能报出来，但 CI 不跑它。	service_node.py:926、video_frame_source.py:59
2	端口不一致：server 默认 8210，CLI、MCP、api_client 默认 8260，默认配置下连不上。	main.py:26、api_client.py:19、cli.py:75、mcp_server.py:127
3	PyEngine 用的是 -e web-studio-runtime，这个环境没有 launcher-runtime feature，dist_ci 也不会改写它，发布包里的 engine 很可能起不来。	services/f8/engine/service.yml:7
4	图编辑会被静默丢弃：保存进行中时，commit 直接 return。SchemaEditor 的 try/catch 永远看不到失败，滑块连续拖动时的修改也会丢。	GraphWorkspace.tsx:590
5	Agent 工具调用被拒绝后，会话状态一直停在 waiting_for_approval，输入框永久禁用。run 超时时，未决的审批和工具调用也不会被清理。	agents/service.py:1393、agents/service.py:1354
6	Agent 窗口的 WebSocket 没有 onclose 重连，服务端重启或 resync 之后会静默停止更新。	AgentWorkspace.tsx:231
7	媒体网关里失败的 hub 仍留在 _hubs 字典中，之后新建的会话都会复用这个死 hub，立即报错。音频侧同样如此。	media.py:231、audio_media.py:190
8	网关在事件循环里、持锁状态下同步执行 zenoh.open()，会卡住所有正在播放的 WebRTC 会话。而且每个 source 都开一个带 256MB SHM 的 session。	media.py:711
一、Web ↔ Server 通信协议（你最担心的部分）
1. 事件通道：一个设计了但没用上，另外造了四个
服务端的 /api/events 已经实现了 epoch/sequence 断点续传加快照的机制，但前端没有任何地方传 epoch 或 after。实际情况是：

每个标签页开 4 条 WebSocket：Graph、Logs、Agent、Presentation 各一条。每条都接收全部事件，包括高频的 presentation.command，同一帧要 JSON.parse 4 次。
高频可视化数据会挤满每个订阅者 256 容量的队列。队列满了之后，下一条可靠事件会触发 resync_required 并断开连接。
服务端每次连接都发一份 stream.snapshot，没人读。
resync 事件复用当前 sequence（events.py:159）。一旦将来客户端开始用续传，就会跳过被丢弃的事件。
建议（最简方案）：

前端做一个单例 EventStream：一条连接、一次解析、按 type 分发、带续传和重连。
可视化数据单独走 /ws/presentation。服务端对每个 (node, command) 只保留最新值（latest-value 合并），并且只推送客户端订阅的节点。它的语义是"最新值"，不是"日志"，不应该和可靠事件共用一个 journal。
删掉无用的 snapshot，或者真正把它用起来。
2. 同一份状态既推送又轮询
数据	推送	轮询
监控数据	runtime.monitor	每 2 秒
部署任务	deploy.*	每 200ms，20 秒硬上限（超过 20 秒的部署会被误判为失败）
代码编辑器所在的项目	graph.committed	每 2 秒拉整个项目
节点运行时状态	无	每个节点每 1.5 秒一次，服务端再按字段拆成多次 read_state
原则是：已经有事件推送的，只在断线期间用轮询兜底。节点状态改成一个 store，批量请求或者走推送。

3. 契约没有单一真源
contracts.ts 是纯手写的。路由都收原始 Request、返回 F8JsonValue，所以 OpenAPI 文档是空的。

建议：

以 msgspec 为唯一真源，建一张显式路由表 (method, path, Req, Resp)。
用 msgspec.json.schema_components 生成 JSON Schema，再生成 contracts.gen.ts。protocol.yml 里的 spec 类型也一起生成。
CI 检查生成物是否过期。
顺带统一大小写：/api/health、/api/capabilities 和 query 参数现在是 snake_case，其余都是 camelCase。
4. 错误处理不统一
错误响应体有三种形态：{detail: str}、{detail: {code, message}}、FastAPI 校验错误的 {detail: [...]}。requestJson 处理不了第三种。
全局把 ValueError 映射成 422、FileNotFoundError 映射成 404，且不记日志（app.py:213-223）。内部 bug 会伪装成客户端错误，违反 AGENTS.md。
前端约 8 个接口直接用裸 fetch，丢掉了服务端返回的 detail。TemplateCapture 收到 success:false 仍然显示 "Applied"。
建议：

统一错误体为 {code, message, details?}。
业务层显式抛 NotFoundError / InvalidRequestError，其余异常落到 500 并记日志。
前端所有请求走 requestJson，RuntimeActionResult 定义成类型，并在一个地方集中检查 success。
5. 安全
没有任何认证，而且缺少 Origin 头的请求直接放行（app.py:108-110）。本机任意进程都能启动服务、通过 modding/unity/apply 写文件、改 API key。如果用 --host 0.0.0.0 启动，整个局域网都能访问。
Origin 检查不比对端口，所以 localhost 上任意端口的页面都能修改状态。
**建议：**每次启动生成一个随机 token，注入到 index.html 或 cookie 里；外部调用方从数据目录读取。Origin 检查做 scheme、host、port 完全匹配。生产配置里去掉 testserver。

6. 死路由
以下路由前端都没有调用。建议逐个判断：删掉，或者注明供谁用。

/api/capabilities
assets 的 import/export
GET /editor/sessions/{id}
PUT /projects/{id}
DELETE /jobs/{id}
/runtime/services/{id}/start|stop|status|active
/api/media/overlays|metrics|sample|media-timestamps，连同整个 overlay 子系统和逐帧映射队列
二、冗余与冲突设计（跨模块）
冗余或冲突	建议
表达式沙箱有 4 份实现：f8pyengine 和 studio_runtime 各一份 _py_expr_eval、data_expr、state_expr（每个文件相差约 300 行，sigmoid 语义都不同），另有 f8pyscript/expr_validator 和 wave_expr_lang 各带一份 AST 白名单。这也是安全问题。	统一收敛到 f8pysdk.expr
video_latest.py 在 f8pyengine 和 f8pyscript 各一份，只差 26 行；error_reporter 也重复。	移到 f8pysdk
日志去重（AGENTS.md 第 6 条）被重复实现了约 20 次。	在 f8pysdk 提供 DedupedErrorReporter
graph.committed 的发布和热键刷新在三个地方各写一遍（两处在 app.py，一处在 automation_tools.py:117）。删除、停止、重启、导入、恢复这些用例的业务逻辑写在路由里，所以 Agent 和 MCP 做不了这些操作。	StudioAutomationTools 升级为真正的用例层（ProjectLifecycle），路由只负责解码、调用、编码
幂等处理做了两遍：GraphStore._processed 在内存里，processed_requests 在 SQLite 里，指纹算法还不同。undo 栈无上限，重启后丢失。	只保留 repository 那一份；undo 设上限
图校验两遍：f8studio_core/graph/validation.py 抛带错误码的异常，f8pysdk/rungraph_validation.py 抛 ValueError。	只保留一个归属方
f8pysdk/service_runtime_tools/session/（约 440 行，Qt 时代的编译器）只有测试在引用，和 f8studio_core/compiler.py 重复。	删除
命令有两条通道：Zenoh request/reply，以及隐藏的保留态 state 字段 __cmd__.*。后者把 RPC 塞进了 state，违背 state/data 的边界。	明确只用于图连线场景，或迁移到 data/exec
zenoh_naming 和 f8_naming 重叠，Python 和 C++ 两边都有 3 个无人调用的 key 函数，而且和实际使用的 key 格式不一致。	合并成一个模块，删除死函数
wire 约定（key 模板、控制端点名、帧头布局、fingerprint 规范化、monitor schema）在 Python 和 C++ 里各手写一遍，已经出现了分歧：debug_data 只有 Python 有。	写进 protocol.yml（或新建 wire.yml）统一生成，并加跨语言 golden 测试
5 个 repository 各自连接 SQLite，各自设 PRAGMA 和建表，连接从不关闭，只有 projects 表有迁移版本。	做一个 StudioDatabase 统一管理
前端 Video/AudioSessionPool 约 95% 相同，网关的 Media/AudioSessionManager 也基本一样；而且只有视频的关闭请求带了 keepalive。	各抽一个泛型实现
viz_* 算子：video、audio、tcode 没有继承 _viz_base，节流刷新逻辑复制了 3 份，还有 7 个几乎一样的 factory。	统一继承基类，抽出 ThrottledEmitter
渲染器相关知识分散在 5 处；扩展的 sdk.ts 没人引用，TCode 直接写死在核心代码里。	内置渲染器也走 extension registry
服务 id studio 和 studio_<epoch> 之间的映射散落在 3 个模块里。	抽一个 StudioServiceIdentity
三、各子系统要点
Agents

一把全局 asyncio.Lock 同时管所有会话，但有几条写路径没拿锁，cancel 和完成保存之间会互相覆盖。
每次状态变化都整条会话（包括 base64 图片）读出、改写、存回，并推给前端，前端再重新拉取一遍。
图编辑工具有两套（patch 和 proposal），校验不一致。MCP 的 graph_apply_patch 绕过了人工审批。
审批时只校验 graph revision，不校验 layout revision。
关键词路由的 "graph-builder-v1" 是演示代码，却是默认 provider，应移到测试里。
service.py 有 1472 行，建议拆成 session_store、approval_gate、tool_recorder、model_tools、run_loop 五个模块。
决策节点

被取代或过期的 exec 没有任何输出，下游流程会静默中断（decision.py:178）。
运行时依赖 Studio HTTP 服务，而且挂在 agents 服务下面。应该独立成 DecisionService。
前端

GraphWorkspaceInner 有 1100 行、20 多个 state。项目文档有三份副本：project、projectRef 和 React Flow 的 nodes/edges。
addSpec、history、uploadGraph 绕过了 revision 检查。
3D 骨架视图每一帧都重建整个场景，并泄漏 helper 的材质（SkeletonViewport.tsx:136）。
SDK

Zenoh 每个订阅都用 1ms 的 sleep 轮询 try_recv，CPU 占用随路由数线性增长。应改成回调加 call_soon_threadsafe。
retained_get 在 Zenoh 下只返回本进程写入的值，在内存版 transport 下返回全集群的值，测试通过的是生产环境不存在的行为。
set_rungraph 失败时没有日志（micro.py:494）。
codegen 脚本用 try/except 在两种生成方式之间回退（违反 AGENTS.md），而依赖 datamodel-code-generator 没有在 pixi 里声明，当前无法复现生成结果。
媒体

flow/scalar 预览用纯 Python 逐像素 struct.unpack，一帧 1080p 要秒级时间，应改用 numpy。
viz_wave 和 viz_text 关闭时不发 .detach，刷新页面后已删除的节点会重新出现。
四、工程治理
.github/workflows 里只有 dist 和 docs 两个工作流，没有 pytest、typecheck、lint 这类 CI。根目录 pytest 当前有 2 个失败。
quality_exceptions 的阈值是 844，实际只有 60，等于没设。.importlinter 指向已经不存在的 f8pystudio。ruff 只开了语法级规则。pyright 两套配置标准冲突（basic 和 strict）。
services/**/describe.json 被 gitignore，但服务发现优先读它，而且没有失效机制。已经和代码不一致：cppengine 的缓存缺了 8 个算子。测试也依赖这些本地缓存。
C++ 的 main.cpp 样板代码复制了约 10 份，应提供 run_service_main<Service>()。
建议路线图
第 1 周：止血。 修掉 P0 的 8 条 bug。加 Linux CI（pytest、typecheck、quality_exceptions），把阈值收紧到当前实际值。
协议收敛： 单例 EventStream 加独立的 presentation 通道；去掉重复的轮询；统一错误体；加 token 鉴权。
契约生成： 以 msgspec 为真源生成 TS；把 wire 约定写进 schema，覆盖 Python 和 C++ 两端。
去重： 把 f8pysdk.expr、DedupedErrorReporter、video_latest 移进 SDK；统一会话池和 SessionManager；删除 session/ 编译器和死路由；合并 naming 模块。
分层： 建 ProjectLifecycle 用例层和 StudioDatabase；拆分 agents 的 service.py 和 GraphWorkspace。
需要的话，我可以从 P0 开始逐条修复，每条附带测试。


provider_probe.py
已经确认你说的所有问题. 可以开始修复, 并进行改良
I'll work in phases, starting with the P0 bugs and a baseline test run so I don't mistake existing failures for regressions. First, the available tasks and the current test state:


Bash Inspect pixi tasks and web package scripts
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n '/\[tasks\]/,/^\[environments/p' pixi.toml | head -80; grep -n "^\[feature\..*tasks\]" -A12 pixi.toml | grep -iE "test|lint|type|quality|vitest|web" | head -30; cat packages/f8studio_web/package.json | head -40

OUT
58-[feature.test.dependencies]
59-pytest = ">=8.4.2,<9"
61-[feature.test.pypi-dependencies]
63-import-linter = ">=2.11,<3"
65:[feature.test.tasks]
66-pytest = "python -m pytest"
67-pytest_sdk = "python -m pytest -q packages/f8pysdk/tests"
70-lint = "ruff check packages/f8proclauncher/f8proclauncher packages/f8pymppose/f8pymppose packages/f8pyscript/f8pyscript packages/f8pyaudiofeat/f8pyaudiofeat packages/f8pydl/f8pydl packages/f8pysdk/f8pysdk packages/f8pyengine/f8pyengine packages/f8studio_core/f8studio_core packages/f8media_protocol/f8media_protocol packages/f8media_gateway/f8media_gateway packages/f8studio_server/f8studio_server"
71-typecheck = "basedpyright -p pyrightconfig.json"
72-quality_exceptions = "python scripts/quality/except_metrics.py packages --exclude-glob '**/tests/**' --max-broad 844 --max-silent 40"
73-lint_imports = "lint-imports"
148-cpp_test_release = "python scripts/cpp_ci.py test"
149-cpp_bench_latest_video = "python scripts/cpp_ci.py configure && python scripts/cpp_ci.py build-target f8cpp_bench_latest_video_transport && build/Release/bin/f8cpp_bench_latest_video_transport"
172-[feature.web-studio.pypi-dependencies]
188:[feature.web-studio.tasks]
193-studio_agent_model_smoke = "python scripts/web_studio/probe_agent_model.py"
196-unitymods_test = "python scripts/unitymods_ci.py test"
200-[feature.web-studio-dev.dependencies]
203:[feature.web-studio-dev.tasks]
204-studio_web_dev = "npm --prefix packages/f8studio_web run dev"
205-studio_web_build = "npm --prefix packages/f8studio_web run build"
207-[feature.web-studio-test.dependencies]
208-pytest = ">=8.4.2,<9"
211-[feature.web-studio-test.pypi-dependencies]
215:[feature.web-studio-test.tasks]
216-studio_core_test = "python -m pytest -q packages/f8studio_core/tests"
217-studio_server_test = "python -m pytest -q packages/f8studio_server/tests"
218-studio_media_gateway_test = "python -m pytest -q packages/f8media_gateway/tests"
219-studio_web_test = "npm --prefix packages/f8studio_web run test"
220-studio_web_typecheck = "npm --prefix packages/f8studio_web run typecheck"
{
  "name": "@feel8/studio-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite --host 127.0.0.1 --port 5173",
    "build": "tsc -b && vite build",
    "test": "vitest run",
    "test:e2e": "playwright test",
    "test:graph-performance": "playwright test --config playwright.graph-performance.config.ts",
    "typecheck": "tsc -b --pretty false"
  },
  "dependencies": {
    "@types/three": "^0.180.0",
    "@xyflow/react": "^12.11.6",
    "lucide-react": "^1.47.0",
    "monaco-editor": "^0.52.2",
    "osr-emu": "^0.7.0",
    "react": "^19.3.0",
    "react-dom": "^19.3.0",
    "three": "^0.180.0"
  },
  "devDependencies": {
    "@playwright/test": "^1.63.0",
    "@testing-library/jest-dom": "^7.0.1",
    "@testing-library/react": "^16.3.3",
    "@types/react": "^19.3.0",
    "@types/react-dom": "^19.3.0",
    "@vitejs/plugin-react": "^6.1.1",
    "jsdom": "^30.1.0",
    "typescript": "^7.0.2",
    "vite": "^8.3.0",
    "vitest": "^5.0.1"
  }
}
Bash Show pixi environments
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n '/^\[environments\]/,$p' pixi.toml; sed -n 215,235p pixi.toml

OUT
[environments]
# Pixi tasks are feature-scoped, not environment-scoped. If a task's feature is
# included by multiple environments (for example both `default` and
# `studio-runtime`), `pixi run <task>` without `-e` can resolve differently
# depending on the active Pixi environment. Keep `service.yml` launch entries
# explicit with `-e` for shared runtime tasks.
default = { features = ["python", "sdk", "engine", "web-studio", "web-studio-dev", "vision", "audio", "test", "debug"], solve-group = "default" }
studio-runtime = { features = ["python", "sdk", "engine", "web-studio", "vision", "audio", "launcher-runtime"], solve-group = "default" }
onnx = { features = ["python", "sdk", "onnx", "vision", "launcher-runtime"], solve-group = "onnx" }
mediapipe = { features = ["python", "sdk", "vision", "mediapipe", "launcher-runtime"], solve-group = "mediapipe" }
doc = { features = ["python", "doc"] }
cpp = { features = ["cpp"], solve-group = "cpp" }
ci = { features = ["python", "ci"], solve-group = "ci" }
launcher = { features = ["launcher"], solve-group = "launcher" }
web-studio = { features = ["python", "sdk", "web-studio", "web-studio-dev"], solve-group = "web-studio" }
web-studio-test = { features = ["python", "sdk", "web-studio", "web-studio-dev", "web-studio-test"], solve-group = "web-studio" }
web-studio-runtime = { features = ["python", "sdk", "engine"], solve-group = "web-studio" }
[feature.web-studio-test.tasks]
studio_core_test = "python -m pytest -q packages/f8studio_core/tests"
studio_server_test = "python -m pytest -q packages/f8studio_server/tests"
studio_media_gateway_test = "python -m pytest -q packages/f8media_gateway/tests"
studio_web_test = "npm --prefix packages/f8studio_web run test"
studio_web_typecheck = "npm --prefix packages/f8studio_web run typecheck"
studio_web_e2e = { cmd = "npm --prefix packages/f8studio_web run test:e2e", depends-on = ["studio_web_build"] }
studio_graph_bench = { cmd = "npm --prefix packages/f8studio_web run test:graph-performance", depends-on = ["studio_web_build"] }
studio_python_typecheck = "basedpyright -p pyrightconfig.web-studio.json"
studio_no_qt_check = "python scripts/web_studio/check_no_qt.py"
studio_dependency_probe = "python scripts/web_studio/probe_environment.py"
studio_runtime_probe = "python scripts/web_studio/probe_real_runtime.py"
studio_media_probe = "python scripts/web_studio/probe_webrtc_media.py"
studio_real_video_probe = "python scripts/web_studio/probe_real_video.py"
studio_media_bench = "python scripts/web_studio/benchmark_media.py"
studio_stream_restart_probe = "python scripts/web_studio/probe_stream_restart.py"
studio_p3_combined_bench = "python scripts/web_studio/benchmark_p3_combined.py"
studio_release_smoke = "python scripts/web_studio/verify_release_install.py"

[environments]
# Pixi tasks are feature-scoped, not environment-scoped. If a task's feature is
Now the baseline, running all suites in parallel:


Bash Baseline studio Python test suites
IN
cd /home/sxs/SS/Feel8/f8studio; (pixi run -e web-studio-test studio_core_test 2>&1 | tail -3) ; (pixi run -e web-studio-test studio_server_test 2>&1 | tail -5); (pixi run -e web-studio-test studio_media_gateway_test 2>&1 | tail -3)

OUT
Command running in background with ID: buaub82ny. Output is being written to: /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/buaub82ny.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /home/sxs/SS/Feel8/f8studio; directory changes made by the backgrounded command do not apply to subsequent commands.
Bash Baseline web tests and type checks
IN
cd /home/sxs/SS/Feel8/f8studio; (pixi run -e web-studio-test studio_web_typecheck 2>&1 | tail -5); (pixi run -e web-studio-test studio_web_test 2>&1 | tail -8); (pixi run typecheck 2>&1 | tail -5); (pixi run -e web-studio-test studio_python_typecheck 2>&1 | tail -5)

OUT
Command running in background with ID: b5vg414ww. Output is being written to: /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/b5vg414ww.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /home/sxs/SS/Feel8/f8studio; directory changes made by the backgrounded command do not apply to subsequent commands.
Bash Baseline full default pytest run
IN
cd /home/sxs/SS/Feel8/f8studio; pixi run pytest -q -x --co -q 2>&1 | tail -3; pixi run pytest -q -p no:cacheprovider 2>&1 | tail -15

OUT
Command running in background with ID: byckyjks9. Output is being written to: /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/byckyjks9.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /home/sxs/SS/Feel8/f8studio; directory changes made by the backgrounded command do not apply to subsequent commands.
Bash Check dist env rewrite and engine yml history
IN
cd /home/sxs/SS/Feel8/f8studio; grep -n "default\|studio-runtime\|web-studio-runtime" scripts/dist_ci.py | head -30; git log --oneline -3 -- services/f8/engine/service.yml; git log -1 -p --stat -- services/f8/engine/service.yml | head -40

OUT
31:DEV_RUNTIME_ENVIRONMENT_NAME = "default"
32:DIST_RUNTIME_ENVIRONMENT_NAME = "studio-runtime"
f37fbe90 feat(studio): deliver Qt-free web studio through P3.5
7f48bb90 Improve dist build and start up experience.
f786a661 feat: add service documentation generation script using `zensical`
commit f37fbe9084e8eb45360d933a1a7355eec7cf64e4
Author: sis92 <feel8.fun@gmail.com>
Date:   Tue Sep 22 01:48:29 2026 -0400

    feat(studio): deliver Qt-free web studio through P3.5
---
 services/f8/engine/service.yml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/services/f8/engine/service.yml b/services/f8/engine/service.yml
index 610a4140..f856f776 100644
--- a/services/f8/engine/service.yml
+++ b/services/f8/engine/service.yml
@@ -4,6 +4,6 @@ label: PyEngine
 version: 0.0.1
 launch:
   command: pixi
-  args: ["run", "-e", "default", "f8pyengine"]
+  args: ["run", "-e", "web-studio-runtime", "f8pyengine"]
   env: {}
   workdir: "../../../"
Bash Read dist_ci environment rewrite code
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n 20,40p scripts/dist_ci.py; sed -n 250,335p scripts/dist_ci.py; grep -n "rewrite\|RUNTIME_ENVIRONMENT" tests/test_dist_ci.py | head

OUT
CPP_PRESET_PATH = DEFAULT_CPP_PRESET_PATH
CPP_PRESET_CANDIDATES = (
    REPO_ROOT / "build" / "generators" / "CMakePresets.json",
    DEFAULT_CPP_PRESET_PATH,
)
CPP_BUILD_PRESET_NAME = "conan-release"
LOCAL_EDITABLE_PATH_PREFIXES = ("packages/", "external/f8unitymods")
# C++ runtime deploy targets are owned by CMake's f8_deploy_all_runtime aggregator.
CPP_DEPLOY_ALL_TARGET = "f8_deploy_all_runtime"
LAUNCHER_ENVIRONMENT_NAME = "launcher"
LAUNCHER_RUNTIME_FEATURE = "launcher-runtime"
DEV_RUNTIME_ENVIRONMENT_NAME = "default"
DIST_RUNTIME_ENVIRONMENT_NAME = "studio-runtime"
WEB_BUNDLE_SOURCE = REPO_ROOT / "packages" / "f8studio_web" / "dist"
WEB_BUNDLE_PACKAGE_DIR = REPO_ROOT / "packages" / "f8studio_server" / "f8studio_server" / "web_dist"


@dataclass(frozen=True)
class LocalEditablePackage:
    package_dir: str
    feature_name: str
    if not isinstance(environments_table, dict):
        raise ValueError(f"Expected [environments] table in {pixi_toml_path}")

    runtime_environment_names: list[str] = []
    for environment_name, environment_spec in environments_table.items():
        if not isinstance(environment_name, str):
            raise ValueError(f"Environment name must be a string in {pixi_toml_path}")
        if not isinstance(environment_spec, dict):
            continue
        features = environment_spec.get("features")
        if not isinstance(features, list):
            continue
        if LAUNCHER_RUNTIME_FEATURE in features:
            runtime_environment_names.append(environment_name)

    if not runtime_environment_names:
        raise ValueError(
            f"No runtime environments were marked with feature '{LAUNCHER_RUNTIME_FEATURE}' in {pixi_toml_path}"
        )

    return runtime_environment_names


def _split_manifest_sections(pixi_text: str) -> list[tuple[str | None, str]]:
    section_matches = list(re.finditer(r"(?m)^\[([^\[\]\n]+)\]\s*$", pixi_text))
    if not section_matches:
        return [(None, pixi_text)]

    sections: list[tuple[str | None, str]] = []
    if section_matches[0].start() > 0:
        sections.append((None, pixi_text[: section_matches[0].start()]))

    for index, match in enumerate(section_matches):
        section_name = match.group(1)
        section_end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(pixi_text)
        sections.append((section_name, pixi_text[match.start() : section_end]))

    return sections


def _feature_name_from_section(section_name: str) -> str | None:
    if not section_name.startswith("feature."):
        return None
    parts = section_name.split(".")
    if len(parts) < 2 or parts[1] == "":
        return None
    return parts[1]


def _rewrite_service_entry_environment_args(
    service_text: str,
    *,
    source_environment_name: str,
    target_environment_name: str,
) -> str:
    pattern = re.compile(
        r'(\bargs:\s*\[\s*["\']run["\']\s*,\s*["\']-e["\']\s*,\s*["\'])'
        + re.escape(source_environment_name)
        + r'(["\'])'
    )
    return pattern.sub(r"\1" + target_environment_name + r"\2", service_text)


def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        original_text = service_entry_path.read_text(encoding="utf-8")
        rewritten_text = _rewrite_service_entry_environment_args(
            original_text,
            source_environment_name=DEV_RUNTIME_ENVIRONMENT_NAME,
            target_environment_name=DIST_RUNTIME_ENVIRONMENT_NAME,
        )
        if rewritten_text == original_text:
            continue
        service_entry_path.write_text(rewritten_text, encoding="utf-8")
        rewritten_paths.append(service_entry_path)
    return rewritten_paths


def _copy_dist_config(dist_dir: Path) -> Path | None:
    config_root = REPO_ROOT / "config"
    if not config_root.is_dir():
        return None
    dist_config_root = dist_dir / "config"
    shutil.copytree(config_root, dist_config_root, dirs_exist_ok=True)
    return dist_config_root
171:    def test_root_manifest_discovers_and_rewrites_unitymods_package(self) -> None:
402:    def test_rewrite_service_entry_environment_args_swaps_default_for_dist_runtime(self) -> None:
409:        rewritten = self.module._rewrite_service_entry_environment_args(
418:    def test_rewrite_dist_service_entries_only_updates_default_runtime_services(self) -> None:
437:        rewritten_paths = self.module._rewrite_dist_service_entries(services_root)
Bash Read dist rewrite tests and call sites
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n 400,450p tests/test_dist_ci.py; grep -n "_rewrite_dist_service_entries\|_runtime_environment_names\|def _.*runtime_environment" scripts/dist_ci.py

OUT
        self.assertIn('platforms = ["win-64"]', filtered)

    def test_rewrite_service_entry_environment_args_swaps_default_for_dist_runtime(self) -> None:
        service_text = (
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "default", "f8pyengine"]\n'
        )

        rewritten = self.module._rewrite_service_entry_environment_args(
            service_text,
            source_environment_name="default",
            target_environment_name="studio-runtime",
        )

        self.assertIn('"run", "-e", "studio-runtime", "f8pyengine"', rewritten)
        self.assertNotIn('"run", "-e", "default", "f8pyengine"', rewritten)

    def test_rewrite_dist_service_entries_only_updates_default_runtime_services(self) -> None:
        services_root = self.root / "services"
        engine_service_path = services_root / "f8" / "engine" / "service.yml"
        detector_service_path = services_root / "f8" / "dl" / "detector" / "service.yml"
        engine_service_path.parent.mkdir(parents=True, exist_ok=True)
        detector_service_path.parent.mkdir(parents=True, exist_ok=True)
        engine_service_path.write_text(
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "default", "f8pyengine"]\n',
            encoding="utf-8",
        )
        detector_service_path.write_text(
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "onnx", "f8pydl_detector"]\n',
            encoding="utf-8",
        )

        rewritten_paths = self.module._rewrite_dist_service_entries(services_root)

        self.assertEqual(rewritten_paths, [engine_service_path])
        self.assertIn(
            '"run", "-e", "studio-runtime", "f8pyengine"',
            engine_service_path.read_text(encoding="utf-8"),
        )
        self.assertIn(
            '"run", "-e", "onnx", "f8pydl_detector"',
            detector_service_path.read_text(encoding="utf-8"),
        )

    def test_copy_dist_config_copies_service_discovery_policy(self) -> None:
        config_root = self.root / "config"
245:def _discover_launcher_runtime_environments(*, pixi_toml_path: Path = PIXI_TOML_PATH) -> list[str]:
313:def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
400:def _filter_dist_environments(pixi_text: str, runtime_environment_names: list[str]) -> str:
517:def _runtime_environment_wheels(
725:    _rewrite_dist_service_entries(dist_dir / "services")
Bash Read dist rewrite call site
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n 700,735p scripts/dist_ci.py

OUT
    parser.add_argument(
        "--reuse-unitymods-assets",
        action="store_true",
        help="Bundle previously built f8unitymods dist assets instead of rebuilding them.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    _build_cpp_runtime()
    _stage_web_bundle()

    platform_tag, platform_dir = _platform_info()
    dist_base_dir = REPO_ROOT / "build" / "dist"
    dist_name = f"f8studio-{platform_tag}"
    dist_dir = dist_base_dir / dist_name

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    (dist_dir / "services").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "services", dist_dir / "services", dirs_exist_ok=True)
    _rewrite_dist_service_entries(dist_dir / "services")
    _copy_dist_config(dist_dir)
    _bundle_unitymods_assets(
        dist_dir,
        build_assets=not bool(args.reuse_unitymods_assets),
    )

    wheels_dir = dist_dir / "wheels"
    runtime_environment_names = _discover_launcher_runtime_environments()
    runtime_feature_names = _discover_environment_feature_names(environment_names=runtime_environment_names)
    local_packages = _discover_local_editable_packages(
Now the fix: dist rewrites both dev environments, then fails the build if any entry still references an environment that won't exist in the dist.


Bash Rewrite dist env rewrite and add validation
IN
cd /home/sxs/SS/Feel8/f8studio; python3 - <<'EOF'
from pathlib import Path
p=Path("scripts/dist_ci.py"); s=p.read_text()
s=s.replace('DEV_RUNTIME_ENVIRONMENT_NAME = "default"\n','# Dev service entries may target either the full dev environment or the slim\n# web-studio runtime; both collapse onto the single dist runtime environment.\nDEV_RUNTIME_ENVIRONMENT_NAMES = ("default", "web-studio-runtime")\n',1)
old='''def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        original_text = service_entry_path.read_text(encoding="utf-8")
        rewritten_text = _rewrite_service_entry_environment_args(
            original_text,
            source_environment_name=DEV_RUNTIME_ENVIRONMENT_NAME,
            target_environment_name=DIST_RUNTIME_ENVIRONMENT_NAME,
        )
        if rewritten_text == original_text:
'''
new='''def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        original_text = service_entry_path.read_text(encoding="utf-8")
        rewritten_text = original_text
        for source_environment_name in DEV_RUNTIME_ENVIRONMENT_NAMES:
            rewritten_text = _rewrite_service_entry_environment_args(
                rewritten_text,
                source_environment_name=source_environment_name,
                target_environment_name=DIST_RUNTIME_ENVIRONMENT_NAME,
            )
        if rewritten_text == original_text:
'''
assert old in s; s=s.replace(old,new)
old2='''def _copy_dist_config(dist_dir: Path) -> Path | None:'''
new2='''_SERVICE_ENTRY_ENVIRONMENT_PATTERN = re.compile(r'\\bargs:\\s*\\[\\s*["\\']run["\\']\\s*,\\s*["\\']-e["\\']\\s*,\\s*["\\']([^"\\']+)["\\']')


def _validate_dist_service_environments(services_root: Path, runtime_environment_names: list[str]) -> None:
    allowed = frozenset(runtime_environment_names)
    problems: list[str] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        for match in _SERVICE_ENTRY_ENVIRONMENT_PATTERN.finditer(service_entry_path.read_text(encoding="utf-8")):
            if match.group(1) not in allowed:
                problems.append(f"{service_entry_path}: pixi environment '{match.group(1)}'")
    if problems:
        raise ValueError(
            "Dist service entries reference environments that are not shipped "
            f"(shipped: {sorted(allowed)}):\\n" + "\\n".join(problems)
        )


def _copy_dist_config(dist_dir: Path) -> Path | None:'''
assert old2 in s; s=s.replace(old2,new2,1)
old3='''    _rewrite_dist_service_entries(dist_dir / "services")
    _copy_dist_config(dist_dir)
    _bundle_unitymods_assets(
        dist_dir,
        build_assets=not bool(args.reuse_unitymods_assets),
    )

    wheels_dir = dist_dir / "wheels"
    runtime_environment_names = _discover_launcher_runtime_environments()
'''
new3='''    _rewrite_dist_service_entries(dist_dir / "services")
    runtime_environment_names = _discover_launcher_runtime_environments()
    _validate_dist_service_environments(dist_dir / "services", runtime_environment_names)
    _copy_dist_config(dist_dir)
    _bundle_unitymods_assets(
        dist_dir,
        build_assets=not bool(args.reuse_unitymods_assets),
    )

    wheels_dir = dist_dir / "wheels"
'''
assert old3 in s; s=s.replace(old3,new3)
p.write_text(s)
EOF
grep -n "DEV_RUNTIME_ENVIRONMENT_NAME\b" -r scripts tests

OUT
(Bash completed with no output)
Background command "Baseline web tests and type checks" completed (exit code 0)
Background command "Baseline studio Python test suites" completed (exit code 0)
Bash Read baseline results
IN
cat /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/b5vg414ww.output /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/buaub82ny.output

OUT
✨ Pixi task (studio_web_typecheck in web-studio-test): npm --prefix packages/f8studio_web run typecheck

> @feel8/studio-web@0.1.0 typecheck
> tsc -b --pretty false

      Tests  85 passed (85)
   Start at  14:25:18
   Duration  5.64s (environment 52%, tests 26%, transform 9%, import 7%, setup 6%)

Environment  jsdom was created 16 times · 24.10s total, 52% of tracked time
             create it once per worker with pool: 'vmThreads' (keeps per-file isolation) or isolate: false (shares it across files)
             learn more: https://vitest.dev/guide/improving-performance#test-environments

  /home/sxs/SS/Feel8/f8studio/packages/f8pydl/f8pydl/service_node.py:926:52 - error: Cannot access attribute "stream_epoch" for class "VideoFramePacket"
    Attribute "stream_epoch" is unknown (reportAttributeAccessIssue)
  /home/sxs/SS/Feel8/f8studio/packages/f8pydl/f8pydl/service_node.py:943:52 - error: Cannot access attribute "stream_epoch" for class "VideoFramePacket"
    Attribute "stream_epoch" is unknown (reportAttributeAccessIssue)
2 errors, 0 warnings, 0 notes
✨ Pixi task (studio_python_typecheck in web-studio-test): basedpyright -p pyrightconfig.web-studio.json
0 errors, 0 warnings, 0 notes

[exited with code 0]
✨ Pixi task (studio_core_test in web-studio-test): python -m pytest -q packages/f8studio_core/tests
..................................                                       [100%]
34 passed in 0.11s
FAILED packages/f8studio_server/tests/test_app.py::test_event_websocket_snapshot_commit_replay_and_origin
FAILED packages/f8studio_server/tests/test_app.py::test_deploy_api_runs_job_through_injected_runtime
FAILED packages/f8studio_server/tests/test_app.py::test_runtime_disconnect_maps_to_service_unavailable
FAILED packages/f8studio_server/tests/test_workspace_services.py::test_python_script_editor_uses_injected_api_and_dynamic_bindings
10 failed, 132 passed in 28.60s
✨ Pixi task (studio_media_gateway_test in web-studio-test): python -m pytest -q packages/f8media_gateway/tests
........................                                                 [100%]
24 passed in 4.12s

[exited with code 0]
The baseline has 10 failing server tests before any of my changes. I need to know why before building on top:


Bash Show server test failure reasons
IN
cd /home/sxs/SS/Feel8/f8studio; pixi run -e web-studio-test python -m pytest -q packages/f8studio_server/tests -p no:cacheprovider 2>&1 | grep -E "^(FAILED|E  )" | head -40

OUT
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085516, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085527, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085531, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085543, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085564, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085609, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085856, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085902, health reported PID 3386201
E                   f8media_protocol.client.MediaGatewayUnavailable: Media Gateway process mismatch: started PID 4085967, health reported PID 3386201
E       ModuleNotFoundError: No module named 'f8pyengine'
FAILED packages/f8studio_server/tests/test_app.py::test_runtime_state_read_returns_retained_node_values
FAILED packages/f8studio_server/tests/test_app.py::test_media_api_rejects_invalid_source_and_quality
FAILED packages/f8studio_server/tests/test_app.py::test_media_rtc_configuration_defaults_to_direct_ice
FAILED packages/f8studio_server/tests/test_app.py::test_media_rtc_configuration_exposes_turn_relay_config
FAILED packages/f8studio_server/tests/test_app.py::test_media_api_unknown_session_is_not_found
FAILED packages/f8studio_server/tests/test_app.py::test_overlay_api_validates_identity_and_reports_monitor_metrics
FAILED packages/f8studio_server/tests/test_app.py::test_event_websocket_snapshot_commit_replay_and_origin
FAILED packages/f8studio_server/tests/test_app.py::test_deploy_api_runs_job_through_injected_runtime
FAILED packages/f8studio_server/tests/test_app.py::test_runtime_disconnect_maps_to_service_unavailable
FAILED packages/f8studio_server/tests/test_workspace_services.py::test_python_script_editor_uses_injected_api_and_dynamic_bindings
Bash Identify process holding media gateway port
IN
ps -o pid,etime,args -p 3386201 | cat; grep -rn "gateway_port\|8211\|media_gateway.*port" /home/sxs/SS/Feel8/f8studio/packages/f8studio_server/f8studio_server/app.py /home/sxs/SS/Feel8/f8studio/packages/f8media_protocol/f8media_protocol/client.py | head

OUT
    PID     ELAPSED COMMAND
3386201    20:52:35 /home/sxs/SS/Feel8/f8studio/.pixi/envs/web-studio/bin/python -m f8media_gateway --host 127.0.0.1 --port 8211 --exit-on-stdin-close
/home/sxs/SS/Feel8/f8studio/packages/f8media_protocol/f8media_protocol/client.py:47:    base_url: str = "http://127.0.0.1:8211"
Background command "Baseline full default pytest run" completed (exit code 0)
Bash Read default pytest baseline
IN
tail -25 /tmp/claude-1000/-home-sxs-SS-Feel8-f8studio/e579b5ef-94f8-4e3f-a9fd-f89f88520bf4/tasks/byckyjks9.output; cd /home/sxs/SS/Feel8/f8studio; cat pytest.ini

OUT
    from cryptography.hazmat.primitives.asymmetric.dh import DHParameterNumbers

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED packages/f8studio_server/tests/test_app.py::test_runtime_state_read_returns_retained_node_values
FAILED packages/f8studio_server/tests/test_app.py::test_media_api_rejects_invalid_source_and_quality
FAILED packages/f8studio_server/tests/test_app.py::test_media_rtc_configuration_defaults_to_direct_ice
FAILED packages/f8studio_server/tests/test_app.py::test_media_rtc_configuration_exposes_turn_relay_config
FAILED packages/f8studio_server/tests/test_app.py::test_media_api_unknown_session_is_not_found
FAILED packages/f8studio_server/tests/test_app.py::test_overlay_api_validates_identity_and_reports_monitor_metrics
FAILED packages/f8studio_server/tests/test_app.py::test_event_websocket_snapshot_commit_replay_and_origin
FAILED packages/f8studio_server/tests/test_app.py::test_deploy_api_runs_job_through_injected_runtime
FAILED packages/f8studio_server/tests/test_app.py::test_runtime_disconnect_maps_to_service_unavailable
FAILED tests/test_cpp_ci.py::CppCiConfigureTest::test_test_command_configures_builds_and_runs_gtest_target
FAILED tests/test_cppengine_operator_coverage.py::CppEngineOperatorCoverageTest::test_cppengine_describes_all_pyengine_operators_except_python_script
11 failed, 997 passed, 1 warning in 62.51s (0:01:02)

[exited with code 0]
[pytest]
norecursedirs =
    tmp
    .tmp
    build
filterwarnings =
    ignore:distutils Version classes are deprecated\. Use packaging\.version instead\.:DeprecationWarning
Bash Find gateway process wiring and port
IN
cd /home/sxs/SS/Feel8/f8studio/packages; grep -n "8211\|gateway_url\|MediaGatewayProcess\|def create_app\|media_gateway" f8studio_server/f8studio_server/app.py f8studio_server/f8studio_server/application.py f8studio_server/f8studio_server/__main__.py | head -30; grep -n "class .*Process\|port\|def start\|mismatch" f8media_protocol/f8media_protocol/client.py | head -40

OUT
f8studio_server/f8studio_server/__main__.py:28:    parser.add_argument("--media-gateway-url", default="http://127.0.0.1:8211")
f8studio_server/f8studio_server/__main__.py:86:                    base_url=args.media_gateway_url,
f8studio_server/f8studio_server/__main__.py:87:                    manage_process=not args.external_media_gateway,
f8studio_server/f8studio_server/__main__.py:95:                media_gateway=gateway,
f8studio_server/f8studio_server/application.py:54:        media_gateway: MediaGateway | None = None,
f8studio_server/f8studio_server/application.py:88:        if media_gateway is None:
f8studio_server/f8studio_server/application.py:89:            media_gateway = RemoteMediaGateway()
f8studio_server/f8studio_server/application.py:90:        self.media_gateway = media_gateway
f8studio_server/f8studio_server/application.py:122:        await self.media_gateway.start()
f8studio_server/f8studio_server/application.py:132:        await self.media_gateway.close()
f8studio_server/f8studio_server/app.py:151:def create_app(
f8studio_server/f8studio_server/app.py:159:    media_gateway: MediaGateway | None = None,
f8studio_server/f8studio_server/app.py:168:        runtime_config=runtime_config, service_roots=service_roots, media_gateway=media_gateway,
f8studio_server/f8studio_server/app.py:226:    async def media_gateway_request_error(_request: Request, exc: MediaGatewayRequestError) -> JSONResponse:
f8studio_server/f8studio_server/app.py:235:    async def media_gateway_unavailable(_request: Request, exc: MediaGatewayUnavailable) -> JSONResponse:
f8studio_server/f8studio_server/app.py:359:        return _json_value(await studio.media_gateway.create_video_session(offer))
f8studio_server/f8studio_server/app.py:364:        return _json_value(await studio.media_gateway.create_audio_session(offer))
f8studio_server/f8studio_server/app.py:368:        closed = await studio.media_gateway.close_audio_session(session_id)
f8studio_server/f8studio_server/app.py:376:        await studio.media_gateway.publish_overlay(result)
f8studio_server/f8studio_server/app.py:381:        mapping = await studio.media_gateway.frame_mapping(session_id, media_timestamp)
f8studio_server/f8studio_server/app.py:388:        return _json_value(await studio.media_gateway.metrics())
f8studio_server/f8studio_server/app.py:391:    async def media_gateway_health() -> F8JsonValue:
f8studio_server/f8studio_server/app.py:392:        return _json_value(await studio.media_gateway.health())
f8studio_server/f8studio_server/app.py:397:            return _json_value(await studio.media_gateway.sample(source, x=x, y=y))
f8studio_server/f8studio_server/app.py:403:        closed = await studio.media_gateway.close_video_session(session_id)
1:from __future__ import annotations
3:import asyncio
4:import logging
5:import sys
6:from dataclasses import dataclass
7:from typing import TypeVar, overload
8:from urllib.parse import quote
10:import httpx
11:import msgspec
13:from f8pysdk.specs import F8JsonValue
15:from .models import (
64:    async def start(self) -> None:
71:                host, port = _loopback_endpoint(self._config.base_url)
78:                    "--port",
79:                    str(port),
103:                        "Media Gateway service mismatch: "
108:                        "Media Gateway protocol mismatch: "
113:                        "Media Gateway process mismatch: "
114:                        f"started PID {process.pid}, health reported PID {health.process_id}"
267:    return host, parsed.port or 80
Bash Read gateway client config and startup
IN
cd /home/sxs/SS/Feel8/f8studio/packages; sed -n 38,135p f8media_protocol/f8media_protocol/client.py; sed -n 255,270p f8media_protocol/f8media_protocol/client.py; sed -n 20,40p f8studio_server/f8studio_server/__main__.py; sed -n 75,100p f8studio_server/f8studio_server/__main__.py

OUT
class MediaGatewayRequestError(RuntimeError):
    def __init__(self, status_code: int, detail: F8JsonValue) -> None:
        super().__init__(f"Media Gateway request failed with HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class RemoteMediaGatewayConfig:
    base_url: str = "http://127.0.0.1:8211"
    manage_process: bool = True
    startup_timeout_s: float = 10.0


class RemoteMediaGateway:
    def __init__(
        self,
        config: RemoteMediaGatewayConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or RemoteMediaGatewayConfig()
        self._client = client
        self._owns_client = client is None
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._config.base_url, timeout=5.0)
        try:
            if self._config.manage_process:
                if self._process is not None:
                    raise RuntimeError("Media Gateway process is already running")
                host, port = _loopback_endpoint(self._config.base_url)
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "f8media_gateway",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--exit-on-stdin-close",
                    stdin=asyncio.subprocess.PIPE,
                )
            await self._wait_until_ready()
        except Exception:
            logger.exception("Media Gateway failed to start base_url=%s", self._config.base_url)
            await self.close()
            raise

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._config.startup_timeout_s
        last_error = "gateway did not respond"
        while asyncio.get_running_loop().time() < deadline:
            process = self._process
            if process is not None and process.returncode is not None:
                raise MediaGatewayUnavailable(f"Media Gateway exited during startup with code {process.returncode}")
            try:
                health = await self.health()
            except MediaGatewayUnavailable as exc:
                last_error = str(exc)
            else:
                if health.service != "f8media-gateway":
                    raise MediaGatewayUnavailable(
                        "Media Gateway service mismatch: "
                        f"expected 'f8media-gateway', received {health.service!r}"
                    )
                if health.protocol_version != MEDIA_API_VERSION:
                    raise MediaGatewayUnavailable(
                        "Media Gateway protocol mismatch: "
                        f"expected {MEDIA_API_VERSION}, received {health.protocol_version}"
                    )
                if process is not None and health.process_id != process.pid:
                    raise MediaGatewayUnavailable(
                        "Media Gateway process mismatch: "
                        f"started PID {process.pid}, health reported PID {health.process_id}"
                    )
                if health.status == "ok":
                    return
                last_error = f"unexpected health status {health.status!r}"
            await asyncio.sleep(0.05)
        raise MediaGatewayUnavailable(f"Media Gateway startup timed out: {last_error}")

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and self._owns_client:
            await client.aclose()
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
            raise MediaGatewayUnavailable(f"Media Gateway returned malformed {response_type.__name__}") from exc


def _loopback_endpoint(base_url: str) -> tuple[str, int]:
    parsed = httpx.URL(base_url)
    if parsed.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Media Gateway URL must use a loopback host")
    if parsed.scheme != "http":
        raise ValueError("Media Gateway URL must use HTTP")
    if parsed.path not in {"", "/"} or parsed.query:
        raise ValueError("Media Gateway URL must not contain a path or query")
    host = "::1" if parsed.host == "::1" else "127.0.0.1"
    return host, parsed.port or 80


__all__ = [
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Trusted HTTP Host/Origin hostname. Repeat for multiple names when binding a wildcard address.",
    )
    parser.add_argument("--port", default=8210, type=int)
    parser.add_argument("--web-dist", type=Path)
    parser.add_argument("--media-gateway-url", default="http://127.0.0.1:8211")
    parser.add_argument("--external-media-gateway", action="store_true")
    parser.add_argument(
        "--turn-url",
        action="append",
        default=[],
        help="Browser TURN URL. Repeat to provide fallback transports.",
    )
    parser.add_argument("--turn-username")
    parser.add_argument("--turn-credential")
    parser.add_argument(
        "--force-turn",
        action="store_true",
            username=turn_username,
            credential=turn_credential,
        )
    rtc_configuration = BrowserRtcConfiguration(
        ice_servers=() if ice_server is None else (ice_server,),
        ice_transport_policy="relay" if args.force_turn else "all",
    )
    try:
        with single_server_instance():
            gateway = RemoteMediaGateway(
                RemoteMediaGatewayConfig(
                    base_url=args.media_gateway_url,
                    manage_process=not args.external_media_gateway,
                )
            )
            allowed_hosts = {*DEFAULT_ALLOWED_HOSTS, *configured_allowed_hosts}
            if host not in {"0.0.0.0", "::"}:
                allowed_hosts.add(host)
            app = create_app(
                web_dist=args.web_dist,
                media_gateway=gateway,
                allowed_hosts=tuple(allowed_hosts),
                rtc_configuration=rtc_configuration,
            )
            local_host = "127.0.0.1" if host == "0.0.0.0" else "::1" if host == "::" else host
            url_host = f"[{local_host}]" if ":" in local_host else local_host
Bash Find all gateway config call sites
IN
cd /home/sxs/SS/Feel8/f8studio; grep -rn "RemoteMediaGatewayConfig\|external-media-gateway\|external_media_gateway\|media-gateway-url\|manage_process\|RemoteMediaGateway(" --include=*.py --include=*.ts --include=*.yml --include=*.toml . 2>/dev/null | grep -v "^./.pixi\|^./build\|^./docs\|^./site\|node_modules"

OUT
(eval):1: no matches found: --include=*.py
Bash Find gateway config call sites via git grep
IN
cd /home/sxs/SS/Feel8/f8studio; git grep -n -E "RemoteMediaGatewayConfig|external-media-gateway|external_media_gateway|media-gateway-url|media_gateway_url|manage_process|RemoteMediaGateway\(" -- ':!docs' ':!site'

OUT
packages/f8media_gateway/tests/test_client.py:14:    RemoteMediaGatewayConfig,
packages/f8media_gateway/tests/test_client.py:29:        gateway = RemoteMediaGateway(
packages/f8media_gateway/tests/test_client.py:30:            RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
packages/f8media_gateway/tests/test_client.py:67:        gateway = RemoteMediaGateway(
packages/f8media_gateway/tests/test_client.py:68:            RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
packages/f8media_gateway/tests/test_client.py:84:        occupying_gateway = RemoteMediaGateway(
packages/f8media_gateway/tests/test_client.py:85:            RemoteMediaGatewayConfig(
packages/f8media_gateway/tests/test_client.py:87:                manage_process=True,
packages/f8media_gateway/tests/test_client.py:90:        conflicting_gateway = RemoteMediaGateway(
packages/f8media_gateway/tests/test_client.py:91:            RemoteMediaGatewayConfig(
packages/f8media_gateway/tests/test_client.py:93:                manage_process=True,
packages/f8media_gateway/tests/test_client.py:113:        gateway = RemoteMediaGateway(
packages/f8media_gateway/tests/test_client.py:114:            RemoteMediaGatewayConfig(
packages/f8media_gateway/tests/test_client.py:116:                manage_process=True,
packages/f8media_protocol/f8media_protocol/client.py:46:class RemoteMediaGatewayConfig:
packages/f8media_protocol/f8media_protocol/client.py:48:    manage_process: bool = True
packages/f8media_protocol/f8media_protocol/client.py:55:        config: RemoteMediaGatewayConfig | None = None,
packages/f8media_protocol/f8media_protocol/client.py:59:        self._config = config or RemoteMediaGatewayConfig()
packages/f8media_protocol/f8media_protocol/client.py:68:            if self._config.manage_process:
packages/f8media_protocol/f8media_protocol/client.py:274:    "RemoteMediaGatewayConfig",
packages/f8studio_server/f8studio_server/__main__.py:10:from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig
packages/f8studio_server/f8studio_server/__main__.py:28:    parser.add_argument("--media-gateway-url", default="http://127.0.0.1:8211")
packages/f8studio_server/f8studio_server/__main__.py:29:    parser.add_argument("--external-media-gateway", action="store_true")
packages/f8studio_server/f8studio_server/__main__.py:84:            gateway = RemoteMediaGateway(
packages/f8studio_server/f8studio_server/__main__.py:85:                RemoteMediaGatewayConfig(
packages/f8studio_server/f8studio_server/__main__.py:86:                    base_url=args.media_gateway_url,
packages/f8studio_server/f8studio_server/__main__.py:87:                    manage_process=not args.external_media_gateway,
packages/f8studio_server/f8studio_server/application.py:89:            media_gateway = RemoteMediaGateway()
packages/f8studio_server/tests/test_app.py:16:from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig
packages/f8studio_server/tests/test_app.py:465:    gateway = RemoteMediaGateway(
packages/f8studio_server/tests/test_app.py:466:        RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
packages/f8studio_server/tests/test_app.py:524:    gateway = RemoteMediaGateway(
packages/f8studio_server/tests/test_app.py:525:        RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
Bash Read gateway process tests
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n 75,135p packages/f8media_gateway/tests/test_client.py

OUT
            await gateway.close()
            await client.aclose()

    asyncio.run(scenario())


def test_managed_gateway_rejects_health_from_another_process() -> None:
    async def scenario() -> None:
        port = unused_loopback_port()
        occupying_gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
            )
        )
        conflicting_gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
            )
        )
        try:
            await occupying_gateway.start()
            with pytest.raises(
                MediaGatewayUnavailable,
                match="process mismatch|exited during startup",
            ):
                await conflicting_gateway.start()
        finally:
            await conflicting_gateway.close()
            await occupying_gateway.close()

    asyncio.run(scenario())


def test_managed_gateway_runs_in_a_separate_process_and_stops() -> None:
    async def scenario() -> None:
        port = unused_loopback_port()
        gateway = RemoteMediaGateway(
            RemoteMediaGatewayConfig(
                base_url=f"http://127.0.0.1:{port}",
                manage_process=True,
                startup_timeout_s=10.0,
            )
        )
        process_id = 0
        try:
            await gateway.start()
            health = await gateway.health()
            process_id = health.process_id
            assert process_id != os.getpid()
            process = gateway._process
            assert process is not None
            assert process.pid == process_id
            assert process.returncode is None
        finally:
            await gateway.close()
        assert process_id > 0
        assert process is not None
        assert process.returncode is not None
The existing tests already pass explicit URLs, so a None default meaning "managed, pick a free port" is backward compatible.


Bash Managed gateway picks a free loopback port
IN
cd /home/sxs/SS/Feel8/f8studio; python3 - <<'EOF'
from pathlib import Path
p=Path("packages/f8media_protocol/f8media_protocol/client.py"); s=p.read_text()
old='''@dataclass(frozen=True)
class RemoteMediaGatewayConfig:
    base_url: str = "http://127.0.0.1:8211"
    manage_process: bool = True
    startup_timeout_s: float = 10.0
'''
new='''@dataclass(frozen=True)
class RemoteMediaGatewayConfig:
    # ``None`` lets a managed gateway bind a free loopback port. The gateway is
    # private to Studio (browsers reach it through the Studio proxy), so a fixed
    # port only creates collisions between instances and test runs.
    base_url: str | None = None
    manage_process: bool = True
    startup_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.base_url is None and not self.manage_process:
            raise ValueError("an external Media Gateway requires base_url")
'''
assert old in s; s=s.replace(old,new)
old='''    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._config.base_url, timeout=5.0)
        try:
            if self._config.manage_process:
                if self._process is not None:
                    raise RuntimeError("Media Gateway process is already running")
                host, port = _loopback_endpoint(self._config.base_url)
'''
new='''    async def start(self) -> None:
        base_url = self._config.base_url
        if base_url is None:
            base_url = f"http://127.0.0.1:{_unused_loopback_port()}"
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
        try:
            if self._config.manage_process:
                if self._process is not None:
                    raise RuntimeError("Media Gateway process is already running")
                host, port = _loopback_endpoint(base_url)
'''
assert old in s; s=s.replace(old,new)
s=s.replace('''            logger.exception("Media Gateway failed to start base_url=%s", self._config.base_url)''','''            logger.exception("Media Gateway failed to start base_url=%s", base_url)''')
old='''def _loopback_endpoint(base_url: str) -> tuple[str, int]:'''
new='''def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _loopback_endpoint(base_url: str) -> tuple[str, int]:'''
s=s.replace(old,new)
s=s.replace("import logging\nimport sys\n","import logging\nimport socket\nimport sys\n")
p.write_text(s)

p=Path("packages/f8studio_server/f8studio_server/__main__.py"); s=p.read_text()
old='''    parser.add_argument("--media-gateway-url", default="http://127.0.0.1:8211")
    parser.add_argument("--external-media-gateway", action="store_true")
'''
new='''    parser.add_argument(
        "--media-gateway-url",
        help="Loopback URL for the Media Gateway. Defaults to a free port when Studio manages the gateway.",
    )
    parser.add_argument(
        "--external-media-gateway",
        action="store_true",
        help="Connect to an already running gateway at --media-gateway-url instead of starting one.",
    )
'''
assert old in s; s=s.replace(old,new); p.write_text(s)
EOF
grep -n "external_media_gateway\|parse_args" packages/f8studio_server/f8studio_server/__main__.py

OUT
17:def _parse_args() -> argparse.Namespace:
50:    return parser.parse_args()
54:    args = _parse_args()
94:                    manage_process=not args.external_media_gateway,
Bash Read end of arg parser
IN
cd /home/sxs/SS/Feel8/f8studio; sed -n 44,52p packages/f8studio_server/f8studio_server/__main__.py

OUT
    parser.add_argument("--turn-credential")
    parser.add_argument(
        "--force-turn",
        action="store_true",
        help="Require browser media to use TURN relay candidates.",
    )
    return parser.parse_args()
Edit __main__.py
Added 3 lines
Bash Rerun gateway and server tests
IN
cd /home/sxs/SS/Feel8/f8studio; pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8media_gateway/tests packages/f8studio_server/tests 2>&1 | tail -4

OUT
packages/f8studio_server/tests/test_workspace_services.py:47: ModuleNotFoundError
=========================== short test summary info ============================
FAILED packages/f8studio_server/tests/test_workspace_services.py::test_python_script_editor_uses_injected_api_and_dynamic_bindings
1 failed, 165 passed in 27.73s
Bash Add engine feature to web-studio-test env
IN
cd /home/sxs/SS/Feel8/f8studio; sed -i 's/^web-studio-test = { features = \["python", "sdk", "web-studio", "web-studio-dev", "web-studio-test"\]/web-studio-test = { features = ["python", "sdk", "engine", "web-studio", "web-studio-dev", "web-studio-test"]/' pixi.toml; grep -n "^web-studio-test" pixi.toml; pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8studio_server/tests/test_workspace_services.py 2>&1 | tail -2; git diff --stat pixi.lock

OUT
249:web-studio-test = { features = ["python", "sdk", "engine", "web-studio", "web-studio-dev", "web-studio-test"], solve-group = "web-studio" }
........                                                                 [100%]
8 passed in 7.54s
 pixi.lock | 24 ++++++++++++++++++++++++
 1 file changed, 24 insertions(+)
The server suite is now fully green. Next P0 item: the f8pydl stream_epoch crash.


Bash Trace stream_epoch source in video transport
IN
cd /home/sxs/SS/Feel8/f8studio/packages; grep -n "stream_epoch\|streamEpoch\|VideoFramePacket(" f8pydl/f8pydl/*.py f8pysdk/f8pysdk/video_transport.py | head -30; sed -n 70,140p f8pydl/f8pydl/video_frame_source.py

OUT
f8pydl/f8pydl/video_frame_source.py:136:        return VideoFramePacket(
f8pydl/f8pydl/service_node.py:926:                            stream_epoch=str(frame.stream_epoch),
f8pydl/f8pydl/service_node.py:943:                            stream_epoch=str(frame.stream_epoch),
f8pydl/f8pydl/service_node.py:987:        stream_epoch: str = "",
f8pydl/f8pydl/service_node.py:1022:            "streamEpoch": str(stream_epoch),
f8pysdk/f8pysdk/video_transport.py:31:    stream_epoch: str = "00000000000000000000000000000000"
f8pysdk/f8pysdk/video_transport.py:91:    stream_epoch: str = "00000000000000000000000000000000",
f8pysdk/f8pysdk/video_transport.py:100:        epoch_int = UUID(hex=stream_epoch).int
f8pysdk/f8pysdk/video_transport.py:102:        raise ValueError("stream_epoch must be a 32-digit UUID hex value") from exc
f8pysdk/f8pysdk/video_transport.py:171:            stream_epoch=f"{(epoch_high << 64) | epoch_low:032x}",
f8pysdk/f8pysdk/video_transport.py:206:        self._stream_epoch = uuid4().hex
f8pysdk/f8pysdk/video_transport.py:297:            stream_epoch=self._stream_epoch,
f8pydl/f8pydl/node_registry.py:89:            "streamEpoch": string_schema(),
    def frame_bytes(self) -> int:
        return int(self.pitch) * int(self.height)

    def release(self) -> None:
        if self._released:
            return
        self.payload.release()
        self._released = True

    def __enter__(self) -> "VideoFramePacket":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def video_source_metadata() -> dict[str, str]:
    return {"payloadKind": "video_frame"}


class LatestVideoFrameSource:
    def __init__(self, *, config: VideoFrameSourceConfig) -> None:
        self._config = config
        self._zenoh_reader: ZenohLatestVideoFrameTransport | None = None
        self._zenoh_open_key = ""
        self._last_signature: tuple[str, int, int] | None = None

    def close(self) -> None:
        self._close_zenoh()
        self._last_signature = None

    def reset(self) -> None:
        self.close()

    def read_latest(
        self,
        *,
        stream_key: str,
        timeout_ms: int,
        dedupe: bool = True,
    ) -> VideoFramePacket | None:
        return self._read_zenoh_latest(stream_key=str(stream_key).strip(), timeout_ms=timeout_ms, dedupe=dedupe)

    def _read_zenoh_latest(self, *, stream_key: str, timeout_ms: int, dedupe: bool) -> VideoFramePacket | None:
        if not stream_key:
            return None
        reader = self._ensure_zenoh_reader(stream_key)
        frame = reader.wait_latest(max(0, int(timeout_ms)))
        if frame is None:
            return None
        return self._packet_from_zenoh(stream_key=stream_key, frame=frame, dedupe=dedupe)

    def _packet_from_zenoh(
        self, *, stream_key: str, frame: LatestVideoFrame, dedupe: bool
    ) -> VideoFramePacket | None:
        signature = (str(stream_key), int(frame.frame_id), int(frame.ts_ms))
        if dedupe and signature == self._last_signature:
            frame.release()
            return None
        if dedupe:
            self._last_signature = signature
        return VideoFramePacket(
            width=int(frame.width),
            height=int(frame.height),
            pitch=int(frame.pitch),
            fmt=int(frame.fmt),
Bash Read packet construction and other constructors
IN
cd /home/sxs/SS/Feel8/f8studio/packages; sed -n 136,150p f8pydl/f8pydl/video_frame_source.py; grep -rn "VideoFramePacket(\|_last_signature" f8pydl | grep -v "video_frame_source.py:1[0-3]" 

OUT
        return VideoFramePacket(
            width=int(frame.width),
            height=int(frame.height),
            pitch=int(frame.pitch),
            fmt=int(frame.fmt),
            frame_id=int(frame.frame_id),
            ts_ms=int(frame.ts_ms),
            payload=frame.payload,
        )

    def _ensure_zenoh_reader(self, key_expr: str) -> ZenohLatestVideoFrameTransport:
        key = str(key_expr or "").strip()
        if self._zenoh_reader is not None and self._zenoh_open_key == key:
            return self._zenoh_reader
        self._close_zenoh()
f8pydl/f8pydl/video_frame_source.py:160:        self._last_signature = None
Bash Carry stream_epoch through VideoFramePacket
IN
cd /home/sxs/SS/Feel8/f8studio/packages; python3 - <<'EOF'
from pathlib import Path
p=Path("f8pydl/f8pydl/video_frame_source.py"); s=p.read_text()
s=s.replace('''    frame_id: int
    ts_ms: int
    payload: memoryview
    _released: bool = False''','''    frame_id: int
    ts_ms: int
    stream_epoch: str
    payload: memoryview
    _released: bool = False''',1)
s=s.replace('''        self._last_signature: tuple[str, int, int] | None = None''','''        self._last_signature: tuple[str, str, int, int] | None = None''')
s=s.replace('''        signature = (str(stream_key), int(frame.frame_id), int(frame.ts_ms))''','''        signature = (str(stream_key), frame.stream_epoch, int(frame.frame_id), int(frame.ts_ms))''')
s=s.replace('''            ts_ms=int(frame.ts_ms),
            payload=frame.payload,''','''            ts_ms=int(frame.ts_ms),
            stream_epoch=frame.stream_epoch,
            payload=frame.payload,''')
p.write_text(s)
EOF
git diff --stat; cd ..; grep -rln "VideoFramePacket(" packages/*/tests tests 2>/dev/null

OUT
 .../f8media_protocol/f8media_protocol/client.py    | 25 ++++++++++++---
 packages/f8pydl/f8pydl/video_frame_source.py       |  6 ++--
 .../f8studio_server/f8studio_server/__main__.py    | 16 ++++++++--
 pixi.lock                                          | 24 +++++++++++++++
 pixi.toml                                          |  2 +-
 scripts/dist_ci.py                                 | 36 +++++++++++++++++-----
 6 files changed, 92 insertions(+), 17 deletions(-)
I'll add regression tests for both fixes, one for the f8pydl epoch and one for dist validation:


Bash Inspect test layout for f8pydl and dist
IN
cd /home/sxs/SS/Feel8/f8studio; ls packages/f8pydl/tests/ 2>/dev/null; grep -n "class LatestVideoFrame\b" -A15 packages/f8pysdk/f8pysdk/video_transport.py | head -25; grep -n "class .*Test\|def setUp\|self.module = \|self.root" tests/test_dist_ci.py | head -8

OUT
__pycache__
test_detection_protocol_field.py
test_detection_sorter_service_node.py
test_model_config_optflow.py
test_model_config_skeleton_protocol.py
test_model_config_tcn_wave.py
test_model_config_yowo_temporal.py
test_onnx_runtime_session.py
test_optflow_flow_pack_and_cache.py
test_optflow_runtime_shape.py
test_service_node_boundaries.py
test_service_paths.py
test_tcn_aggregation.py
test_tcn_runtime_shape.py
test_tcn_service_node.py
test_video_frame_source_transport.py
test_weights_downloader.py
test_yowo_temporal_runtime_shape.py
test_yowo_temporal_service_node.py
23:class LatestVideoFrame:
24-    width: int
25-    height: int
26-    pitch: int
27-    fmt: int
28-    frame_id: int
29-    ts_ms: int
30-    payload: memoryview
31-    stream_epoch: str = "00000000000000000000000000000000"
32-    _released: bool = field(default=False, init=False, repr=False)
33-
34-    @property
35-    def frame_bytes(self) -> int:
36-        return int(self.pitch) * int(self.height)
37-
38-    def payload_bytes(self) -> bytes:
26:class DistCiDiscoveryTest(unittest.TestCase):
27:    def setUp(self) -> None:
28:        self.module = _load_dist_ci_module()
30:        self.root = Path(self.temp_dir.name)
36:        package_dir = self.root / relative_package_dir
49:        pixi_toml_path = self.root / "pixi.toml"
64:            repo_root=self.root,
79:        pixi_toml_path = self.root / "pixi.toml"
Bash Read existing video frame source test
IN
cd /home/sxs/SS/Feel8/f8studio; cat packages/f8pydl/tests/test_video_frame_source_transport.py

OUT
import os
import sys


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for path in (PKG_PYDL, PKG_SDK):
    if path not in sys.path:
        sys.path.insert(0, path)


from f8pydl.video_frame_source import video_source_metadata  # noqa: E402


def test_video_source_metadata_reports_typed_frame_stream() -> None:
    assert video_source_metadata() == {"payloadKind": "video_frame"}
Bash Add f8pydl stream epoch regression tests
IN
cd /home/sxs/SS/Feel8/f8studio; cat >> packages/f8pydl/tests/test_video_frame_source_transport.py <<'EOF'


def _latest_frame(*, stream_epoch: str, frame_id: int = 7, ts_ms: int = 1000) -> "LatestVideoFrame":
    return LatestVideoFrame(
        width=2,
        height=1,
        pitch=8,
        fmt=0,
        frame_id=frame_id,
        ts_ms=ts_ms,
        payload=memoryview(bytearray(8)),
        stream_epoch=stream_epoch,
    )


def test_packet_carries_stream_epoch_from_transport() -> None:
    source = LatestVideoFrameSource(config=VideoFrameSourceConfig())
    packet = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    assert packet is not None
    assert packet.stream_epoch == "a" * 32


def test_dedupe_does_not_drop_same_frame_id_from_restarted_stream() -> None:
    source = LatestVideoFrameSource(config=VideoFrameSourceConfig())
    first = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    repeat = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="a" * 32), dedupe=True)
    restarted = source._packet_from_zenoh(stream_key="k", frame=_latest_frame(stream_epoch="b" * 32), dedupe=True)
    assert first is not None
    assert repeat is None
    assert restarted is not None
EOF
sed -i 's/^from f8pydl.video_frame_source import video_source_metadata  # noqa: E402$/from f8pydl.video_frame_source import LatestVideoFrameSource, VideoFrameSourceConfig, video_source_metadata  # noqa: E402\nfrom f8pysdk.video_transport import LatestVideoFrame  # noqa: E402/' packages/f8pydl/tests/test_video_frame_source_transport.py
sed -i 's/-> "LatestVideoFrame":/-> LatestVideoFrame:/' packages/f8pydl/tests/test_video_frame_source_transport.py
grep -n "class VideoFrameSourceConfig" -A6 packages/f8pydl/f8pydl/video_frame_source.py

OUT
33:class VideoFrameSourceConfig:
34-    config_path: str | None = None
35-    connect: tuple[str, ...] = ()
36-    listen: tuple[str, ...] = ()
37-    shm_pool_bytes: int = 256 * 1024 * 1024
38-
39-    @classmethod
Now the dist validation test:


Bash Add dist tests and rerun tests plus typecheck
IN
cd /home/sxs/SS/Feel8/f8studio; python3 - <<'EOF'
from pathlib import Path
p=Path("tests/test_dist_ci.py"); s=p.read_text()
anchor="    def test_copy_dist_config_copies_service_discovery_policy(self) -> None:"
new='''    def test_rewrite_dist_service_entries_maps_web_studio_runtime_to_dist_runtime(self) -> None:
        engine_service_path = self.root / "services" / "f8" / "engine" / "service.yml"
        engine_service_path.parent.mkdir(parents=True, exist_ok=True)
        engine_service_path.write_text(
            'launch:\\n  command: pixi\\n  args: ["run", "-e", "web-studio-runtime", "f8pyengine"]\\n',
            encoding="utf-8",
        )

        self.module._rewrite_dist_service_entries(self.root / "services")

        self.assertIn('"run", "-e", "studio-runtime", "f8pyengine"', engine_service_path.read_text(encoding="utf-8"))

    def test_validate_dist_service_environments_rejects_unshipped_environment(self) -> None:
        service_path = self.root / "services" / "f8" / "tool" / "service.yml"
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(
            'launch:\\n  command: pixi\\n  args: ["run", "-e", "web-studio", "f8tool"]\\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "'web-studio'"):
            self.module._validate_dist_service_environments(self.root / "services", ["studio-runtime", "onnx"])

'''
assert anchor in s; s=s.replace(anchor,new+anchor,1); p.write_text(s)
EOF
pixi run pytest -q -p no:cacheprovider tests/test_dist_ci.py packages/f8pydl/tests/test_video_frame_source_transport.py 2>&1 | tail -3; pixi run typecheck 2>&1 | tail -2

OUT
✨ Pixi task (pytest in default): python -m pytest -q -p no:cacheprovider tests/test_dist_ci.py packages/f8pydl/tests/test_video_frame_source_transport.py
....................................                                     [100%]
36 passed in 0.45s
✨ Pixi task (typecheck in default): basedpyright -p pyrightconfig.json
0 errors, 0 warnings, 0 notes
Bash Read port defaults and imports
IN
cd /home/sxs/SS/Feel8/f8studio/packages/f8studio_server/f8studio_server; sed -n 1,25p api_client.py; grep -n "8260\|8210\|^from\|^import" cli.py mcp_server.py __main__.py

OUT
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import quote

import httpx

from f8pysdk.specs import F8JsonValue


class StudioApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class StudioApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8260", *, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> StudioApiClient:
__main__.py:1:from __future__ import annotations
__main__.py:3:import argparse
__main__.py:4:import os
__main__.py:5:from ipaddress import ip_address
__main__.py:6:from pathlib import Path
__main__.py:8:import uvicorn
__main__.py:10:from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig
__main__.py:12:from .app import DEFAULT_ALLOWED_HOSTS, create_app
__main__.py:13:from .models import BrowserIceServer, BrowserRtcConfiguration
__main__.py:14:from .server_instance import StudioServerAlreadyRunningError, single_server_instance
__main__.py:26:    parser.add_argument("--port", default=8210, type=int)
cli.py:1:from __future__ import annotations
cli.py:3:import argparse
cli.py:4:import json
cli.py:5:import sys
cli.py:6:import time
cli.py:7:from pathlib import Path
cli.py:8:from typing import cast
cli.py:9:from uuid import uuid4
cli.py:11:from .api_client import StudioApiClient, StudioApiError
cli.py:75:    parser.add_argument("--url", default="http://127.0.0.1:8260", help="Studio Server base URL")
mcp_server.py:1:from __future__ import annotations
mcp_server.py:3:import argparse
mcp_server.py:4:from collections.abc import Callable
mcp_server.py:5:from typing import Any
mcp_server.py:6:from uuid import uuid4
mcp_server.py:8:from .api_client import StudioApiClient
mcp_server.py:127:    parser.add_argument("--studio-url", default="http://127.0.0.1:8260")
Bash Unify Studio default port constant
IN
cd /home/sxs/SS/Feel8/f8studio/packages/f8studio_server/f8studio_server; cat > defaults.py <<'EOF'
from __future__ import annotations

# Single source for the Studio server's default listen address. The server,
# the CLI, the MCP bridge and the HTTP client must agree on it.
DEFAULT_STUDIO_PORT = 8210
DEFAULT_STUDIO_URL = f"http://127.0.0.1:{DEFAULT_STUDIO_PORT}"
EOF
python3 - <<'EOF'
from pathlib import Path
def sub(path, old, new, imp_after, imp):
    p=Path(path); s=p.read_text(); assert old in s, (path, old); s=s.replace(old,new)
    assert imp_after in s, (path, imp_after); s=s.replace(imp_after, imp_after+imp,1); p.write_text(s)
sub("api_client.py",'base_url: str = "http://127.0.0.1:8260"','base_url: str = DEFAULT_STUDIO_URL',"from f8pysdk.specs import F8JsonValue\n","\nfrom .defaults import DEFAULT_STUDIO_URL\n")
sub("cli.py",'default="http://127.0.0.1:8260"','default=DEFAULT_STUDIO_URL',"from .api_client import StudioApiClient, StudioApiError\n","from .defaults import DEFAULT_STUDIO_URL\n")
sub("mcp_server.py",'default="http://127.0.0.1:8260"','default=DEFAULT_STUDIO_URL',"from .api_client import StudioApiClient\n","from .defaults import DEFAULT_STUDIO_URL\n")
sub("__main__.py",'parser.add_argument("--port", default=8210, type=int)','parser.add_argument("--port", default=DEFAULT_STUDIO_PORT, type=int)',"from .app import DEFAULT_ALLOWED_HOSTS, create_app\n","from .defaults import DEFAULT_STUDIO_PORT\n")
EOF
cd /home/sxs/SS/Feel8/f8studio; git grep -n "8260" -- ':!docs' ':!site' ':!*.lock'

OUT
Binary file packages/f8implayer/font/wqy-microhei.ttc matches
packages/f8studio_server/tests/test_app.py:836:        async with httpx.AsyncClient(transport=transport, base_url="http://vpn.test:8260") as client:
packages/f8studio_server/tests/test_app.py:840:                headers={"origin": "http://vpn.test:8260"},
Bash Read events, presentation, monitors modules
IN
cd /home/sxs/SS/Feel8/f8studio/packages/f8studio_server/f8studio_server; cat events.py; cat studio_runtime/presentation.py; cat monitors.py

OUT
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import msgspec

from f8pysdk.specs import F8JsonValue


class EventEnvelope(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    event_id: str
    server_epoch: str
    sequence: int
    type: str
    scope: str
    timestamp: str
    payload: F8JsonValue


@dataclass(frozen=True)
class OpenEventStream:
    subscription_id: str
    queue: asyncio.Queue[EventEnvelope]
    replay: tuple[EventEnvelope, ...]
    snapshot_required: bool
    current_sequence: int
    oldest_sequence: int


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EventJournal:
    def __init__(
        self,
        *,
        server_epoch: str,
        retention: int = 2048,
        log_retention: int = 1000,
        subscriber_queue_size: int = 256,
    ) -> None:
        if retention < 1:
            raise ValueError("event retention must be positive")
        if log_retention < 1:
            raise ValueError("log retention must be positive")
        if subscriber_queue_size < 1:
            raise ValueError("subscriber queue size must be positive")
        self._server_epoch = server_epoch
        self._events: deque[EventEnvelope] = deque(maxlen=retention)
        self._logs: deque[EventEnvelope] = deque(maxlen=log_retention)
        self._subscriber_queue_size = subscriber_queue_size
        self._subscribers: dict[str, asyncio.Queue[EventEnvelope]] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()

    @property
    def server_epoch(self) -> str:
        return self._server_epoch

    async def publish(
        self,
        *,
        event_type: str,
        scope: str,
        payload: F8JsonValue,
        reliable: bool = True,
    ) -> EventEnvelope:
        async with self._lock:
            self._sequence += 1
            event = EventEnvelope(
                event_id=uuid4().hex,
                server_epoch=self._server_epoch,
                sequence=self._sequence,
                type=event_type,
                scope=scope,
                timestamp=_timestamp(),
                payload=payload,
            )
            if reliable:
                self._events.append(event)
            if (
                event_type == "service.log"
                or event_type.startswith("deploy.")
                or event_type.startswith("service.process_")
                or event_type in {"runtime.error", "media.error", "server.error"}
            ):
                self._logs.append(event)
            for subscription_id, queue in tuple(self._subscribers.items()):
                if queue.full():
                    if not reliable:
                        continue
                    self._replace_with_resync_event(subscription_id, queue)
                    continue
                queue.put_nowait(event)
            return event

    async def recent_logs(self, *, limit: int = 500, before_sequence: int | None = None) -> tuple[EventEnvelope, ...]:
        if limit < 1:
            raise ValueError("log limit must be positive")
        if before_sequence is not None and before_sequence < 1:
            raise ValueError("before_sequence must be positive")
        async with self._lock:
            logs = tuple(self._logs)
            if before_sequence is not None:
                logs = tuple(event for event in logs if event.sequence < before_sequence)
            return logs[-limit:]

    async def open_stream(
        self,
        *,
        client_epoch: str | None,
        after_sequence: int | None,
    ) -> OpenEventStream:
        async with self._lock:
            oldest_sequence = self._events[0].sequence if self._events else self._sequence + 1
            valid_cursor = (
                client_epoch == self._server_epoch
                and after_sequence is not None
                and after_sequence >= oldest_sequence - 1
                and after_sequence <= self._sequence
            )
            replay = (
                tuple(event for event in self._events if event.sequence > after_sequence)
                if valid_cursor and after_sequence is not None
                else ()
            )
            subscription_id = uuid4().hex
            queue: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=self._subscriber_queue_size)
            self._subscribers[subscription_id] = queue
            return OpenEventStream(
                subscription_id=subscription_id,
                queue=queue,
                replay=replay,
                snapshot_required=not valid_cursor,
                current_sequence=self._sequence,
                oldest_sequence=oldest_sequence,
            )

    async def close_stream(self, subscription_id: str) -> None:
        async with self._lock:
            self._subscribers.pop(subscription_id, None)

    def _replace_with_resync_event(
        self,
        subscription_id: str,
        queue: asyncio.Queue[EventEnvelope],
    ) -> None:
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(
            EventEnvelope(
                event_id=uuid4().hex,
                server_epoch=self._server_epoch,
                sequence=self._sequence,
                type="stream.resync_required",
                scope="server",
                timestamp=_timestamp(),
                payload={
                    "reason": "subscriber_queue_overflow",
                    "subscriptionId": subscription_id,
                },
            )
        )


__all__ = ["EventEnvelope", "EventJournal", "OpenEventStream"]
from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, cast

import msgspec

from f8pysdk.specs import F8JsonValue

from ..events import EventJournal
from ..models import PresentationCommand


logger = logging.getLogger(__name__)


class PresentationOutlet(Protocol):
    def emit(
        self,
        node_id: str,
        command: str,
        payload: dict[str, Any],
        *,
        ts_ms: int | None = None,
    ) -> None: ...


class EventPresentationOutlet:
    def __init__(self, events: EventJournal) -> None:
        self._events = events
        self._tasks: set[asyncio.Task[object]] = set()
        self._latest: dict[tuple[str, str], PresentationCommand] = {}
        self._closed = False

    def emit(
        self,
        node_id: str,
        command: str,
        payload: dict[str, Any],
        *,
        ts_ms: int | None = None,
    ) -> None:
        if self._closed:
            return
        normalized_payload = cast(F8JsonValue, msgspec.to_builtins(payload, str_keys=True))
        if not isinstance(normalized_payload, dict):
            raise TypeError("presentation payload must encode to an object")
        if command.endswith(".detach"):
            for key in tuple(self._latest):
                if key[0] == node_id:
                    del self._latest[key]
        else:
            self._latest[(node_id, command)] = PresentationCommand(
                node_id=node_id,
                command=command,
                payload=normalized_payload,
                ts_ms=ts_ms,
            )
        task = asyncio.create_task(
            self._events.publish(
                event_type="presentation.command",
                scope=f"node:{node_id}",
                payload={
                    "nodeId": node_id,
                    "command": command,
                    "payload": normalized_payload,
                    "tsMs": ts_ms,
                },
                reliable=False,
            ),
            name=f"presentation:{node_id}:{command}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def snapshot(self) -> tuple[PresentationCommand, ...]:
        return tuple(
            sorted(
                self._latest.values(),
                key=lambda item: (
                    item.ts_ms if item.ts_ms is not None else -1,
                    item.node_id,
                    item.command,
                ),
            )
        )

    def _task_done(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("presentation event publication failed", exc_info=error)

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._latest.clear()


__all__ = ["EventPresentationOutlet", "PresentationOutlet"]
from __future__ import annotations

import asyncio
import logging

import msgspec

from f8pysdk.codec import decode_as
from f8pysdk.specs import F8MonitorSnapshot

from .events import EventJournal


logger = logging.getLogger(__name__)


class MonitorEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    value: F8MonitorSnapshot
    ts: int


class RuntimeMonitorStore:
    def __init__(self, events: EventJournal, *, studio_service_id: str | None = None) -> None:
        self._events = events
        self._studio_service_id = studio_service_id
        self._latest: dict[tuple[str, str], F8MonitorSnapshot] = {}
        self._lock = asyncio.Lock()
        self._reported_decode_errors: set[str] = set()

    async def ingest(self, key: str, payload: bytes) -> None:
        try:
            envelope = decode_as(payload, MonitorEnvelope)
        except ValueError as exc:
            signature = f"{type(exc).__name__}:{exc}"
            if signature not in self._reported_decode_errors:
                self._reported_decode_errors.add(signature)
                logger.warning("invalid runtime monitor payload key=%s", key, exc_info=exc)
            return
        snapshot = envelope.value
        service_id = str(snapshot.serviceId)
        if service_id == "studio" and self._studio_service_id is not None:
            return
        if service_id.startswith("studio_"):
            if service_id != self._studio_service_id:
                return
            snapshot = msgspec.structs.replace(
                snapshot,
                serviceId="studio",
                nodeId="studio" if str(snapshot.nodeId) == service_id else snapshot.nodeId,
            )
        identity = (str(snapshot.serviceId), str(snapshot.nodeId))
        async with self._lock:
            self._latest[identity] = snapshot
        await self._events.publish(
            event_type="runtime.monitor",
            scope=f"service:{snapshot.serviceId}",
            payload=msgspec.to_builtins(snapshot, str_keys=True),
            reliable=False,
        )

    async def snapshot(self) -> tuple[F8MonitorSnapshot, ...]:
        async with self._lock:
            return tuple(self._latest[key] for key in sorted(self._latest))


__all__ = ["MonitorEnvelope", "RuntimeMonitorStore"]
Bash Read WS handler and all publish sites
IN
cd /home/sxs/SS/Feel8/f8studio/packages/f8studio_server/f8studio_server; sed -n 845,910p app.py; grep -n "journal.publish\|publish(" *.py agents/*.py studio_runtime/*.py | head -40

OUT
                service_id,
                call=payload.call,
                params=payload.params,
            ),
        )

    @app.websocket("/api/events")
    async def events(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket.headers.get("origin"), resolved_allowed_hosts):
            await websocket.close(code=1008, reason="websocket origin is not allowed")
            return
        after_text = websocket.query_params.get("after")
        try:
            after_sequence = None if after_text is None else int(after_text)
        except ValueError:
            await websocket.close(code=1008, reason="after must be an integer")
            return
        stream = await studio.events.open_stream(
            client_epoch=websocket.query_params.get("epoch"),
            after_sequence=after_sequence,
        )
        await websocket.accept()
        try:
            if stream.snapshot_required:
                projects = await asyncio.to_thread(studio.projects.list)
                await websocket.send_json(
                    {
                        "eventId": "snapshot",
                        "serverEpoch": studio.server_epoch,
                        "sequence": stream.current_sequence,
                        "type": "stream.snapshot",
                        "scope": "server",
                        "payload": {
                            "projects": _json_value(projects),
                            "oldestSequence": stream.oldest_sequence,
                        },
                    }
                )
            for event in stream.replay:
                await websocket.send_json(_json_value(event))
            while True:
                event = await stream.queue.get()
                await websocket.send_json(_json_value(event))
                if event.type == "stream.resync_required":
                    await websocket.close(code=1013, reason="event stream resynchronization required")
                    return
        except WebSocketDisconnect:
            return
        finally:
            await studio.events.close_stream(stream.subscription_id)

    if has_web_assets:
        assets_path = resolved_web_dist / "assets"
        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="web-assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def web_app(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            requested_path = (resolved_web_dist / path).resolve()
            if path and requested_path.is_relative_to(resolved_web_dist) and requested_path.is_file():
                return FileResponse(requested_path)
            return FileResponse(index_path)

    return app
automation_tools.py:123:        await self._events.publish(
app.py:227:        await studio.events.publish(
app.py:236:        await studio.events.publish(
app.py:246:        await studio.events.publish(
app.py:313:        await studio.events.publish(event_type="asset.created", scope=f"asset:{record.asset_id}", payload=_json_value(record))
app.py:320:        await studio.events.publish(event_type="asset.created", scope=f"asset:{record.asset_id}", payload=_json_value(record))
app.py:331:        await studio.events.publish(event_type="asset.updated", scope=f"asset:{record.asset_id}", payload=_json_value(record))
app.py:337:        await studio.events.publish(event_type="asset.deleted", scope=f"asset:{asset_id}", payload={"assetId": asset_id})
app.py:416:        await studio.events.publish(
app.py:440:        await studio.events.publish(
app.py:456:        await studio.events.publish(
app.py:467:        await studio.events.publish(
app.py:496:        await studio.events.publish(
app.py:610:            await studio.events.publish(
jobs.py:258:        await self._events.publish(
studio_runtime/presentation.py:61:            self._events.publish(
monitors.py:54:        await self._events.publish(
events.py:65:    async def publish(
processes.py:53:        await self._events.publish(
processes.py:63:        await self._events.publish(
processes.py:93:            self._events.publish(
agents/service.py:323:        await self._publish(updated)
agents/service.py:337:        await self._publish(updated)
agents/service.py:346:        await self._events.publish(
agents/service.py:391:        await self._publish(started)
agents/service.py:438:        await self._publish(updated)
agents/service.py:480:        await self._publish(cancelled)
agents/service.py:544:            await self._publish(completed)
agents/service.py:1241:        await self._publish(waiting)
agents/service.py:1289:            await self._publish(updated)
agents/service.py:1329:            await self._publish(updated)
agents/service.py:1340:            await self._publish(updated)
agents/service.py:1371:        await self._publish(stopped)
agents/service.py:1373:    async def _publish(self, record: AgentSessionRecord) -> None:
agents/service.py:1375:        await self._events.publish(
The event, presentation and monitor code is clear now. Next I need how node runtime state is read, and how the web presentation store reduces commands:


Bash Read runtime gateway state cache and read route
IN
cd /home/sxs/SS/Feel8/f8studio/packages; sed -n 180,330p f8studio_server/f8studio_server/runtime.py; sed -n 820,845p f8studio_server/f8studio_server/app.py

OUT
@dataclass(frozen=True)
class RuntimeConfig:
    bus_backend: BusBackend = "zenoh"
    client_service_id: str = "webstudio"
    zenoh_config_path: str | None = None
    zenoh_connect: tuple[str, ...] = ()
    zenoh_listen: tuple[str, ...] = ()
    zenoh_shm_pool_bytes: int = 256 * 1024 * 1024
    endpoint_ready_timeout_s: float = 4.0
    request_timeout_s: float = 1.0
    request_attempts: int = 3
    deploy_timeout_s: float = 15.0


def _error_message(error: F8CommandError | None | msgspec.UnsetType) -> str:
    if error is None or isinstance(error, msgspec.UnsetType):
        return ""
    return str(error.message)


@dataclass
class ZenohRuntimeGateway:
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    _transport: RuntimeTransport | None = field(default=None, init=False, repr=False)
    _monitor_subscription: RuntimeSubscription | None = field(default=None, init=False, repr=False)
    _state_subscription: RuntimeSubscription | None = field(default=None, init=False, repr=False)
    _state_values: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def _build_transport(self) -> RuntimeTransport:
        if self.config.bus_backend == "mem":
            from f8pysdk.testing import InMemoryCluster, InMemoryTransport

            return InMemoryTransport(cluster=InMemoryCluster())
        if self.config.bus_backend != "zenoh":
            raise ValueError("runtime gateway supports only zenoh or mem bus backends")
        return ZenohTransport(
            ZenohTransportConfig(
                service_id=self.config.client_service_id,
                config_path=self.config.zenoh_config_path,
                connect=self.config.zenoh_connect,
                listen=self.config.zenoh_listen,
                shm_pool_bytes=self.config.zenoh_shm_pool_bytes,
            )
        )

    async def _connected_transport(self) -> RuntimeTransport:
        transport = self._transport
        if transport is not None:
            return transport
        async with self._connect_lock:
            transport = self._transport
            if transport is not None:
                return transport
            transport = self._build_transport()
            await transport.connect()
            self._transport = transport
            return transport

    async def close(self) -> None:
        async with self._connect_lock:
            transport = self._transport
            self._transport = None
            monitor_subscription = self._monitor_subscription
            self._monitor_subscription = None
            state_subscription = self._state_subscription
            self._state_subscription = None
            self._state_values.clear()
        if monitor_subscription is not None:
            await monitor_subscription.unsubscribe()
        if state_subscription is not None:
            await state_subscription.unsubscribe()
        if transport is not None:
            await transport.close()

    async def start_monitoring(self, callback: RuntimeMonitorCallback) -> None:
        transport = await self._connected_transport()
        if self._monitor_subscription is None:
            subscription = await transport.subscribe(
                "f8/svc/*/nodes/*/data/monitor",
                cb=callback,
            )
            self._monitor_subscription = cast(RuntimeSubscription, subscription)
        if self._state_subscription is None:
            state_subscription = await transport.retained_watch(
                "f8/svc/*/state/nodes/*/state/**",
                cb=self._ingest_state,
                with_initial=True,
            )
            self._state_subscription = cast(RuntimeSubscription, state_subscription)

    async def _ingest_state(self, key: str, payload: bytes) -> None:
        self._state_values[key] = bytes(payload)

    async def _request(self, key: str, payload: bytes, *, timeout_s: float | None = None) -> bytes:
        transport = await self._connected_transport()
        raw = await transport.request(
            key,
            payload,
            timeout=self.config.request_timeout_s if timeout_s is None else timeout_s,
            raise_on_error=True,
        )
        if not raw:
            raise RuntimeError(f"empty runtime response from {key}")
        return raw

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        service_id = ensure_token(service_id, label="service_id")
        request = F8StatusRequest(
            reqId=new_id(),
            args=F8EmptyArgs(),
            meta={"actor": "webstudio", "source": "api"},
        )
        response = decode_as(
            await self._request(svc_endpoint_key(service_id, "status"), encode_obj(request)),
            F8StatusReply,
        )
        if not response.ok or response.result is None or isinstance(response.result, msgspec.UnsetType):
            raise RuntimeError(_error_message(response.error) or f"status rejected by {service_id}")
        result = response.result
        return ServiceRuntimeStatus(
            service_id=str(result.serviceId),
            service_class=str(result.serviceClass),
            runtime_instance_id=str(result.runtimeInstanceId),
            active=bool(result.active),
            rungraph_graph_id="" if isinstance(result.rungraphGraphId, msgspec.UnsetType) else str(result.rungraphGraphId),
            rungraph_revision="" if isinstance(result.rungraphRevision, msgspec.UnsetType) else str(result.rungraphRevision),
            rungraph_fingerprint=(
                "" if isinstance(result.rungraphFingerprint, msgspec.UnsetType) else str(result.rungraphFingerprint)
            ),
        )

    async def _wait_until_ready(self, service_id: str) -> ServiceRuntimeStatus:
        deadline = asyncio.get_running_loop().time() + self.config.endpoint_ready_timeout_s
        last_error = ""
        while True:
            try:
                return await self.status(service_id)
            except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"service endpoint not ready: {service_id}: {last_error}")
            await asyncio.sleep(0.1)

    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult:
            studio.runtime.set_state(
                service_id,
                node_id=payload.node_id,
                field=payload.field,
                value=payload.value,
            ),
        )

    @app.post("/api/runtime/services/{service_id}/nodes/{node_id}/state:read")
    async def read_node_state(service_id: str, node_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, RuntimeStateReadRequest)
        normalized_fields = tuple(dict.fromkeys(field.strip() for field in payload.fields if field.strip()))
        if len(normalized_fields) > 128:
            raise HTTPException(status_code=422, detail="at most 128 state fields may be read at once")
        fields = await asyncio.gather(
            *(studio.runtime.read_state(service_id, node_id=node_id, field=field) for field in normalized_fields)
        )
        return _json_value(RuntimeNodeState(service_id=service_id, node_id=node_id, fields=tuple(fields)))

    @app.post("/api/runtime/services/{service_id}/commands")
    async def invoke_service_command(service_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ServiceCommandRequest)
        return await runtime_result(
            f"command:{service_id}",
            studio.runtime.invoke_command(
                service_id,
Bash Read web presentation store and state hook
IN
cd /home/sxs/SS/Feel8/f8studio/packages/f8studio_web/src; cat presentation/PresentationStore.tsx; cat graph/useRuntimeNodeState.ts

OUT
import { createContext, type ReactNode, useCallback, useContext, useEffect, useState, useSyncExternalStore } from 'react';

import { fetchPresentationSnapshot } from '../api/client';
import type { JsonValue, PresentationCommand } from '../api/contracts';
import { extensionRendererForCommand, extensionRendererById } from '../extensions/registry';

export type PresentationRenderer = 'text' | 'wave' | 'track' | 'video' | 'audio' | 'three_d' | (string & {});

export interface PresentationOutput {
  readonly nodeId: string;
  readonly renderer: PresentationRenderer;
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly updatedAt: number;
}

type Listener = () => void;

export function parsePresentationCommand(value: unknown): PresentationCommand | null {
  if (typeof value !== 'object' || value === null) return null;
  const envelope = value as Record<string, unknown>;
  if (envelope.type !== 'presentation.command' || typeof envelope.payload !== 'object' || envelope.payload === null) return null;
  const command = envelope.payload as Record<string, unknown>;
  if (typeof command.nodeId !== 'string' || typeof command.command !== 'string' || typeof command.payload !== 'object' || command.payload === null || Array.isArray(command.payload)) return null;
  return {
    nodeId: command.nodeId,
    command: command.command,
    payload: command.payload as Readonly<Record<string, JsonValue>>,
    tsMs: typeof command.tsMs === 'number' ? command.tsMs : null,
  };
}

function rendererFor(command: string): PresentationRenderer | null {
  if (command.startsWith('viz.text.')) return 'text';
  if (command.startsWith('viz.wave.')) return 'wave';
  if (command.startsWith('viz.track.')) return 'track';
  if (command.startsWith('viz.video.')) return 'video';
  if (command.startsWith('viz.audio.')) return 'audio';
  if (command.startsWith('viz.three_d.')) return 'three_d';
  return extensionRendererForCommand(command)?.id ?? null;
}

export class PresentationStore {
  private readonly outputs = new Map<string, PresentationOutput>();
  private outputsSnapshot: ReadonlyMap<string, PresentationOutput> = new Map();
  private readonly outputListeners = new Set<Listener>();
  private readonly nodeListeners = new Map<string, Set<Listener>>();
  private readonly connectionListeners = new Set<Listener>();
  private socket: WebSocket | null = null;
  private retryTimer: number | null = null;
  private retryCount = 0;
  private started = false;
  private connected = false;
  private snapshotController: AbortController | null = null;

  readonly getOutputsSnapshot = (): ReadonlyMap<string, PresentationOutput> => this.outputsSnapshot;
  readonly getConnectionSnapshot = (): boolean => this.connected;

  getOutputSnapshot(nodeId: string): PresentationOutput | null {
    return this.outputs.get(nodeId) ?? null;
  }

  subscribeOutputs = (listener: Listener): (() => void) => {
    this.outputListeners.add(listener);
    return () => this.outputListeners.delete(listener);
  };

  subscribeConnection = (listener: Listener): (() => void) => {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  };

  subscribeNode(nodeId: string, listener: Listener): () => void {
    const existing = this.nodeListeners.get(nodeId);
    if (existing === undefined) this.nodeListeners.set(nodeId, new Set([listener]));
    else existing.add(listener);
    return () => {
      const listeners = this.nodeListeners.get(nodeId);
      listeners?.delete(listener);
      if (listeners?.size === 0) this.nodeListeners.delete(nodeId);
    };
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  stop(): void {
    this.started = false;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.snapshotController?.abort();
    this.snapshotController = null;
    const socket = this.socket;
    this.socket = null;
    socket?.close();
    this.setConnected(false);
  }

  applyCommand(command: PresentationCommand): void {
    const renderer = rendererFor(command.command);
    if (renderer === null) return;
    if (command.command.endsWith('.detach')) {
      if (!this.outputs.delete(command.nodeId)) return;
      this.publishChanges([command.nodeId]);
      return;
    }

    const prior = this.outputs.get(command.nodeId);
    const updatedAt = command.tsMs ?? Date.now();
    if (prior !== undefined && prior.updatedAt > updatedAt) return;
    const priorPayload = prior?.renderer === renderer ? prior.payload : {};
    const extensionReducer = extensionRendererById(renderer)?.reduce;
    const payload = extensionReducer !== undefined
      ? extensionReducer(command.command, priorPayload, command.payload)
      : renderer === 'three_d' && command.command === 'viz.three_d.world_up'
        ? { ...priorPayload, ...command.payload }
        : command.payload;
    this.outputs.delete(command.nodeId);
    this.outputs.set(command.nodeId, { nodeId: command.nodeId, renderer, payload, updatedAt });

    const changedNodeIds = [command.nodeId];
    while (this.outputs.size > 32) {
      const oldestNodeId = this.outputs.keys().next().value;
      if (typeof oldestNodeId !== 'string') break;
      this.outputs.delete(oldestNodeId);
      changedNodeIds.push(oldestNodeId);
    }
    this.publishChanges(changedNodeIds);
  }

  private publishChanges(nodeIds: readonly string[]): void {
    this.outputsSnapshot = new Map(this.outputs);
    for (const listener of this.outputListeners) listener();
    for (const nodeId of new Set(nodeIds)) {
      for (const listener of this.nodeListeners.get(nodeId) ?? []) listener();
    }
  }

  private setConnected(connected: boolean): void {
    if (this.connected === connected) return;
    this.connected = connected;
    for (const listener of this.connectionListeners) listener();
  }

  private connect(): void {
    if (!this.started) return;
    let socket: WebSocket;
    try {
      socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/events`);
    } catch (error: unknown) {
      console.error('Failed to open presentation event stream', error);
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.retryCount = 0;
      this.setConnected(true);
      this.snapshotController?.abort();
      const controller = new AbortController();
      this.snapshotController = controller;
      void fetchPresentationSnapshot(controller.signal).then(
        (commands) => {
          if (this.socket !== socket || controller.signal.aborted) return;
          for (const command of commands) this.applyCommand(command);
        },
        (reason: unknown) => {
          if (!controller.signal.aborted) console.error('Failed to load presentation snapshot', reason);
        },
      );
    };
    socket.onmessage = (event) => {
      if (this.socket !== socket) return;
      let decoded: unknown;
      try {
        decoded = JSON.parse(String(event.data));
      } catch (error: unknown) {
        console.error('Invalid presentation event JSON', error);
        return;
      }
      const command = parsePresentationCommand(decoded);
      if (command !== null) this.applyCommand(command);
    };
    socket.onerror = () => socket.close();
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.snapshotController?.abort();
      this.snapshotController = null;
      this.setConnected(false);
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (!this.started || this.retryTimer !== null) return;
    const delay = Math.min(5_000, 300 * 2 ** this.retryCount);
    this.retryCount += 1;
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }
}

const PresentationStoreContext = createContext<PresentationStore | null>(null);

export function PresentationProvider({ children }: { readonly children: ReactNode }) {
  const [store] = useState(() => new PresentationStore());
  useEffect(() => {
    store.start();
    return () => store.stop();
  }, [store]);
  return <PresentationStoreContext.Provider value={store}>{children}</PresentationStoreContext.Provider>;
}

function usePresentationStore(): PresentationStore {
  const store = useContext(PresentationStoreContext);
  if (store === null) throw new Error('Presentation hooks require PresentationProvider');
  return store;
}

export function usePresentationOutput(nodeId: string): PresentationOutput | null {
  const store = usePresentationStore();
  const subscribe = useCallback((listener: Listener) => store.subscribeNode(nodeId, listener), [nodeId, store]);
  const getSnapshot = useCallback(() => store.getOutputSnapshot(nodeId), [nodeId, store]);
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function usePresentationOutputs(): ReadonlyMap<string, PresentationOutput> {
  const store = usePresentationStore();
  return useSyncExternalStore(store.subscribeOutputs, store.getOutputsSnapshot, store.getOutputsSnapshot);
}

export function usePresentationConnected(): boolean {
  const store = usePresentationStore();
  return useSyncExternalStore(store.subscribeConnection, store.getConnectionSnapshot, store.getConnectionSnapshot);
}
import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchRuntimeNodeState } from '../api/client';
import type { GraphNode, RuntimeStateField } from '../api/contracts';

export function useRuntimeNodeState(node: GraphNode, names: readonly string[]): Readonly<Record<string, RuntimeStateField>> {
  const [values, setValues] = useState<Readonly<Record<string, RuntimeStateField>>>({});
  const reportedError = useRef(false);
  const key = useMemo(() => names.join('\u0000'), [names]);
  useEffect(() => {
    const controller = new AbortController();
    const fields = key === '' ? [] : key.split('\u0000');
    setValues({});
    const load = async () => {
      if (fields.length === 0) {
        setValues({});
        return;
      }
      try {
        const state = await fetchRuntimeNodeState(node.serviceId, node.nodeId, fields, controller.signal);
        if (!controller.signal.aborted) {
          setValues(Object.fromEntries(state.fields.map((field) => [field.field, field])));
          reportedError.current = false;
        }
      } catch (reason: unknown) {
        if (controller.signal.aborted) return;
        setValues({});
        if (!reportedError.current) {
          reportedError.current = true;
          console.error(`Failed to read runtime state for ${node.nodeId}`, reason);
        }
      }
    };
    void load();
    const timer = window.setInterval(() => void load(), 1500);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [node.nodeId, node.serviceId, key]);
  return values;
}
Agent:Fix agents subsystem issues
IN
You are implementing fixes in the repo /home/sxs/SS/Feel8/f8studio (branch dev). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it strictly (no getattr/setattr/hasattr on known attrs, no silent/broad excepts except at boundaries with logger.exception, explicit static typed code, use `pixi run ...` for everything; Studio server tests run in the `web-studio-test` env: `pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8studio_server/tests/...`; f8pyengine tests: `pixi run python -m pytest -q -p no:cacheprovider packages/f8pyengine/tests/...`).

CONCURRENCY RULES (important): several other agents are editing other parts of this same working tree right now. You OWN ONLY: packages/f8studio_server/f8studio_server/agents/** , packages/f8studio_server/tests/test_agents.py, test_decisions.py, test_agent_provider_settings.py (and new test files you create for agents), packages/f8pyengine/f8pyengine/operators/decision.py and its tests, packages/f8studio_server/f8studio_server/mcp_server.py. Do NOT edit other files. If you absolutely need a change in packages/f8studio_server/f8studio_server/app.py (another agent is heavily editing it), keep it minimal and confined to agent/decision route handlers, re-read the file right before each edit, and report it. Never revert or "fix" changes you see in files you don't own. Do not git commit, do not git stash/checkout/reset. Do not download or pip-install anything; do not create files at repo root. Do not kill any running processes (the user has a Studio running on ports 8210/8211). If tests outside your area fail because of other agents' in-progress work, ignore them and note it.

TASKS (from a verified review; line numbers approximate). service.py = packages/f8studio_server/f8studio_server/agents/service.py
1. BUG: `_resolve_record_approval` (~service.py:1393) only moves status to `running` when approved; on deny/expire/invalidate the session stays `waiting_for_approval` while the model run continues (agent_framework turns tool exceptions into tool error results, so `except ApprovalDeniedError` at ~547 only works for the deterministic provider). Make every approval outcome set an explicit, correct status. Decide deliberately: a denial should be reported to the model as a tool result "user denied" and the run continues in `running` state (simplest, matches the library). Make sure the UI prompt box gets re-enabled (status leaves waiting_for_approval).
2. BUG: run timeout (wait_for 300s ~505) equals approval TTL (~77); `_finish_stopped` (~1354) leaves approval pending and tool calls running. Create ONE shared helper that finalizes a record for any terminal outcome (cancel, timeout, failure, interrupted-at-startup): resolve pending approval (expired/cancelled), close open tool calls with a terminal status. Use it from cancel(), _finish_stopped, _mark_interrupted_sessions, failure path. Exclude time spent waiting for approval from the run timeout (e.g. timeout applies per model/tool step, or pause the budget while waiting) — pick the simplest correct approach and document it in a short comment.
3. RACE: single global asyncio.Lock (~254) but several read-modify-write paths bypass it (final save ~534-543, _finish_stopped ~1362, _expire_approval ~1343). Introduce per-session locks and a single `_mutate(session_id, fn) -> record` helper through which EVERY write goes; add a terminal-status guard so a completion save cannot overwrite a cancelled record (and vice versa).
4. Approvals must also check layout revision (~421-431) OR (preferred, simpler) agent patches that contain no layout operations should not pin expected_layout_revision (graph_edits.py ~106, service.py ~835). Check packages/f8studio_core/f8studio_core/graph/store.py semantics (read-only for you) to choose correctly.
5. graph_edits.build_patch adds RefreshInstalledSpecOp for EVERY refreshable node (~graph_edits.py:67-72). Only refresh nodes the patch touches.
6. Remove the raw `graph_preview_patch`/`graph_apply_patch` tools from the model tool set (_model_tools ~710-741) so the model has one edit path (propose -> validate -> approve -> apply). Keep code_write. Make the propose path the single validated path.
7. `code_write` (~820-831): run compile() and create the diff artifact inside the tracked `_tool` call, and only store the artifact as part of the tracked call (syntax errors must show up as a failed tool call).
8. `_code_target` (~564-567): remove the call made only for side effects; replace `next(...)` with an explicit lookup that raises a clear ValueError.
9. Move the deterministic keyword-routed "graph-builder-v1" provider (~506-533, 956-1134, 1420-1433, providers.py ~92) out of production code into a test fake behind the same provider interface; it must no longer be the default provider (models.py ~130, mcp_server.py ~81; the web default in AgentWorkspace.tsx ~59 is owned by another agent — just report what the web should change). If removing it would break the product flow (e.g. no provider configured), make the "no provider configured" state explicit and error clearly instead.
10. providers.py: collapse the three near-identical client-construction branches (~138-177) into a per-protocol table; centralize provider-specific rules (the `"typesafe"` checks in providers.py ~50,80 and provider_settings.py ~163-213) into one `ProviderDescriptor` dataclass registry. Remove the `api_key or "local"` hack if not needed, or make it explicit per descriptor. provider_probe.py (~249-277): split capability parsing per protocol; ~327: keep the validation reason in the error.
11. Split service.py (~1470 lines) into cohesive modules, e.g. session_store.py (CRUD + _mutate + interrupted recovery), approval_gate.py, tool_recorder.py, model_tools.py (replace the 23 closures with an explicit RunToolkit class with typed fields), run_loop.py. Keep the public AgentService API used by app.py unchanged (check app.py usage read-only).
12. Repository: stop re-serializing images on every update if feasible within scope: at minimum make list() not decode full records (store summary columns). If moving images to a separate table is feasible with a simple schema migration (repository has its own schema), do it; otherwise report.
13. Events: `_publish` emits agent.session.updated; include `updatedAt` and a monotonically increasing `version` in the payload so the web can skip redundant GETs. Also publish `agent.session.deleted` on delete if not already.
14. decision.py (f8pyengine): superseded/stale/generation-changed pending requests (~178-180, 278-290) currently produce no exec output → downstream exec flow silently stops. Emit on the `error` exec port (or add a `dropped` exec port if it fits the node spec better) with a reason. Provider-neutral error messages in app.py decision route are owned elsewhere—just report.
15. Port default: mcp_server.py now imports DEFAULT_STUDIO_URL from ..defaults (already done by lead) — keep it.
Add/adjust tests for every behavior change (deny → status running and prompt usable; timeout cleanup; cancel-vs-complete race guard; build_patch refresh scope; decision drop emits error). Run the agents/decisions test files plus `pixi run -e web-studio-test studio_python_typecheck` (basedpyright strict for studio) and make them pass.

FINAL REPORT (your only output that reaches the lead): list of files changed/created/deleted, per-task what you did (or why not), test commands + results, any follow-ups needed in files you don't own (be specific: file, what to change). Keep it under 900 words.
Agent:Fix media gateway lifecycle
IN
You are implementing fixes in the repo /home/sxs/SS/Feel8/f8studio (branch dev). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it strictly (no getattr/setattr/hasattr on known attrs, no silent/broad excepts except at real boundaries with logger.exception, explicit static typed code). Use pixi: `pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8media_gateway/tests`, type check `pixi run -e web-studio-test studio_python_typecheck`.

CONCURRENCY RULES: other agents are editing other parts of this same working tree right now. You OWN ONLY: packages/f8media_gateway/** , packages/f8media_protocol/** (note: the lead already changed f8media_protocol/client.py so a managed gateway picks a free loopback port when base_url is None — keep that), packages/f8pysdk/f8pysdk/binary_stream_transport.py, video_transport.py, audio_transport.py (and tests for these). Do NOT edit any other files (in particular not f8studio_server/app.py, not the web frontend, not other f8pysdk files). If a change is needed elsewhere, describe it in your report instead. Never revert changes you see in files you don't own. No git commit/stash/checkout/reset. Do not download or pip-install anything and do not create files at the repo root. Do not kill running processes (user has Studio running on 8210/8211). Adding a runtime dependency (e.g. numpy) requires a pyproject/pixi change: numpy is already in the pixi `vision` feature; check whether the environments that run the gateway (web-studio, studio-runtime, web-studio-test in /home/sxs/SS/Feel8/f8studio/pixi.toml) include it — if not, do NOT edit pixi.toml; report the exact change needed and implement numpy usage only if already available in those environments.

TASKS (from a verified review; line numbers approximate):
1. HIGH: `_acquire_hub` (media.py ~711; audio_media.py ~339) calls `create_frame_producer` synchronously under the manager lock; that runs `zenoh.open()` + `time.sleep` (binary_stream_transport.py ~116-134) on the event loop and stalls all live WebRTC sessions. Also each source opens its own Zenoh session with a 256MB SHM pool, and `sample()` (~695-702) opens one per HTTP call. Fix: one shared Zenoh session per gateway process (owned by the gateway service, closed on shutdown), per-source subscribers created off the event loop (asyncio.to_thread) and without holding the lock during the blocking part (use a per-source "opening" future so concurrent acquirers of the same source wait on the same open). Remove fixed settle sleeps if they're not needed, or justify them.
2. HIGH: a failed hub stays in `_hubs` (media.py ~231-236 marks `_closed`; audio_media.py ~190-195) so new sessions reuse a dead hub and fail immediately. Evict failed hubs; surface the error to the session (log with context, dedup repeated identical errors per AGENTS.md rule 6).
3. Janitor (~743-765): also reap sessions stuck in new/connecting beyond a timeout; make the janitor loop robust (a failing close_session must be logged with traceback and must not kill the janitor).
4. Dedupe: `MediaSessionManager` (media.py ~585-776) and `AudioSessionManager` (audio_media.py ~247-404) duplicate refcount/janitor/close/negotiate. Extract a generic typed `SessionManager[HubT, TrackT]` (or composition helper) used by both.
5. Performance: flow/scalar preview conversion is per-pixel pure Python struct.unpack (media.py ~316-400), the overlay draw is per-pixel (overlay.py), audio sample conversion per-sample (audio_media.py ~138-150). Use numpy if available per the dependency rule above. Reduce redundant copies of 1080p frames where straightforward (bytes() → memoryview, avoid double copies).
6. Overlay subsystem: the web never enables overlay (client sends overlay=false) and the server routes /api/media/overlays, media-timestamps, sample, metrics have no production caller. DELETE the overlay subsystem (overlay.py, overlay/frame-mapping code in media.py incl. the per-frame MediaFrameMapping deque, gateway routes for overlays/media-timestamps, and corresponding models in f8media_protocol) — BUT the Studio server (f8studio_server/app.py ~373-399, not yours) still proxies these; to keep the tree importable, keep the f8media_protocol `MediaGateway` Protocol methods that app.py calls only if removing them would break imports, and list exactly which server routes/lines the lead must delete. Prefer: remove them from the protocol and gateway, and report the app.py lines to delete (the lead will delete them right after you finish). Keep `metrics` and `health` if they're useful for diagnostics; `sample` should be removed unless tests/probes in scripts/web_studio use it (check read-only and report).
7. Server-side media proxy duplication: server app.py ~356-406 is a hand-written 1:1 proxy of gateway app.py ~75-116. Provide in f8media_protocol a `create_media_router(gateway: MediaGateway) -> fastapi.APIRouter` (only if fastapi is already a dependency of f8media_protocol or the gateway; otherwise put it in f8media_gateway and report) used by the gateway app itself, and report how the server should mount it (the lead will wire it into app.py).
8. Silent excepts in your area (grep `except` in your files) — fix per AGENTS.md.
Add/adjust tests: hub eviction after failure → new session gets a fresh hub; concurrent acquire of same source opens once; janitor survives a failing close; shared session reused. Keep all gateway tests passing and studio_python_typecheck clean for your files.

FINAL REPORT (your only output that reaches the lead): files changed/created/deleted, per-task summary, exact follow-up edits needed outside your ownership (file:line + what), test commands + results. Under 900 words.
Agent:SDK dedupe and cleanup
IN
You are implementing refactors in the repo /home/sxs/SS/Feel8/f8studio (branch dev). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it strictly (no getattr/setattr/hasattr on known attrs, no silent/broad excepts except at real boundaries with logger.exception + dedupe in hot loops, explicit static typed code). Use pixi: `pixi run python -m pytest -q -p no:cacheprovider <paths>`, `pixi run typecheck`, `pixi run -e web-studio-test studio_python_typecheck`, `pixi run lint`.

CONCURRENCY RULES: other agents are editing other parts of this same working tree right now. You OWN: packages/f8pysdk/** EXCEPT binary_stream_transport.py, video_transport.py, audio_transport.py (owned by the media agent); packages/f8pyengine/** EXCEPT operators/decision.py; packages/f8pyscript/**; packages/f8pydl/**, packages/f8pymppose/**, packages/f8pyaudiofeat/**, packages/f8proclauncher/** (only for adopting the shared error reporter); packages/f8studio_server/f8studio_server/studio_runtime/operators/{_py_expr_eval,data_expr,state_expr,_runtime_errors}.py (ONLY these four files in the server); packages/f8studio_core/f8studio_core/graph/validation.py and compiler.py; packages/f8cppsdk/** (naming only); scripts/protocol_codegen_msgspec.py, scripts/gen_cpp_protocol_models.py. You may add ONE dependency line to /home/sxs/SS/Feel8/f8studio/pixi.toml for datamodel-code-generator (another agent may edit other parts of pixi.toml; re-read right before editing, change only your line). Do not touch anything else; report needed changes instead. Never revert changes you see in files you don't own. No git commit/stash/checkout/reset. Do NOT pip download/install anything manually and do not create files at repo root (a previous agent left .whl files there — don't). Do not kill running processes.

TASKS (from a verified review; line numbers approximate). Do them in this order and keep tests green after each:
1. Expression sandbox unification (security-relevant): there are diverged copies: packages/f8pyengine/f8pyengine/operators/{_py_expr_eval,data_expr,state_expr,_runtime_errors}.py vs packages/f8studio_server/f8studio_server/studio_runtime/operators/{same}.py (~300 differing lines each, different sigmoid semantics), plus own AST allow-lists in packages/f8pyscript/f8pyscript/expr_validator.py and packages/f8pyengine/f8pyengine/wave_expr_lang.py (~60). Create `f8pysdk.expr` (validator/allow-list + evaluator + the shared data_expr/state_expr node core logic parameterized by what differs) and make all consumers thin wrappers. Merge semantics carefully: take the superset/most-correct behavior (e.g. numpy-aware sigmoid) and keep every existing test passing; where the two copies genuinely differ in behavior that a test pins, report it. The studio operators must keep their operator class ids/specs unchanged.
2. Shared `f8pysdk.errors.DedupedErrorReporter(logger, *, interval_s)` with `report(signature, exc, context)` (log once per signature per interval with traceback, count suppressed repeats) — then replace the ~20 ad-hoc `_set_last_error/_should_log_repeating_*/_log_error_once/_record_exception` copies in f8pydl (service_node, optflow, tcnwave, detection_sorter), f8pymppose/service_node.py, f8pyaudiofeat (core, rhythm), f8pyengine operators (data_expr, state_expr, wave_*, handy_out, lovense_out, buttplug_out, recorder, program_wave, auto_sampler). Do not change externally visible state/port behavior (e.g. if a node exposes lastError state keep it).
3. Merge duplicated `video_latest.py` (f8pyengine script_utils vs f8pyscript, ~26 lines differ) and error_reporter overlap into f8pysdk; keep import paths used by user scripts working via a thin re-export only if user-facing scripts import them (check docs/examples read-only).
4. Naming: merge `f8pysdk/zenoh_naming.py` into `f8pysdk/f8_naming.py` (single module; update all importers across the repo — importers in files you don't own: list them in report rather than editing, EXCEPT pure import-line changes which you may make anywhere, re-reading first). Delete dead `zenoh_cmd_key`, `zenoh_endpoint_key`, `zenoh_command_key` in Python and C++ (f8cppsdk zenoh_naming.cpp/.h) — they also disagree with the real keys. Move hard-coded key f-strings from service_bus/runtime.py (~244-246) and service_runtime_tools/deploy/readiness.py (~43,52,130) into the naming module.
5. Delete the dead legacy compiler `f8pysdk/service_runtime_tools/session/` (compiler.py, loader.py) and its test packages/f8pysdk/tests/test_session_compiler.py; update service_runtime_tools/__init__ docs. Verify no importers (git grep) first.
6. Validation single owner: f8studio_core/graph/validation.py (~196-278) and f8pysdk/rungraph_validation.py (~85-377) both validate (different error types); compiler.py ~489 runs both. Make rungraph_validation the single owner of rungraph-level rules raising a typed error with a code; Studio maps it to GraphValidationError codes; remove the duplicate checks. Keep f8studio_core tests passing.
7. Codegen: scripts/protocol_codegen_msgspec.py tries CLI then silently falls back to the Python API with different options (~186-206) — forbidden "guess the API" pattern. Keep ONE path, declare datamodel-code-generator in the pixi `sdk` or `test` feature (whichever the script runs in; one line), regenerate, and verify generated/__init__.py is byte-identical or only trivially different (report diff). If you cannot solve the env without network, still remove the fallback and report. Replace the getattr(generated, "F8...") calls (~78-85) with direct imports.
8. SDK hygiene: micro.py ~494 set_rungraph failure swallowed into reply with no log → logger.exception. control_endpoints.py ~90 returns reqId="" on decode error → echo reqId if parseable. workflow/lifecycle.py ~81-94: try/finally so transport.close() always runs. service_bus/runtime.py ~552-558: fire-and-forget create_task without reference + debug-only log → track tasks and log properly. internal/command.py ~252,273: dedupe key must not include the exception message text. state/router.py ~349-351 hasattr/getattr duck-typing: add a `SubscriptionHandle` Protocol (async unsubscribe) in runtime_transport.py and make subscribe/serve/retained_watch return it (update ZenohTransport and InMemoryTransport). codec.py: remove ignored *args/**kwargs from validate_as/dump_json/copy_model and update call sites (46 sites pass mode="json" — mechanical); dump_json must raise TypeError on unsupported types instead of str() fallback. Delete dead aliases: ServiceBusMicroEndpoints (micro.py ~516), video_frame_schema/audio_chunk_schema (_specs/schema.py ~341-360) if unused, the ignored `queue=` parameter chain (bus subscribe_key → router → zenoh_transport ~233), tuple-watch branch in readiness.py ~73-86. Registry: remove alias methods `register`/`create` and `registry_modules` if unused; keep public behavior.
9. Zenoh pumps (zenoh_transport.py ~373-421) poll try_recv with asyncio.sleep(0.001) per subscription (≈1 kHz wakeups per key). Replace with zenoh callback handlers + loop.call_soon_threadsafe into an asyncio.Queue (bounded, with explicit drop policy for data subscriptions consistent with current semantics). Keep ordering guarantees. Run the SDK tests + any zenoh integration tests.
10. `retained_get` semantics differ between Zenoh (local writes only, ~340) and InMemoryCluster (whole cluster). Rename to make local semantics explicit (`local_retained_get`) OR implement a real Zenoh get; make the in-memory transport match; fix readiness.py usages.

Run: SDK tests, f8pyengine tests, f8pyscript tests, f8pydl tests, f8studio_core tests, `pixi run typecheck`, `pixi run -e web-studio-test studio_python_typecheck`, `pixi run lint`. Baseline before you start: root `pixi run pytest -q` had 11 failures, all fixed or environmental except tests/test_cpp_ci.py and tests/test_cppengine_operator_coverage.py (owned by another agent) — don't chase those.

FINAL REPORT (only output that reaches the lead): files changed/created/deleted; per task done/not done + why; behavior differences resolved in the expression merge; follow-ups needed in files you don't own (file + exact change); test commands + results. Under 1100 words.
Agent:CI and tooling quality gates
IN
You are implementing tooling fixes in the repo /home/sxs/SS/Feel8/f8studio (branch dev). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it. Use pixi for everything (`pixi run ...`).

CONCURRENCY RULES: other agents are editing other parts of this same working tree right now (Python packages, web frontend, server). You OWN ONLY: .github/workflows/**, scripts/quality/**, scripts/update_static_describes.py, scripts/generate_node_atlas.py, scripts/generate_service_docs.py, scripts/service_process_smoke.py and other unreferenced scripts under scripts/ (NOT scripts/dist_ci.py, NOT scripts/protocol_codegen_msgspec.py, NOT scripts/gen_cpp_protocol_models.py), top-level tests/*.py (NOT tests/test_dist_ci.py which the lead just edited), .importlinter, CMakeLists.txt option for demo, services/f8/** yml files (NOT describe.json semantics in code), packages/f8pysdk/f8pysdk/service_runtime_tools/inventory/describe.py (cache invalidation only), and the [tasks] sections of /home/sxs/SS/Feel8/f8studio/pixi.toml (another agent may add one dependency line elsewhere in pixi.toml — re-read right before editing and change only task lines). Do not touch other files; report needed changes instead. Never revert changes you see elsewhere. No git commit/stash/checkout/reset. Do not download/pip-install anything, do not create files at repo root. Do not kill running processes (user has Studio on 8210/8211).

TASKS (from a verified review):
1. There is NO CI quality gate: .github/workflows has only dist-windows.yml and docs-pages.yml. Add .github/workflows/ci.yml (Linux, pixi via prefix-dev/setup-pixi, cache) running: `pixi run pytest -q` (default env), `pixi run -e web-studio-test studio_server_test`, `studio_core_test`, `studio_media_gateway_test`, `pixi run typecheck`, `pixi run -e web-studio-test studio_python_typecheck`, `pixi run quality_exceptions`, `pixi run lint`, web: `studio_web_typecheck` + `studio_web_test` (needs npm ci in packages/f8studio_web). Skip C++ builds in this workflow (note as follow-up) unless trivially cheap. Mirror how dist-windows.yml sets up pixi. Also add a pixi task `ci_checks` that runs the Python + web checks locally in sequence (depends-on list) so devs can run the same thing.
2. Fix the 2 failing root tests: tests/test_cpp_ci.py:166 (expects relative "build", gets absolute tmp path — decide whether the test or scripts/cpp_ci.py is wrong; you may NOT edit scripts/cpp_ci.py if the script is right; if the script is wrong report it) and tests/test_cppengine_operator_coverage.py (reads gitignored stale services/**/describe.json caches and asserts legacy `uiControl` keys). Make tests independent of local gitignored caches: build describes in-process from source registries where possible (Python services: import registry/app and call describe); for C++ engine coverage compare against the C++ source registry (grep/parse registration in packages/f8cppengine sources, read-only) with an explicit `EXPECTED_PY_ONLY = {"f8.fbx_skeleton_player", "f8.decision", ...}` set documented with reasons. Same for tests/test_service_telemetry_state_contract.py (~50 asserts describe_paths exist — must work on a fresh clone).
3. Static describe caches (services/**/describe.json, gitignored) are trusted by discovery (packages/f8pysdk/f8pysdk/service_runtime_tools/inventory/describe.py ~66-99) with no invalidation. Add invalidation: update_static_describes.py writes a sidecar/embedded fingerprint (e.g. hash of the service entry + mtime/hash of the launch target's package sources or binary), discovery ignores a cache whose fingerprint doesn't match (log at info why), and add `--check` mode to update_static_describes.py like generate_service_docs.py --check. Keep it simple and deterministic; document the fingerprint rule in a docstring. Also fix its `except Exception: pass` (~30,44) → json.JSONDecodeError etc.
4. Quality gates: `quality_exceptions` uses --max-broad 844 --max-silent 40 but actual is ~60 broad → set to current actual counts (measure) so the ratchet works. Make scripts/quality/except_metrics.py AST-based: a handler is "silent" if it catches Exception/BaseException/bare (incl. tuples containing them) and its body neither logs (logger.*/logging.*/log_*) nor raises; count `except (Exception, X)` as broad. Update tests/test_except_metrics.py. Report the new counts per package.
5. .importlinter: root_package = f8pystudio (doesn't exist; `pixi run lint_imports` fails). Rewrite contracts for the real packages: f8studio_core must not import f8studio_server; f8pysdk must not import f8studio_*/f8pyengine; nothing outside f8pysdk imports f8pysdk._specs or f8pysdk.service_bus.internal (if existing violations exist, list them as `ignore_imports` with a TODO and report them). Make `pixi run lint_imports` pass.
6. services/f8/cppengine/service.yml is actually a Windows entry (win/...exe) while other native services use service.win.yml — rename consistently if the discovery code (inventory/entry.py) supports it (check read-only), and add a test that every service*.yml pixi `-e` value is one of the environments that exist in pixi.toml and is either `default`, `web-studio-runtime`, `onnx`, or `mediapipe` (the dist script rewrites default and web-studio-runtime to studio-runtime).
7. Service-entry discovery duplicated in scripts/generate_service_docs.py (~294-310) and scripts/generate_node_atlas.py (~47-58, drifted: no darwin branch): make them import find_service_dirs/load_service_entry from f8pysdk inventory.
8. Dead/broken scripts: scripts/service_process_smoke.py imports a non-existent module — fix or delete; scripts/bench_cpp_python_video.py, udp_skeleton_dump.py, udp_skeleton_stub.py, web_studio/probe_studio_presentation.py are unreferenced — delete them unless docs (read-only check under docs/) reference them, in which case add pixi tasks instead.
9. CMakeLists.txt: put f8sdk_demo behind option(F8_BUILD_EXAMPLES OFF).
Do NOT tighten ruff rules or pyright strictness yet (other agents are mid-refactor) — instead write the proposed ruff/pyright config diff in your report.

Run all affected tests (`pixi run pytest -q -p no:cacheprovider tests`, `pixi run lint_imports`, `pixi run quality_exceptions`) and report results. FINAL REPORT (only output that reaches the lead): files changed/created/deleted, per-task summary, new metric counts, proposed ruff/pyright changes, follow-ups outside your ownership. Under 900 words.
Agent:Event and live channel protocol
IN
You are implementing the new Studio push protocol in the repo /home/sxs/SS/Feel8/f8studio (branch dev), both server (Python, FastAPI + msgspec) and web (React/TS). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it strictly (explicit typed code, no silent excepts, no getattr magic). Use pixi: server tests `pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8studio_server/tests`, `pixi run -e web-studio-test studio_python_typecheck`, web `pixi run -e web-studio-test studio_web_typecheck` and `pixi run -e web-studio-test studio_web_test`.

CONCURRENCY RULES: other agents are editing other areas of this working tree simultaneously. You OWN: packages/f8studio_server/f8studio_server/{events.py, monitors.py, app.py (the events websocket, presentation/monitor/state-read routes, and any route you need for this protocol), models.py (models you need), runtime.py (only the state ingest/read path), studio_runtime/presentation.py}, new server modules you create (e.g. live.py), server tests test_events.py/test_monitors.py/test_app.py parts about events/presentation/monitors/state; and on the web: packages/f8studio_web/src/api/**, src/presentation/PresentationStore.tsx, src/graph/useRuntimeNodeState.ts, src/logs/LogsWorkspace.tsx, the websocket/polling parts of src/graph/GraphWorkspace.tsx, src/agents/AgentWorkspace.tsx (ONLY its websocket/refresh part), src/editor/CodeStateWorkspace.tsx (polling part), src/app/App.tsx (provider wiring), new web modules you create, and their tests. Another agent owns packages/f8studio_server/f8studio_server/agents/** — do not edit it; it will add `updatedAt`/`version` to agent.session.updated payloads. Another agent owns f8media_gateway and may later ask for media proxy route removals in app.py — leave media routes alone. Re-read any shared file immediately before each edit. Never revert others' changes. No git commit/stash/checkout/reset. Do not download/install packages (no new npm deps), do not create files at repo root, do not kill processes (user's Studio runs on 8210/8211).

CURRENT PROBLEMS (verified): the server's /api/events (app.py ~851, events.py) supports epoch/after resume but NO web client uses it; every tab opens 4 sockets to /api/events (GraphWorkspace.tsx ~460, LogsWorkspace.tsx ~158, AgentWorkspace.tsx ~231 which never reconnects, PresentationStore.tsx ~151), all receiving high-rate unreliable `presentation.command` and `runtime.monitor` events which fill the 256-slot per-subscriber queues and force resync closes; the resync event reuses the current sequence (events.py ~159, would skip dropped events on resume); `stream.snapshot` with a projects list is sent but unused; monitors are both pushed and polled every 2s (GraphWorkspace ~519-534); node runtime state is polled per node every 1.5s (useRuntimeNodeState.ts, setInterval with overlapping requests, new objects every tick) and the server fans out one read_state per field; deploy jobs are pushed as deploy.* but polled every 200ms with a 20s hard cap that reports longer deploys as failures (GraphWorkspace ~1109-1119); CodeStateWorkspace polls the whole project every 2s (~110) instead of using graph.committed; EventPresentationOutlet.emit creates one asyncio Task per frame (presentation.py ~60) and journal.publish takes a global lock per frame; the web presentation store compares server tsMs against client Date.now() fallback (clock skew drops).

TARGET DESIGN (decided by the lead — implement exactly this; keep it minimal):
A) `/api/events` = ordered, reliable, resumable DOMAIN EVENT LOG only.
 - Remove the `reliable` flag: only reliable events exist in the journal; presentation and monitor data no longer go through it.
 - Client connects `?epoch=<serverEpoch>&after=<lastSequence>` (both optional). First server message is always `{"type":"stream.hello","serverEpoch":..., "sequence":<current>, "resumed": bool}` (typed msgspec struct). If resumed, replay events after the cursor follow; otherwise the client must refetch the REST state it shows. Delete `stream.snapshot` and `stream.resync_required`.
 - On subscriber queue overflow the server simply closes that socket with code 1013; the client reconnects with its cursor (resume works if still within retention). This fixes the sequence bug by construction.
 - Event `type` becomes a closed set: a Python `StudioEventType` Literal/enum used by every publisher (grep all `.publish(` call sites: app.py, automation_tools.py, jobs.py, processes.py, agents/service.py — for files you don't own, do NOT edit; just make publish accept the typed value in a backward-compatible way and list call sites) and a matching TS union.
 - Web: ONE singleton `EventStream` (e.g. src/api/eventStream.ts + React context/hook `useStudioEvents(types, handler)` and `useEventStreamResync(handler)` or similar) that tracks epoch/sequence, reconnects with backoff, parses each message once, dispatches by type, and signals consumers to refetch when `resumed=false` (also on first connect). Replace all per-component sockets with it. AgentWorkspace must thus reconnect.
B) NEW `/api/live` = LATEST-VALUE MIRROR (websocket).
 - Server `LiveValueHub` (new module live.py): `set(key, value)`, `delete_prefix(prefix)`, synchronous and non-blocking (single event loop; no global asyncio.Lock per frame), holds the latest value per key. Each connected socket has a coalescing pending map (key → value | deleted) + asyncio.Event; its sender loop swaps the map and sends one batch. First message `{"type":"live.snapshot","values":{key:value,...}}`, then `{"type":"live.patch","set":{...},"delete":[...]}`. No queue can overflow; intermediate values may be skipped by design.
 - Keys: `presentation/<nodeId>/<command>` → `{"command","payload","tsMs","seq"}` where seq is a hub-assigned monotonically increasing integer (a `.detach` command deletes prefix `presentation/<nodeId>/`); `monitor/<serviceId>/<nodeId>` → F8MonitorSnapshot builtins (keep the existing studio_<epoch>→studio id mapping in monitors.py); `state/<serviceId>/<nodeId>/<field>` → decoded runtime state value (the server already keeps a retained watch cache in runtime.py `_ingest_state` — publish decoded values there, with the same logical-studio-id mapping that read_state applies; check StudioBoundRuntimeGateway in runtime.py).
 - Replace the presentation outlet's per-frame task+journal publish with hub.set (no tasks). Keep `EventPresentationOutlet` API used by viz operators (emit(node_id, command, payload, ts_ms=...)) unchanged — rename the class if you like but keep operator call sites compiling (they're in studio_runtime/operators, not yours; don't edit them).
 - Web: ONE singleton `LiveStore` (src/api/liveStore.ts) with reconnect, a Map mirror, and hooks via useSyncExternalStore: `useLiveValue(key)`, `useLivePrefix(prefix)` returning stable snapshots (only notify listeners whose keys changed; return the same object identity when unchanged). PresentationStore becomes a thin derivation over LiveStore: node output = fold of that node's commands ordered by `seq` using the existing reducer logic (extension reducers, three_d world_up merge) — no Date.now()/tsMs ordering. Runtime node state and monitors come from LiveStore; delete the polling in useRuntimeNodeState, GraphWorkspace monitor polling, NodeInspector polling.
 - Delete now-unused REST routes and client functions: GET /api/presentation, GET /api/runtime/monitors, POST .../nodes/{n}/state:read — BUT first check (read-only) whether packages/f8studio_server/f8studio_server/api_client.py, mcp_server.py, cli.py, agents/**, scripts/web_studio/*.py or e2e specs use them; if used there, keep the route and report.
C) Replace remaining polling with events: deploy job following uses deploy.* events (fetch job once on resync), no 20s cap (a job ends when a terminal event/status arrives; keep a generous stall timeout only if the server can hang, and say so); CodeStateWorkspace refreshes on graph.committed for its project (+ refetch on resync).
Make the graph.committed payload identical in shape wherever published (include runtimeErrors if PatchResult has it) — report call sites in files you don't own.

TESTS: server — hello/resume/replay/non-resumed/overflow-close-then-resume-without-loss tests; LiveValueHub coalescing, delete_prefix, snapshot-on-connect; presentation detach removes node keys; monitor mapping. Web — vitest for EventStream (resume cursor on reconnect, resync callback, single parse/dispatch) with a fake WebSocket, LiveStore (snapshot + patch, identity stability, per-key notification), PresentationStore fold by seq. Update existing tests that assumed old behavior (App.test.tsx, PresentationStore.test.ts, e2e specs only if trivially adjustable — report e2e specs you couldn't update; don't run playwright).

FINAL REPORT (only output that reaches the lead): wire protocol summary as implemented (message shapes), files changed/created/deleted, routes removed/kept (and why), call sites in files you don't own that should switch to the typed event type or payload shape, test commands + results. Under 1100 words.
The five agents are running. The server-internals work that doesn't touch app.py also has no file overlap with them, so I'm starting a sixth agent for it now:


Agent:Server internals refactor
IN
You are implementing server-internal refactors in the repo /home/sxs/SS/Feel8/f8studio (branch dev). Read /home/sxs/SS/Feel8/f8studio/AGENTS.md first and follow it strictly (explicit typed code, no getattr/setattr/hasattr on known attrs, no silent excepts; boundary catches log with logger.exception; dedupe logs in hot loops). Use pixi: `pixi run -e web-studio-test python -m pytest -q -p no:cacheprovider packages/f8studio_server/tests packages/f8studio_core/tests`, `pixi run -e web-studio-test studio_python_typecheck`.

CONCURRENCY RULES: several other agents are editing this working tree now. You OWN ONLY these files under packages/f8studio_server/f8studio_server/: project_repository.py, job_repository.py, assets.py, local_integration.py, native_hotkeys.py, catalog.py, processes.py, editor.py, editor_context.py, lsp.py, projects.py, studio_runtime/operators/viz_*.py, studio_runtime/operators/_viz_base.py, studio_runtime/operators/__init__.py, studio_runtime/registry.py, studio_runtime/service.py, new modules you create (e.g. database.py, hotkey_actions.py, background_tasks.py, studio_identity.py); plus packages/f8studio_core/f8studio_core/graph/store.py; and tests for these. SHARED with care: application.py (another agent may add a LiveValueHub wiring there; re-read immediately before each edit; keep edits minimal). NOT YOURS: app.py, events.py, monitors.py, runtime.py, models.py, studio_runtime/presentation.py, automation_tools.py, agents/**, studio_runtime/operators/{_py_expr_eval,data_expr,state_expr,_runtime_errors}.py, f8studio_core compiler.py/validation.py. If a change is needed there, describe it in your report. Never revert others' changes. No git commit/stash/checkout/reset. Don't download/install packages, don't create files at repo root, don't kill processes (user's Studio runs on 8210/8211).

TASKS (from a verified review; line numbers approximate):
1. SQLite: project_repository.py ~60, job_repository.py ~37, assets.py ~181, local_integration.py ~543 (and agents/repository.py ~22, not yours) each open their own connections, set their own PRAGMAs, create schema; `with sqlite3.connect(...) as c` never closes (only commits); local_integration doesn't enable foreign_keys; deploy_jobs has an FK to projects created by another class; only projects has a schema version; `_text/_integer/_bytes/now` helpers are copied in 3 files. Create one `StudioDatabase` (database.py) owning connect/close (contextlib.closing or explicit close), PRAGMAs (foreign_keys, WAL/busy_timeout as currently used), a single schema-version/migrations table with per-component migration steps, and typed row helpers. Repositories receive it. Keep the on-disk schema compatible with existing user data files (existing DBs must open and migrate; write a test that opens a DB created by the old code path — construct it with raw SQL mirroring the old CREATE statements). Report how agents/repository.py should adopt it.
2. f8studio_core/graph/store.py: undo/redo (~360) hold unbounded full-document copies → cap depth (e.g. 100, constant). In-memory idempotency `_processed/_remember` (~359,523) duplicates the SQLite processed_requests in project_repository (~84) with a different fingerprint scheme (store._fingerprint vs projects._request_fingerprint ~37): keep ONE (the repository's, which survives restart); remove the in-memory one if projects.py always goes through the repository; make sure idempotent retry semantics and tests still hold.
3. Hotkeys: X11 backend (native_hotkeys.py ~520-537) uses the same Display from the listener thread and from asyncio.to_thread workers without Xlib.threaded; polls every 10ms. Use the same command-queue pattern the Win32 backend uses (~239-262): the listener thread owns the Display and executes register/unregister commands. Every graph patch calls refresh_hotkeys → unregister_all + re-register everything (local_integration.py ~473-510) and `_hotkey_status` is mutated without `_hotkey_lock`: diff bindings (register added, unregister removed, keep unchanged), guard status with the lock.
4. application.py ~139-262 has ~130 lines of hotkey action business logic inside the composition root, re-implementing the runtime state-sync decision with different rules from automation_tools.apply_patch (~65-89: job status in {succeeded, partially_failed}) vs item.success. Extract `HotkeyActionService` (hotkey_actions.py) and move generic spec helpers (_enum_values, _schema_default, _pool_values) into f8studio_core/graph (a new small module you create there is fine, e.g. f8studio_core/graph/spec_values.py). Unify the state-sync rule: create a `RuntimeStateSync` helper in your new module that encodes ONE rule; report to the lead the exact automation_tools.py change to use it.
5. catalog.py ~53-57 `refresh` does clear()+re-register on the shared ServiceCatalog that ServiceProcessManager (processes.py ~31) reads without the lock → build a new catalog and swap atomically; the process manager asks CatalogService for the current catalog. (jobs.py ~229 `_ensure_process` silently skips spawn when can_start is False — not yours; report if it should raise.)
6. Fire-and-forget tasks: processes.py ~92-101 create_task without keeping references; create a small `BackgroundTasks` helper (keeps references, logs exceptions with context on completion, cancel+await on close) and use it in your files; viz_video.py ~248 / viz_audio.py ~121 start `_ensure_config_loaded` untracked and close() doesn't cancel it → a late emit after `.detach` resurrects the node; track and cancel on close.
7. Viz operators: viz_wave.close (~72-83) and viz_text.close (~80-92) never emit `.detach` → server snapshot keeps removed nodes; fix. Silent `except ...: pass` at viz_audio.py ~123,137, viz_text.py ~88,92; viz_video.py ~264 logs cleanup failures at debug; viz_wave.py ~163-167 wraps asyncio.sleep in a pointless except → fix per AGENTS.md. Unify: make viz_video/viz_audio/viz_tcode extend StudioVizRuntimeNodeBase (_viz_base.py); move typed state readers (int/float/bool/str with clamp) into the base, replacing copies `_get_int_state` (viz_audio ~194, viz_video ~430, viz_wave ~204) and `_config_state_value` (viz_video ~471, viz_wave ~234); extract a `ThrottledFlusher` component replacing `_schedule_refresh/_flush_after/_flush` in viz_track ~269, viz_three_d ~242, viz_wave ~138; pass `presentation` in constructors and replace the 7 identical factories in operators/__init__.py ~44-86 with one generic factory. Fix viz_video scaleMode default disagreement (spec "fit" ~127 vs runtime "native" ~229,360) — pick the spec default. Keep operator class ids, state field names, and emitted command names/payload shapes unchanged (the web depends on them). Remove config fields that the web never reads ONLY if you can confirm (read-only grep in packages/f8studio_web/src) they're unused AND they don't affect server behavior; otherwise leave and report.
8. Studio service id aliasing is spread over runtime.py ~95-174, monitors.py ~41-49 (hard-coded "studio"/"studio_"), app.py ~754-765: create `StudioServiceIdentity` (studio_identity.py: logical_id, physical_id, to_physical(), to_logical(), is_studio()) — implement and test it; report the exact adoption edits for runtime.py/monitors.py/app.py (not yours).
9. editor.py: one global RLock held across LSP round trips (~176-228) serializes all sessions; no idle timeout/limit on sessions (one basedpyright per abandoned tab). Per-session locks, TTL reaper (idle > N minutes → close), max session count with a clear error.
10. Blocking work in constructors inside create_app: CatalogService.__init__ runs discovery (~catalog.py:34), EditorSessionService.__init__ deletes dirs (~editor.py:100-104); move to explicit async start() steps called from the application's startup (application.py start) — keep minimal.
11. Enum compare by string value: application.py ~152 `port.kind.value == "state"` → compare enum members.
12. server_instance.py ~40 locks per user, ignoring F8STUDIO_DATA_DIR → lock per data dir (you may edit server_instance.py and its test).
Add tests for each behavior change. Keep all server/core tests and studio_python_typecheck passing (failures clearly caused by other agents' in-progress files: note and ignore).

FINAL REPORT (only output that reaches the lead): files changed/created/deleted; per task done/not + why; exact follow-up edits needed in files you don't own (file:line + change); test commands + results. Under 1000 words.
Six agents are now working in parallel, each with its own files:

Stream	Scope
Agents	Stuck approval status, timeout cleanup, per-session locks, a single edit path, splitting service.py, dropped exec outputs in decision nodes
Media gateway	Opening Zenoh off the event loop with one shared session, evicting failed hubs, one shared session manager, numpy conversions, removing overlay
SDK	One expression sandbox, a shared log-dedupe helper, naming merge, deleting the dead compiler, single codegen path, callback-based Zenoh receive instead of 1 ms polling
CI/tooling	CI workflow, fixing the 2 failing root tests, invalidating the describe cache, AST-based exception metrics, fixing .importlinter
Event protocol	/api/events resumable log plus the new /api/live mirror, both server and web; removes polling and the extra sockets
Server internals	Shared StudioDatabase, undo cap and a single idempotency store, hotkey thread safety, atomic catalog refresh, viz operator base class, editor session locks
Once the protocol stream finishes, the second round starts:

Frontend: a queue for graph commits so edits aren't dropped, splitting GraphWorkspace, 3D in-place updates, one shared RTC session pool.
Protocol: a single error format, token auth with exact Origin matching, generating TS types from msgspec, deleting dead routes.
Server: moving business logic out of app.py routes into a project lifecycle service (delete/stop/restart/import/restore).
Last: tightening ruff and pyright.
I won't commit anything unless you ask. I'll report back as each stream finishes.