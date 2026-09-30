# 大清洗审计与执行清单（清理前快照）

执行状态（2026-09-30）：本清单中的清理项已实施，结果、验证和迁移方式见 [完成记录](CLEANUP_REPORT.md)。下文保留清理前证据，文件行号和基线结果不代表当前代码。

审计日期：2026-09-29。范围：仓库自有 Python SDK、Studio Core/Server/Web、C++ SDK 的协议与路由，以及服务静态描述、构建入口和相关测试。初始审计阶段仅新增本文档，未改动运行逻辑。第三方代码、外部部署和用户已有工程未做穷尽审计。

结论：优先消除协议多源和校验旁路，再删除无生产调用的旧入口。不能按 `legacy` 关键词批量删除：部分转换仍被仓库里的静态描述实际依赖。

## 1. 优先级最高：协议字段存在多套真值

- `schemas/protocol.yml:542` 的 `F8DataPortSpec` 同时保留 `payload` / `stream` 与顶层 `valueSchema` / `payloadKind` / `delivery`。
- `packages/f8pysdk/f8pysdk/_specs/schema.py:383` 起的端口工厂同时写入新旧字段；schema 和 delivery 被重复表达。
- Python `data_port_payload_kind()`（同文件第 146 行）优先 `payload.kind`，再看 `payloadKind`，最后根据 schema 猜测。
- C++ `packages/f8cppsdk/src/rungraph_routes.cpp:185` 同样优先 `payload.kind`。
- Web `packages/f8studio_web/src/graph/connectionRules.ts:16` 却优先 `payloadKind`。

**影响**：例如顶层 `payloadKind=json`、嵌套 `payload.kind=video_frame` 时，前端与运行时判断不同。这是代码可直接证明的优先级冲突；本次没有增加端到端复现测试。

**清理方案**：先统一消费规则并对冲突输入报错；迁移所有端口生产者和静态描述后，以 `payload`、`stream` 为目标规范，逐步移除旧字段及 schema 形状推断。连带检查图编辑器、编译器、语义 revision/fingerprint、路由和文档生成。修改协议源并重新生成 Python/C++/TS/API 合约，不手改生成文件。

**验收**：Python、C++、Web 对同一端口 fixture 得出一致结果；冲突被明确拒绝；媒体边仍走二进制流；仅 UI 元数据变化不改变语义 revision。

## 2. 校验旁路与旧 authoring 转换必须分步拆除

- `packages/f8pysdk/f8pysdk/service_runtime_tools/inventory/describe.py:304`：`F8ServiceDescribe` 校验失败后只记 debug，若存在 `service` 就继续返回字典，并补默认 `operators`。
- `packages/f8pysdk/f8pysdk/_specs/builtin_fields.py:195`：转换 `required`、字符串 `uiControl`、`canEditRequired`、字符串 exec ports，并丢弃旧 `launch`。
- `packages/f8cppsdk/src/describe_builtins.cpp:195`：也有旧描述转换逻辑。
- 实际依赖仍存在：`services/f8/audiocap/describe.json:42` 含 `uiControl`，多个服务静态描述含 `canEditRequired`。

**清理方案**：先核对当前服务 `--describe` 和静态缓存、修正生产者并重新生成；对所有服务描述严格校验通过后，删除旧转换。发现非法描述应记录服务路径、校验位置和 traceback，并在 discovery 边界拒绝该服务；不能继续传递未经验证的数据。旧用户数据如需保留，使用独立的一次性迁移工具，不在运行时反复归一化。

**验收**：逐个严格解析仓库静态 describe；非法字段的负例拒绝；catalog refresh 与部署测试通过。

## 3. 可优先删除的旧入口与别名

以下“无调用”指仓库检索未发现生产调用，不代表能证明外部 SDK 使用方不存在。

| 候选 | 当前证据 | 执行动作 |
| --- | --- | --- |
| `f8pysdk/service_runtime_tools/session/` | 3 个文件共 450 行；编译入口与 loader 的仓库调用仅来自该目录导出和专用测试。Server jobs 已调用 `f8studio_core.compile_document` | 删除旧 session loader/compiler/export；保留或迁移仍有价值的编译语义测试到 Core；不让 SDK 反向依赖 Studio Core |
| `video_frame_schema` / `audio_chunk_schema` | 仅定义和 `specs.py`、`_specs/schema.py` 导出，未发现调用 | 删除别名和导出；元数据调用保留明确的 metadata helper，端口声明使用 port factory |
| `zenoh_endpoint_key` / `zenoh_cmd_key` | Python 只有定义、导出和旧路径断言；C++ 只有声明/定义。当前服务控制端点使用 `svc_endpoint_key` | 删除旧 helper、`schemas/runtime-keys.json` 中对应模板及旧测试断言；重新生成 naming 产物 |
| Viz base 的 `presentation` setter | `_viz_base.py:86` 明确标记兼容；未发现节点 setter 调用，工厂已构造注入 | 移除 setter；随后逐个检查构造路径，再决定是否把 outlet 改成必填 |
| `tests/test_shm_region_lifetime.cpp` | 引用已不存在的 `f8cppsdk/shm_region.h`；当前 CMake 未注册该测试 | 删除孤立测试文件 |
| README 的旧 SHM 路径 | 宣称支持 `legacy_shm`，并给出不存在的 `scripts/audioshm_viewer.py` 命令 | 修正文档；不要据此误删仍在使用的 Zenoh 自有 SHM 优化 |

## 4. Transport 的接口伪装与动态生命周期

- `packages/f8pysdk/f8pysdk/runtime_transport.py` 将 subscribe/serve/retained_watch 的返回类型全部写成 `Any`。
- `service_bus/state/router.py:350` 通过 `getattr(stop)` / `getattr(unsubscribe)` 猜关闭方法，再判断是否为 coroutine；没有关闭方法时直接返回。
- `service_runtime_tools/deploy/readiness.py` 仍支持 tuple 形式的 watch，另有 `watch.stop()` 路径。
- 当前 Zenoh 与 InMemory watch 实现已有明确关闭方法，可统一协议。
- `zenoh_transport.py:253` 接收 `queue` 后直接 `del queue`，上游仍逐层传递这个参数。

**清理方案**：定义明确的订阅句柄 Protocol，统一 `async unsubscribe()`，修改实际实现、Protocol、调用方与测试替身；删除 tuple 分支与运行时猜 API。核实 InMemory 和所有调用点不依赖 queue-group 语义后，移除无效 `queue` 参数链；保留真正的数据 FIFO 和队列容量配置。

`zenoh_config.py` 还会写旧版 pool key，而 pixi 已约束 Zenoh 1.9。应在当前支持版本实际验证配置键后删除旧写入，并明确哪些配置失败必须报错。不能把所有 optional 配置都当作无用兼容。

## 5. Codec 包装隐藏了错误

`packages/f8pysdk/f8pysdk/codec.py:121` 起：

- `validate_as` / `dump_json` 接收并忽略任意额外参数。
- `copy_model` 仅提取 `update`，其余参数被忽略；不支持的对象可能原样返回。
- `_coerce_json_compatible` 最后用 `str(value)` 包装未知对象；循环容器转换为 `None`，可能把错误数据伪装成合法 JSON。

**清理方案**：给 helper 显式签名，删除调用方无效参数；对支持的 msgspec、JSON、数值类型明确转换，对不支持类型和循环引用报错。Codec 内通用序列化的动态访问属于仓库规则允许的例外，重点是转换契约与错误可见性，不是机械移除每个 `getattr`。

**验收**：UNSET/null、数值标量、容器、模型复制、未知对象、循环对象分别有明确结果；失败从边界得到可定位日志。

## 6. 重复状态：已有证据，但需要按语义收敛

### Presentation

`packages/f8studio_server/f8studio_server/studio_runtime/presentation.py:34` 的 `_latest` 保存所有命令；内置高频命令同时写入 `EventJournal.live`。Web `PresentationStore.tsx:117` 恢复 REST snapshot 时却排除内置 renderer，仅恢复扩展命令。

**候选方案**：让内置 latest-value 输出只有 LiveValueHub 一个服务端权威存储；扩展增量命令继续使用事件流及其恢复机制。先核对 snapshot 的其他消费者、detach、重连、排序和资源上限，再收缩 `_latest`。现有测试明确要求 snapshot 保留内置命令，所以这是合约调整，不是死字段删除。

前端 `outputsSnapshot`、prefix selection 等副本服务于 React 稳定快照和选择性通知，不能因“重复 Map”直接删除。`connected` 当前由事件流驱动，而内置输出走 live socket，连接状态语义也应重新定义。

### Agent provider 设置

`agents/provider_settings.py:161` 同时处理默认 model、models 列表、旧 supports_image 和每模型 capabilities，在每次 view 中合成 legacy capability。前端还有 legacy provenance 保留逻辑。

**候选方案**：把旧图像能力迁移为对应模型的 capability，默认模型只保留选择关系，能力展示从模型信息派生。迁移必须保留人工确认、thinking 来源，以及“删除配置后不能被环境默认值复活”的行为。不能直接删除 `legacy` 分支或枚举。

## 7. 明确保留的边界

- `StudioBoundRuntimeGateway` 做逻辑 Studio ID 到私有实例 ID 的映射和特殊停止处理，不是空转包装。
- `VideoSessionPool` / `AudioSessionPool` 已共用 `RtcSessionPool`，各自仍承担 source/quality、媒体类型和 API 适配。
- `StateWriteOrigin` 与 `StateWriteSource` 分别用于权限和诊断/传播，不能仅因枚举项重叠而合并。
- 服务配置、生命周期、低频语义状态保留 state；延迟/FPS/逐帧计数使用 monitor/data。本次定向搜索未证明存在新的高频 state 违规，不作“全部合规”的结论。
- 骨骼协议 v1/JSON 支持涉及外部生产者，未审计实际部署版本，不列入可直接删除清单。

## 8. 建议实施批次与验收

1. **无生产调用残留**：旧 session 模块、死别名、旧命名模板、孤立 SHM 测试、失效文档。迁移有价值的测试覆盖，检查公开 API 移除说明。
2. **收紧接口**：订阅句柄、无效 queue 参数、Codec 显式签名；跑 SDK 和消费者测试、类型检查。
3. **协议单源**：先统一三端读取与冲突处理，再迁移生产者/静态描述、删旧转换和字段；生成合约后检查漂移，跑 Python/C++/Web 路由测试及真实媒体部署/重启验证。
4. **状态归属**：Presentation 和 provider 配置分别实施；覆盖重连、detach、扩展增量、配置迁移和删除语义。

每批保持可独立回退；不新增永久兼容包装来掩盖迁移缺口。API 和协议变更需在发布说明中明确。

## 9. 本次验证

已执行：

```sh
pixi run pytest packages/f8pysdk/tests/test_session_compiler.py packages/f8pysdk/tests/test_session_loader.py packages/f8pysdk/tests/test_data_flow_routing.py packages/f8pysdk/tests/test_builtin_state_fields.py packages/f8studio_core/tests/test_graph_domain.py packages/f8studio_server/tests/test_live.py -q
```

结果：**84 passed**。这是选定路径的现有基线，不是清洗后的验证。尚未运行全量 Python 测试、静态类型检查、Web/C++ 测试、真实 Zenoh/WebRTC 集成验证，也未逐个严格校验静态 describe。
