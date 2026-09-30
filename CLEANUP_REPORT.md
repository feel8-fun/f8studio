# 大清洗完成记录

日期：2026-09-30。范围为 [清理前审计](CLEANUP_AUDIT.md) 中列出的仓库自有路径；所有修改保留在工作区，未提交，也未修改用户已有工程或配置。

## 已完成

- **协议单源**：数据端口只保留 `payload` / `stream`；删除顶层 `valueSchema`、`payloadKind`、`delivery` 及运行时 schema 形状推断。同步更新 Python/C++ 生产者、Studio 编译器、Web 编辑器、文档解析器、测试和生成合约。
- **严格 authoring 校验**：移除 Python/C++ 运行时旧字段转换；非法 describe 拒绝进入 catalog，输出上下文及 traceback。描述集合不再静默过滤非法元素；只接受 describe envelope。22 份静态描述已从当前服务重新生成并验证。
- **旧入口删除**：删除 SDK session loader/compiler，使用 Studio Core 编译入口；删除旧媒体 schema 别名、Zenoh 命名 helper/template、MicroEndpoints 别名、注册器 register/create 别名及无调用的动态模块加载入口、孤立 SHM 测试。修正 README 失效路径和任务名。
- **接口明确化**：订阅句柄统一 `SubscriptionHandle.async unsubscribe()`；移除 tuple watch、生命周期猜测、无效 transport queue 参数。Zenoh 配置使用当前版本支持的键，配置错误直接暴露。
- **Codec 收紧**：显式参数签名，移除被忽略的调用参数；真实深复制；保留 UNSET/null、NumPy 数值和 msgspec discriminator；循环引用和不支持类型明确报错。
- **状态归属收敛**：内置 presentation 输出仅存于 LiveValueHub，REST snapshot 按需派生，扩展命令独立保留。连接状态要求 event/live 两条连接均就绪；Viz outlet 必须通过构造注入，删除 setter 和占位工厂。
- **Provider 配置**：能力以 modelCapabilities 为唯一持久化来源，supportsImage 响应按模型派生；保留人工能力、thinking 来源及未修改模型的配置。旧持久化字段只由离线工具转换。

## 调用方与数据迁移

这些是有意的破坏性 API/协议变更；外部 SDK 调用方也需要同步更新。

当前数据端口示例：

```json
{
  "name": "values",
  "payload": {"kind": "json", "valueSchema": {"type": "number"}},
  "stream": {"delivery": "fifo"}
}
```

媒体使用 `payload.kind = video_frame / audio_chunk` 和 `payload.metadataSchema`；未指定 delivery 时按 FIFO 处理，需要 latest 时显式声明。Authoring exec ports 是对象（例如 `{"name":"run"}`），runtime exec ports 仍是字符串。

旧 JSON 文档、describe 或 f8graph v3 导出文件，可对备份副本执行：

```sh
pixi run python scripts/migrate_authoring_contracts.py path/to/document.json
pixi run python scripts/migrate_authoring_contracts.py --providers path/to/providers.json
```

脚本**原地改写传入文件**，不会遍历改写任意 stateValues 或用户 JSON schema；f8graph v3 会重算 definitionRef 并验证导入。不自动迁移 SQLite 工程数据库，已有数据库中的旧工程应使用旧版本先导出，再迁移文件并导入。运行时不再自动接受旧 authoring 字段。

调用方改用 `video_frame_metadata_schema` / `audio_chunk_metadata_schema` 或对应 port factory、`svc_endpoint_key`、`register_operator_factory` / `create_runtime_node`；注册函数显式接收 registry，Viz 工厂显式注入 presentation。删除 codec 调用中无效的 mode/by_alias 参数及 transport queue 参数，订阅资源通过 `await handle.unsubscribe()` 释放。

## 验证

- 全量 Python：`pixi run pytest packages tests -q --disable-warnings`，**1052 passed / 7 skipped / 1 warning**。后续描述校验收紧另跑定向回归，**34 passed**（包含新增的 4 个畸形集合用例）；SDK 全套复验 **279 passed**。
- Web 单元测试：**23 files / 106 tests passed**；生产构建成功，仅保留 bundle 体积提示。
- 浏览器：desktop 图编辑、代码状态、媒体与 Provider 设置共 25 项；首次 24 项通过，1 项因测试未初始化认证 cookie 失败。修复为共享浏览器已认证 request 后，Provider 设置文件 **3/3 通过**；其余 22 项此前通过。
- C++：完整构建与 runtime 部署成功；CTest 注册项全部通过；CPython/Lua smoke 可执行程序退出码为 0。
- 静态检查：lint、Python 类型检查、Studio 严格类型检查、3 条 import contracts 全部通过；异常审计无 silent except；生成 API 合约检查无漂移。
- 真实运行时：PyEngine 启动/部署及数据处理通过；WebRTC 解码 640×360 连续变化帧通过；重启恢复约 0.231 秒，session/source 均保持 1。浏览器 1080p 视频延迟探针 60 个样本，p95 73ms（仅代表本机本次测量）。测试服务已停止。

## 保留边界与限制

- 保留外部 LMEX v1 协议、StudioBoundRuntimeGateway ID 映射、媒体池 API 适配、StateWriteOrigin/Source、React 稳定快照以及有意义的 legacy 来源枚举。
- 完整服务文档生成仍被仓库缺失的 `docs/modules/manual/operators/f8-cppengine/f8-data-mux.md` 阻断；本次文档解析器回归通过，但不能宣称整站文档生成成功。
- 第三方代码、外部部署及用户数据库不在本次穷尽审计或自动迁移范围内。
