# 后续架构工作

2026-09-30 核对。视频订阅共享、表达式 AST 白名单、Studio 身份映射、决策客户端生命周期和 presentation registry 已完成收敛，实施流水账已清理。以下保留项尚未作为已确认的运行时缺陷。

## 待评估的结构改进

- **Agent 会话持久化**：继续以完整记录原子保存。`AgentSessions._mark_interrupted_sessions` 在重启时恢复会话和工具调用状态；没有写放大测量或 schema 变更需求时，不引入增量日志或迁移框架。后续优化需先测量记录尺寸、保存频率，并覆盖工具结果与待审批状态恢复。
- **C++ 入口样板**：已核对 `f8cppengine_service`、CVKit 五个服务、音频采集、屏幕采集和播放器的 CMake 目标及入口。CVKit 存在相似启动代码，但音频设备枚举/SDL、平台屏幕采集、播放器 mpv 和 Engine 的链接与启动需求不同。当前保留 SDK 已有 `runtime_cxxopts` 和 describe helper；如抽取服务 runner，应单独覆盖各平台构建、`--describe`、专用 CLI 和退出顺序。
- **进一步共享表达式与错误报告**：需先为上表中的差异设计显式策略，并证明不会扩大语言权限、改变异常/监控时序；当前不以减少文件数为目的强行统一。

公开路由删除、GraphStore 独立幂等移除、命令状态通道替换、离线 provider 删除，仍不是已确认的清理任务。

## 表达式与图校验的兼容性边界

进一步共享实现之前，需保持下表中的调用方契约。三个 Python 校验器已经共用 SDK 的 AST 节点白名单；函数与求值语义仍不同。

| 契约 | Engine | PyExpr 服务 | Studio 内置运行时 |
| --- | --- | --- | --- |
| 全局函数 | 包含聚合、range、sorted、sigmoid | abs/float/int/min/max/round | 与 Engine 的函数名称相同 |
| NumPy | 可选启用 | 可选启用 | 明确拒绝启用 |
| sigmoid | 标量及 NumPy 数组 | 未暴露 | 仅标量 |
| JSON 序列索引 | 经 `int(key)` 转换 | 经 `int(key)` 转换 | 仅整数，拒绝 bool |
| JSON 包装 | dict/list/tuple | dict/list/tuple | Mapping/list/tuple，支持 len |
| 错误契约 | 编译错误字符串；求值异常交调用方处理 | 编译错误字符串；求值返回结果/异常对象 | 独立 NumPy 禁用消息、编译前缀和代码文件名 |

因此不合并求值器、JSON 包装器、函数集合或错误报告。共享语法不扩大任何调用方的函数权限，也不改变数值语义。

图校验继续分层：`f8studio_core.graph.validation` 接收编辑文档，验证 ID、端口、状态 schema 和 UI 配置，并提供结构化错误码；`f8pysdk.rungraph_validation` 接收运行图，校验 exec/state/data 边及 payload kind，允许部分端点描述缺失时跳过相应检查。二者输入完整性和错误契约不同，不应相互替换；已有 `data_port_payload_kind` 等协议规则继续共享。

## 脚本错误报告的兼容性边界

Engine 使用同步 bus，具有待发送错误、恢复清除和 `error_seq`；PyScript 使用异步回调，独立表达式服务还使用不同错误码。当前保留适配器。进一步抽取需覆盖发布前 bus 未就绪、重复错误限频、恢复清除与异步关闭，不改变监控时序。
