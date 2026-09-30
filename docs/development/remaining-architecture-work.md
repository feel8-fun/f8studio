# 后续架构工作

2026-09-30 核对当前代码后保留的结构性提案。这些事项尚未作为已确认的运行时缺陷；实施前应明确归属、兼容性和回归范围。

- `f8pyengine/operators/script_utils/video_latest.py` 与 `f8pyscript/video_latest.py` 仍有两份实现。评估能否在 SDK 共享订阅、解码和生命周期逻辑；同时核对脚本错误报告的重复实现。
- 表达式求值及图校验仍由不同层实现。统一之前要逐项对照允许语法、数值语义、错误报告和各调用方的输入契约。
- `f8studio_server/runtime.py` 与 `monitors.py` 各自处理逻辑 `studio` 和私有 `studio_<epoch>` 身份。若收敛到共同的身份对象，需覆盖部署、状态、监控和事件中的双向映射。
- `SystemOneDecisionClient` 仍由 `AgentService` 创建，决策路由经 `studio.agents.decisions` 调用。可评估独立生命周期和依赖边界；Agent 会话持久化仍按完整记录保存，优化时需保留恢复语义。
- 内置 presentation 渲染仍有显式分支，扩展渲染走 registry。统一时需覆盖快照恢复、媒体和 3D 输出。C++ 服务入口样板是否值得抽取，应在核对现有构建目标后单独决定。

历史审阅中提出的公开路由删除、GraphStore 独立幂等移除、命令状态通道替换、离线 provider 删除及无 schema 变更时引入 SQLite 迁移框架，均不是已确认的清理任务；先验证调用方和行为契约再决定。
