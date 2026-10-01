# 工具、Skill 与资源扩展

Extension 是安装和版本管理单元，可以组合提供服务、一次性工具、Agent skills 和参考资源。工具扩展不需要 node，也不需要非空 `serviceClasses`。这套架构尚未把 Unity 安装功能迁移进扩展，现有 Unity/Local 集成保持原状。

## 发布清单

ZIP 的根目录包含 `config/extensions.json`。只有提供服务的包才需要 `config/service-index.json`；没有服务的包可以省略它。`serviceClasses`、`tools`、`skills` 和 `resources` 至少声明一项。

```json
{
  "schemaVersion": "f8extensionCatalog/1",
  "extensions": [{
    "extensionId": "example-tools",
    "name": "Example Tools",
    "version": "1.0.0",
    "description": "An example one-shot tool package",
    "runtime": { "kind": "native" },
    "tools": [{
      "toolId": "inspect",
      "name": "Inspect target",
      "description": "Inspect a directory and return a result",
      "command": "${F8_PACKAGE_ROOT}/bin/inspect",
      "workdir": "${F8_PACKAGE_ROOT}/bin",
      "fields": [{ "name": "target", "label": "Target directory", "kind": "string", "required": true }],
      "platforms": ["linux"],
      "timeoutSeconds": 300,
      "requiresConfirmation": true
    }],
    "skills": [{ "skillId": "workflow", "path": "${F8_PACKAGE_ROOT}/skills/workflow/SKILL.md" }],
    "resources": [{ "resourceId": "profiles", "path": "${F8_PACKAGE_ROOT}/resources/profiles.json", "description": "Known configurations" }]
  }]
}
```

原生工具的可执行文件必须位于包内，可用 `platforms` 声明支持的平台。Python 工具声明 `command: python`，复用扩展声明的受管理 `bundled`、`workspace`、`pixi` 或 `shared` 运行环境；`shared` 工具使用 `args: [-m, module]`，运行扩展安装后的隔离代码。Studio 不按字符串导入或调度扩展 Python 方法。

工具入口和素材应随包提供。此阶段由发布者构建 ZIP；SDK 的 service wheel/runtime 打包入口仍用于服务扩展，不宣称它已自动打包任意工具目录。

## 输入、任务与结果

字段支持 `string`、`integer`、`number`、`boolean`，以及字符串 `choices`、默认值和必填声明。页面根据清单生成表单；服务端检查字段名称、值类型、枚举值和输入长度。参数作为 JSON 写入进程 stdin，不拼接 shell 命令。

```json
{"schemaVersion":"f8toolInput/1","arguments":{"target":"/games/example"}}
```

工具把日志写到 stderr，stdout 只输出一个结果对象：

```json
{"schemaVersion":"f8toolResult/1","success":true,"message":"Inspection complete","data":{"matched":false}}
```

任务状态为 `queued`、`running`、`succeeded`、`failed`、`cancelled`。非零退出、非法结果、超时或输出超限都会保留错误信息和 traceback。stdout/stderr 各限制 1 MiB。任务记录保存在 Studio 数据目录的 `tool-jobs/`，包括输入、扩展版本、结果和诊断；Studio 重启时将未完成任务标记为失败，不自动重新执行。取消和关闭 Studio 会结束受管理的工具进程。

一个扩展同时执行一个工具任务。执行期间不能停用或卸载该扩展。卸载只清理扩展安装，不删除工具已经修改的外部游戏目录；任务历史也保留。

## 页面和 Agent

Tools 工作区列出已安装且启用的工具，提供表单、执行确认、任务结果和取消入口。Extensions 页面继续管理包生命周期，并显示服务、工具和 skill 数量。

页面和 Agent 使用同一个任务服务：

- `GET /api/extension-tools`
- `POST /api/extension-tools/{extension_id}/{tool_id}/run`
- `GET /api/tool-jobs`、`GET /api/tool-jobs/{job_id}`
- `POST /api/tool-jobs/{job_id}/cancel`
- `GET /api/extension-resources`
- `GET /api/extension-resources/{extension_id}/{resource_id}`（UTF-8 文本）
- `GET /api/extension-resources/{extension_id}/{resource_id}/file`（下载素材）

默认工具要求 `confirm: true`。Agent 的通用执行入口始终通过现有审批 UI 确认输入，再返回 job ID；可查询任务结果。工具审批不绑定 graph revision，图编辑审批仍保留原有 revision 校验。

扩展 skill 使用 `<extensionId>:<skillId>` 名称，避免与用户/内置 skills 冲突。只有已安装且启用的扩展提供 tools、skills 和 resources；停用后立即撤出。skill 可以指导 Agent 通过 resource ID 获取 profiles 和参考资料。资源必须在清单中声明，解析后不能越过包边界；文本读取限制 1 MiB，skill 限制 64 KiB。插件二进制等素材使用文件下载接口。

扩展包执行在独立进程中，避免导入 Studio 核心；这不提供不可信代码沙箱。游戏检测、安装幂等性、修改预览和经验记录由具体工具实现。本阶段没有新增 Unity、Spine 或 Live2D 功能，也没有把工具扩展自动暴露为 MCP 工具；当前 Agent 的模型工具接口已接入。
