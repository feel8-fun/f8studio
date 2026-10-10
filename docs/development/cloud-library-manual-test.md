# Cloud Library 本地实测

可连接支持 Library v2 的 Cloud server 实测；需要独立测试账号和数据库时，可用本地 Cloud 沙盒验证完整交互。沙盒运行真实 Cloud API、登录流程和所有 SQL 迁移，使用独立 SQLite 数据库；不会访问线上数据库。它不是生产 Worker/D1 的部署验收。

在仓库根目录另开一个终端，保持前台运行：

```sh
pixi run -e web-studio cloud_sandbox
```

Cloud URL 为 `http://127.0.0.1:8787`。默认数据保存在 `cloud/.wrangler/library-sandbox/`，重启会保留账号、作品、版本、点赞和关注。Ctrl+C 停止；端口被占用时可指定另一个端口：

```sh
pixi run -e web-studio cloud_sandbox --port 8788
```

| 测试账号 | 密码 | 用途 |
| --- | --- | --- |
| `author@sandbox.test` | `sandbox-password` | 发布和管理自己的作品 |
| `reader@sandbox.test` | `sandbox-password` | 使用他人的作品、点赞、关注和派生创作 |

两个账号都是已验证的普通用户。沙盒初始为空，由作者从 Studio 发布真实模板；不把缺失扩展的测试 fixture 当作可用组件。

## 作者流程

1. 使用你自行启动的 Studio。前端已重新构建；重启 Studio 加载新的服务端后刷新网页。
2. 在图中添加 PyEngine 和 Tick，修改 Tick 的频率，右键 Tick 保存 Variant；也可选中多个节点保存 Component。
3. 进入 Assets → My Local。用 All / Graphs / Components / Variants 筛选本地图和草稿，列表共用一个搜索框；选中草稿后确认预览和配置正确，编辑名称、Markdown 介绍后点击 Save draft。未保存的编辑不能发布。
4. 打开全局导航栏右上角 Settings → Cloud，在 Cloud URL 输入沙盒地址，点击 Save connection，再点击 Sign in to Cloud，使用作者账号登录。
5. 点击草稿的 Publish Variant to Cloud 或 Publish Component to Cloud，确认许可、可见性和版本说明，点击 Publish。应得到 v1。
6. 再次发布相同内容，应显示 Content unchanged · v1。仅编辑或运行图不会上传。
7. 进入 My Cloud，修改云端介绍或可见性。云端 metadata 保存不增加内容版本，也不改本地草稿。设为 Private 后其他账号不能发现或读取。
8. 点击 Edit local draft，修改模板内容并保存，再 Publish update。应生成 v2，v1 仍可从版本选择中预览。
9. 在 My Cloud、Discover、Following 切换，确认导航、搜索和作品列表都在主左侧栏，右侧只显示选中作品的详情。
   在搜索框下切换 All / Graphs / Components / Variants，确认只出现所选类型，
   More results 继续加载同一类型；无结果时提示筛选无匹配。可以分别发布同名的
   Graph、Component、Variant，确认 All 中通过图标和类型区分，删除或点赞其中
   一份不会影响另外两份。更新 Studio 和 Cloud 服务端后才具备类型筛选能力。
10. 在自己的作品详情右上角 More publication actions 选择 Delete Cloud publication。取消应保留作品；确认后作品和所有发布版本消失，本地草稿、项目及已插入节点保留。返回草稿后可重新发布为新的云端作品。

完整图从 Assets → My Local → Graphs 的 Publish Project to Cloud 发布，发布前可勾选本次携带的配置。运行值、本机路径等受字段策略限制；排除某个实例值不删除节点定义中的 default/examples。

## 本地快照

1. 在 Assets → My Local → Graphs 选择项目，点击 Save snapshot，填写名称和备注。它只保存本地图检查点，不会上传到 Cloud。
2. 在 Local snapshots 中编辑名称和备注，确认图内容和快照创建时间不变。
3. 修改图后，点击快照的 Restore 并确认，当前图应恢复为保存时的内容，revision 作为新修改递增。
4. 删除快照需要确认；当前图与云端发布不受影响。
5. 返回 My Local 的 All，可在同一个列表切换本地图、Variant 和 Component。草稿发布选项在独立对话框里，JSON 编辑和导出、删除从 More draft actions 进入；图详情仍提供 Local snapshots。

## 使用者流程

1. 在 Settings → Cloud 点击 Sign out 后重新登录 reader。Cloud 授权页若仍记住作者账号，可填写 reader 的邮箱密码并使用 Use a different account。
2. 进入 Discover，按名称、标签或作者搜索。选择作品，应显示 Another author’s Cloud work，不出现云端修改和删除操作。
3. 点赞/取消点赞、关注作品和作者。在 Following 中验证结果；取消点赞不应取消关注。
4. 用 Version to use 选择 v1。预览、Add node、Create local draft 或 Open as local project 均使用所选固定版本。
5. 在图的 Add 搜索框选择 Online，搜索模板、查看详情并添加。插入展开为普通节点，可一次撤销；浏览和添加不会自动创建本地草稿。
6. Create local draft 后修改本地内容，再发布。应创建自己的新作品，并保留原作品来源；原作者的作品不变。

## 中断与恢复

- 关闭沙盒后，本地草稿和节点库仍能使用；在线查询显示错误并提供 Retry search / Retry details / Retry component / Retry variant。
- 重启同一个沙盒后重试，原作品、历史版本、登录凭据和社区关系仍有效。
- 发布回包丢失时，刷新网页仍可 Retry publication；重试保存原 request ID、版本和配置排除选择，不自动上传后续编辑。
- 删除回包丢失时可在确认框重试；不会因内容已删除而卡住。旧发布请求不能重新恢复已删除作品。
- Disconnect Cloud 明确断开本机连接；本地草稿和云端作品都不会被删除。

线上验收另需 Cloudflare 认证、生产 D1 增量迁移和 Worker 部署；本地沙盒通过不代表生产域名已经可用。
