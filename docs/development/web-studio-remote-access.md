# Web Studio 远程访问

本文说明现行 HTTP、WebRTC 与 SSH/TURN 连接方式。基本启动见 [Studio 使用指南](../getting-started/studio.md)。

VPN/LAN 直连模式必须用 `--allowed-host` 显式列出可信地址；默认仍只监听 loopback。通配 bind 不会通配 HTTP Host/Origin。Gateway 控制端口保持 loopback。WebRTC 媒体不经过 Studio HTTP 端口，客户端还必须能访问 Gateway SDP 中公布的动态 UDP ICE candidate；若 VPN 或防火墙不允许该流量，仍需 TURN/TCP/TLS 或明确的 UDP 端口策略。

## 严格 VPN / SSH 模式

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
  <user>@<server>
```

浏览器从 `/api/media/rtc-configuration` 读取运行时 ICE 配置。启用 `--force-turn` 后使用 `iceTransportPolicy=relay`，并在发送 offer 前等待 ICE gathering 完成；这使非 trickle TURN candidate 确实进入 SDP。2026-09-22 使用真实 Chrome 强制 TURN/TCP 验证 synthetic 视频、音频和波形均通过，coturn allocation 计数证明媒体走 relay。loopback TURN 模式依赖 SSH 身份验证，只适合受控远程开发。Studio 在此模式下也保持 loopback bind；Host allowlist 不是身份验证，不应为了 SSH 转发绑定公网接口。面向多用户部署应使用 TURN/TLS 443、短期凭据、正式证书和应用层身份验证。
