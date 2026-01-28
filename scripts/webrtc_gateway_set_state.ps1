param(
  [Parameter(Mandatory = $false)]
  [string]$ServiceId = "webrtc_gateway",

  [Parameter(Mandatory = $false)]
  [string]$NatsUrl = "nats://127.0.0.1:4222",

  [Parameter(Mandatory = $true)]
  [string]$Field,

  [Parameter(Mandatory = $true)]
  [string]$Value
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
  $jsonValue = $Value | ConvertFrom-Json
} catch {
  $jsonValue = $Value
}

$payload = @{
  nodeId = $ServiceId
  field  = $Field
  value  = $jsonValue
} | ConvertTo-Json -Compress

$subject = "svc.$ServiceId.set_state"

Write-Host "nats request --no-templates -s $NatsUrl $subject $payload"
nats request --no-templates -s $NatsUrl $subject $payload
