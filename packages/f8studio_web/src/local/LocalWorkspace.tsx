import { CheckCircle2, Gamepad2, HardDrive, Radio, RefreshCw, Usb, XCircle } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import {
  applyUnityInstall,
  detectModdingTarget,
  fetchLocalCapabilities,
  fetchSerialPorts,
  previewUnityInstall,
  verifySkeletonUdp,
} from '../api/client';
import type { JsonValue, LocalCapability, SerialPortInfo, SkeletonUdpVerification, UnityInstallPlan } from '../api/contracts';

export function LocalWorkspace() {
  const [capabilities, setCapabilities] = useState<readonly LocalCapability[]>([]);
  const [serialPorts, setSerialPorts] = useState<readonly SerialPortInfo[]>([]);
  const [targetPath, setTargetPath] = useState('');
  const [detection, setDetection] = useState<Readonly<Record<string, JsonValue>> | null>(null);
  const [plan, setPlan] = useState<UnityInstallPlan | null>(null);
  const [udpPort, setUdpPort] = useState(39540);
  const [verification, setVerification] = useState<SkeletonUdpVerification | null>(null);
  const [status, setStatus] = useState('Ready');

  useEffect(() => {
    const controller = new AbortController();
    void fetchLocalCapabilities(controller.signal).then(setCapabilities, (error: unknown) => setStatus(error instanceof Error ? error.message : 'Capability load failed'));
    return () => controller.abort();
  }, []);

  const detect = useCallback(async () => {
    try { setDetection(await detectModdingTarget(targetPath)); setPlan(null); setStatus('Target inspected'); }
    catch (error: unknown) { setStatus(error instanceof Error ? error.message : 'Detection failed'); }
  }, [targetPath]);

  const preview = useCallback(async () => {
    try { setPlan(await previewUnityInstall(targetPath)); setStatus('Review every write before applying'); }
    catch (error: unknown) { setStatus(error instanceof Error ? error.message : 'Preview failed'); }
  }, [targetPath]);

  const apply = useCallback(async () => {
    if (plan === null) return;
    try { await applyUnityInstall(plan.planId, true); setStatus('Unity installation completed'); setPlan(null); }
    catch (error: unknown) { setStatus(error instanceof Error ? error.message : 'Install failed'); }
  }, [plan]);

  return <section className="local-workspace" aria-label="Local integrations">
    <div className="capability-band">
      {capabilities.map((item) => <div className={`capability capability-${item.status}`} key={item.capability} title={item.reason}>
        {item.status === 'available' ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
        <span><strong>{item.capability.replaceAll('_', ' ')}</strong><small>{item.backend} · {item.status}</small></span>
      </div>)}
    </div>
    <div className="local-columns">
      <section className="local-panel">
        <header><Gamepad2 size={17} /><h2>Game target</h2></header>
        <label className="field-stack">Executable or game directory<input className="plain-input" value={targetPath} onChange={(event) => setTargetPath(event.target.value)} /></label>
        <div className="button-row"><button className="command-button" type="button" onClick={() => void detect()}>Detect</button><button className="command-button" type="button" disabled={detection?.engine !== 'unity'} onClick={() => void preview()}>Preview install</button></div>
        {detection !== null && <pre className="data-preview">{JSON.stringify(detection, null, 2)}</pre>}
        {plan !== null && <div className="install-plan"><h3>Planned writes</h3>{plan.actions.map((action) => <div key={action}>{action}</div>)}{plan.filesToWrite.map((path) => <code key={path}>{path}</code>)}{plan.blockingErrors.map((error) => <strong key={error}>{error}</strong>)}<button className="command-button primary" type="button" disabled={plan.blockingErrors.length > 0} onClick={() => void apply()}>Confirm and apply</button></div>}
      </section>
      <section className="local-panel">
        <header><Radio size={17} /><h2>Skeleton UDP</h2></header>
        <label className="field-stack">Listen port<input className="plain-input" type="number" min={1} max={65535} value={udpPort} onChange={(event) => setUdpPort(Number(event.target.value))} /></label>
        <button className="command-button" type="button" onClick={() => { setStatus('Listening for decoded frames'); void verifySkeletonUdp(udpPort).then((result) => { setVerification(result); setStatus(result.verified ? 'Skeleton stream verified' : 'No complete frame decoded'); }, (error: unknown) => setStatus(error instanceof Error ? error.message : 'UDP verification failed')); }}><Radio size={14} />Verify stream</button>
        {verification !== null && <div className={`verification ${verification.verified ? 'verified' : ''}`}><strong>{verification.decodedFrameCount} decoded / {verification.packetCount} packets</strong><span>{verification.modelNames.join(', ') || 'No models'}</span>{verification.decoderErrors.map((error) => <code key={error}>{error}</code>)}</div>}
      </section>
      <section className="local-panel">
        <header><Usb size={17} /><h2>Serial devices</h2><button className="icon-button bordered" type="button" aria-label="Refresh serial ports" title="Refresh serial ports" onClick={() => void fetchSerialPorts().then(setSerialPorts, (error: unknown) => setStatus(error instanceof Error ? error.message : 'Serial scan failed'))}><RefreshCw size={15} /></button></header>
        <div className="device-list">{serialPorts.map((port) => <div key={port.device}><HardDrive size={15} /><span><strong>{port.device}</strong><small>{port.description || port.hardwareId}</small></span></div>)}{serialPorts.length === 0 && <div className="empty-state">No scan results</div>}</div>
      </section>
    </div>
    <div className="local-status" role="status">{status}</div>
  </section>;
}
