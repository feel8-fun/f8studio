import { CircleStop, Play, Radio, Volume2, VolumeX } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';

import { closeAudioSession, createAudioSession } from '../api/client';
import { createRtcPeerConnection, waitForIceGatheringComplete } from './rtc';

type AudioStatus =
  | { readonly kind: 'idle' }
  | { readonly kind: 'connecting' }
  | { readonly kind: 'playing'; readonly sampleRate: number; readonly channels: number }
  | { readonly kind: 'error'; readonly message: string };

interface AudioGraph {
  readonly context: AudioContext;
  readonly gain: GainNode;
  readonly analyser: AnalyserNode;
  readonly animationFrame: number;
}

interface NegotiatedAudio {
  readonly sampleRate: number;
  readonly channels: number;
}

function drawAudio(canvas: HTMLCanvasElement, analyser: AnalyserNode, mode: 'waveform' | 'spectrum'): void {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('2D canvas context is unavailable');
  const samples = new Uint8Array(mode === 'waveform' ? analyser.fftSize : analyser.frequencyBinCount);
  if (mode === 'waveform') analyser.getByteTimeDomainData(samples);
  else analyser.getByteFrequencyData(samples);
  context.fillStyle = '#080a0c';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = mode === 'waveform' ? '#65c99e' : '#62a9e8';
  context.lineWidth = 2;
  context.beginPath();
  for (let index = 0; index < samples.length; index += 1) {
    const x = index * width / Math.max(1, samples.length - 1);
    const normalized = (samples[index] ?? (mode === 'waveform' ? 128 : 0)) / 255;
    const y = mode === 'waveform' ? normalized * height : height - normalized * height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
}

export function WebRtcAudio() {
  const [source, setSource] = useState('synthetic://tone');
  const [status, setStatus] = useState<AudioStatus>({ kind: 'idle' });
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(0.5);
  const [visualization, setVisualization] = useState<'waveform' | 'spectrum'>('waveform');
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const sessionRef = useRef<string | null>(null);
  const negotiatedRef = useRef<NegotiatedAudio | null>(null);
  const graphRef = useRef<AudioGraph | null>(null);
  const operationRef = useRef(0);
  const visualizationRef = useRef<'waveform' | 'spectrum'>('waveform');

  useEffect(() => { visualizationRef.current = visualization; }, [visualization]);

  const disposeGraph = useCallback(() => {
    const graph = graphRef.current;
    graphRef.current = null;
    if (graph !== null) {
      cancelAnimationFrame(graph.animationFrame);
      void graph.context.close().catch((error: unknown) => console.error('Failed to close audio context', error));
    }
  }, []);

  const stop = useCallback(async () => {
    operationRef.current += 1;
    const peer = peerRef.current;
    peerRef.current = null;
    peer?.close();
    const sessionId = sessionRef.current;
    sessionRef.current = null;
    negotiatedRef.current = null;
    const audio = audioRef.current;
    if (audio !== null) {
      audio.pause();
      audio.srcObject = null;
    }
    disposeGraph();
    try {
      if (sessionId !== null) await closeAudioSession(sessionId);
      setStatus({ kind: 'idle' });
    } catch (error: unknown) {
      setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'Audio session close failed' });
    }
  }, [disposeGraph]);

  useEffect(() => () => {
    operationRef.current += 1;
    const peer = peerRef.current;
    peerRef.current = null;
    peer?.close();
    const sessionId = sessionRef.current;
    sessionRef.current = null;
    negotiatedRef.current = null;
    if (sessionId !== null) {
      void closeAudioSession(sessionId).catch((error: unknown) => console.error('Failed to release audio session', error));
    }
    disposeGraph();
  }, [disposeGraph]);

  useEffect(() => {
    const graph = graphRef.current;
    if (graph !== null) graph.gain.gain.value = muted ? 0 : volume;
  }, [muted, volume]);

  const start = useCallback(async () => {
    const operation = operationRef.current + 1;
    operationRef.current = operation;
    const normalizedSource = source.trim();
    if (normalizedSource.length === 0) {
      setStatus({ kind: 'error', message: 'Enter an audio source' });
      return;
    }
    setStatus({ kind: 'connecting' });
    let peer: RTCPeerConnection;
    try {
      peer = await createRtcPeerConnection();
      if (operationRef.current !== operation) {
        peer.close();
        return;
      }
    } catch (error: unknown) {
      setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'RTC configuration failed' });
      return;
    }
    let createdSessionId: string | null = null;
    peerRef.current = peer;
    peer.addTransceiver('audio', { direction: 'recvonly' });
    peer.onconnectionstatechange = () => {
      if (peerRef.current !== peer) return;
      if (peer.connectionState === 'connected') {
        const negotiated = negotiatedRef.current;
        if (negotiated === null) return;
        setStatus((current) => current.kind === 'connecting' ? {
          kind: 'playing',
          sampleRate: negotiated.sampleRate,
          channels: negotiated.channels,
        } : current);
      } else if (peer.connectionState === 'failed') {
        peerRef.current = null;
        peer.close();
        const failedSessionId = sessionRef.current;
        sessionRef.current = null;
        negotiatedRef.current = null;
        if (failedSessionId !== null) {
          void closeAudioSession(failedSessionId).catch((error: unknown) => {
            console.error('Failed to release failed audio session', error);
          });
        }
        setStatus({
          kind: 'error',
          message: `WebRTC connection failed (ICE: ${peer.iceConnectionState}). Check UDP, VPN, or TURN connectivity.`,
        });
      }
    };
    peer.ontrack = (event) => {
      if (peerRef.current !== peer) return;
      const audio = audioRef.current;
      const canvas = canvasRef.current;
      if (audio === null || canvas === null) return;
      const stream = event.streams[0] ?? new MediaStream([event.track]);
      audio.srcObject = stream;
      const context = new AudioContext({ sampleRate: 48_000 });
      const input = context.createMediaStreamSource(stream);
      const gain = context.createGain();
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.35;
      gain.gain.value = muted ? 0 : volume;
      input.connect(analyser);
      analyser.connect(gain);
      gain.connect(context.destination);
      let animationFrame = 0;
      const render = () => {
        drawAudio(canvas, analyser, visualizationRef.current);
        animationFrame = requestAnimationFrame(render);
        const current = graphRef.current;
        if (current !== null && current.context === context) {
          graphRef.current = { ...current, animationFrame };
        }
      };
      animationFrame = requestAnimationFrame(render);
      graphRef.current = { context, gain, analyser, animationFrame };
      void context.resume();
    };
    try {
      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      await waitForIceGatheringComplete(peer);
      if (operationRef.current !== operation || peerRef.current !== peer) {
        peer.close();
        return;
      }
      if (peer.localDescription === null) throw new Error('Browser did not create a local description');
      const answer = await createAudioSession(normalizedSource, peer.localDescription);
      createdSessionId = answer.sessionId;
      if (operationRef.current !== operation || peerRef.current !== peer) {
        peer.close();
        await closeAudioSession(createdSessionId);
        return;
      }
      sessionRef.current = answer.sessionId;
      negotiatedRef.current = { sampleRate: answer.sampleRate, channels: answer.channels };
      await peer.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    } catch (error: unknown) {
      peer.close();
      if (peerRef.current === peer) {
        peerRef.current = null;
        sessionRef.current = null;
        negotiatedRef.current = null;
      }
      if (createdSessionId !== null) await closeAudioSession(createdSessionId).catch((closeError: unknown) => {
        console.error('Failed to release audio session after negotiation failure', closeError);
      });
      if (operationRef.current === operation) {
        setStatus({ kind: 'error', message: error instanceof Error ? error.message : 'Audio negotiation failed' });
      }
    }
  }, [muted, source, volume]);

  const active = status.kind === 'connecting' || status.kind === 'playing';
  return (
    <section className="audio-workspace" aria-label="WebRTC audio">
      <div className="media-toolbar audio-toolbar">
        <label className="source-field">
          <Radio size={15} aria-hidden="true" />
          <input value={source} onChange={(event) => setSource(event.target.value)} disabled={active} aria-label="Audio source" />
        </label>
        <label className="volume-control">
          <Volume2 size={16} aria-hidden="true" />
          <input type="range" min="0" max="1" step="0.01" value={volume} onChange={(event) => setVolume(Number(event.target.value))} aria-label="Volume" />
        </label>
        <button className="icon-button audio-mute" type="button" aria-label={muted ? 'Unmute' : 'Mute'} title={muted ? 'Unmute' : 'Mute'} onClick={() => setMuted((value) => !value)}>
          {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
        </button>
        {active ? (
          <button className="command-button" type="button" onClick={() => void stop()}><CircleStop size={16} />Stop</button>
        ) : (
          <button className="command-button primary" type="button" onClick={() => void start()}><Play size={16} />Play</button>
        )}
      </div>
      <div className="audio-stage">
        <div className="audio-visualization-mode segment" role="tablist" aria-label="Audio visualization">
          <button role="tab" aria-selected={visualization === 'waveform'} className={visualization === 'waveform' ? 'selected' : ''} onClick={() => setVisualization('waveform')}>Wave</button>
          <button role="tab" aria-selected={visualization === 'spectrum'} className={visualization === 'spectrum' ? 'selected' : ''} onClick={() => setVisualization('spectrum')}>Spectrum</button>
        </div>
        <audio ref={audioRef} playsInline />
        <canvas ref={canvasRef} data-testid="audio-waveform" width={1200} height={360} />
        <div className={`media-status status-${status.kind}`} role="status">
          {status.kind === 'idle' && 'Ready'}
          {status.kind === 'connecting' && 'Negotiating'}
          {status.kind === 'playing' && `${status.sampleRate} Hz · ${status.channels === 1 ? 'Mono' : status.channels === 2 ? 'Stereo' : 'Source channels'}`}
          {status.kind === 'error' && status.message}
        </div>
      </div>
    </section>
  );
}
