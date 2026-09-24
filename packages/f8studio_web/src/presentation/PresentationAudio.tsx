import { RefreshCw, Volume2, VolumeX } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import type { JsonValue } from '../api/contracts';
import { audioSessionPool, type AudioSessionLease, type AudioSessionSnapshot } from '../media/AudioSessionPool';

const CONNECTING: AudioSessionSnapshot = { kind: 'connecting', stream: null };

function useAudioSession(source: string): readonly [AudioSessionSnapshot, () => void] {
  const [snapshot, setSnapshot] = useState<AudioSessionSnapshot>(CONNECTING);
  const leaseRef = useRef<AudioSessionLease | null>(null);

  useEffect(() => {
    setSnapshot(CONNECTING);
    if (source === '') return;
    const lease = audioSessionPool.acquire(source);
    leaseRef.current = lease;
    const update = () => setSnapshot(lease.getSnapshot());
    const unsubscribe = lease.subscribe(update);
    update();
    return () => {
      unsubscribe();
      lease.release();
      if (leaseRef.current === lease) leaseRef.current = null;
    };
  }, [source]);

  return [snapshot, () => leaseRef.current?.retry()];
}

function drawAudio(canvas: HTMLCanvasElement, analyser: AnalyserNode, mode: 'waveform' | 'spectrum'): boolean {
  const context = canvas.getContext('2d');
  if (context === null) return false;
  const samples = new Uint8Array(mode === 'waveform' ? analyser.fftSize : analyser.frequencyBinCount);
  if (mode === 'waveform') analyser.getByteTimeDomainData(samples);
  else analyser.getByteFrequencyData(samples);
  const hasSignal = samples.some((sample) => mode === 'waveform' ? Math.abs(sample - 128) > 3 : sample > 3);
  context.fillStyle = '#080a0c';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = mode === 'waveform' ? '#65c99e' : '#62a9e8';
  context.lineWidth = 2;
  context.beginPath();
  for (let index = 0; index < samples.length; index += 1) {
    const x = index * canvas.width / Math.max(1, samples.length - 1);
    const normalized = (samples[index] ?? (mode === 'waveform' ? 128 : 0)) / 255;
    const y = mode === 'waveform' ? normalized * canvas.height : canvas.height - normalized * canvas.height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();
  return hasSignal;
}

export function PresentationAudio({
  payload,
  compact = false,
}: {
  readonly payload: Readonly<Record<string, JsonValue>>;
  readonly compact?: boolean;
}) {
  const source = typeof payload.audioStreamKey === 'string' ? payload.audioStreamKey.trim() : '';
  const channel = typeof payload.channel === 'number' ? Math.max(0, Math.min(16, Math.floor(payload.channel))) : 0;
  const historyMs = typeof payload.historyMs === 'number' ? Math.max(20, Math.min(680, payload.historyMs)) : 250;
  const throttleMs = typeof payload.throttleMs === 'number' ? Math.max(0, Math.min(60000, payload.throttleMs)) : 20;
  const [snapshot, retry] = useAudioSession(source);
  const [mode, setMode] = useState<'waveform' | 'spectrum'>('waveform');
  const [listening, setListening] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [audioSuspended, setAudioSuspended] = useState(false);
  const [hasSignal, setHasSignal] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const modeRef = useRef(mode);
  const throttleRef = useRef(throttleMs);
  modeRef.current = mode;
  throttleRef.current = throttleMs;

  useEffect(() => {
    const audio = audioRef.current;
    if (audio === null) return;
    audio.srcObject = snapshot.stream;
    if (snapshot.stream !== null) {
      void audio.play().catch((error: unknown) => console.error('Failed to start Audio Viz media track', error));
    }
    return () => {
      audio.pause();
      if (audio.srcObject === snapshot.stream) audio.srcObject = null;
    };
  }, [snapshot.stream]);

  useEffect(() => {
    const stream = snapshot.stream;
    const canvas = canvasRef.current;
    if (stream === null || canvas === null) return;
    setHasSignal(false);
    let context: AudioContext | null = null;
    try {
      const activeContext = new AudioContext({ sampleRate: 48_000 });
      context = activeContext;
      const input = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      const gain = context.createGain();
      analyser.fftSize = Math.max(256, Math.min(32768, 2 ** Math.ceil(Math.log2(historyMs * 48))));
      analyser.smoothingTimeConstant = 0.35;
      gain.gain.value = 0;
      const splitter = context.createChannelSplitter(channel + 1);
      input.connect(splitter);
      splitter.connect(analyser, channel);
      analyser.connect(gain);
      gain.connect(context.destination);
      contextRef.current = context;
      gainRef.current = gain;
      setAudioError(null);
      activeContext.onstatechange = () => setAudioSuspended(activeContext.state === 'suspended');
      setAudioSuspended(context.state === 'suspended');
      let animationFrame = 0;
      let lastDraw = 0;
      let lastSignal = performance.now();
      let signalPresent = false;
      const render = (now: number) => {
        if (now - lastDraw >= throttleRef.current) {
          if (drawAudio(canvas, analyser, modeRef.current)) {
            lastSignal = now;
            if (!signalPresent) {
              signalPresent = true;
              setHasSignal(true);
            }
          } else if (signalPresent && now - lastSignal > 1500) {
            signalPresent = false;
            setHasSignal(false);
          }
          lastDraw = now;
        }
        animationFrame = requestAnimationFrame(render);
      };
      animationFrame = requestAnimationFrame(render);
      void context.resume().catch((error: unknown) => {
        setAudioError(error instanceof Error ? error.message : 'Audio visualization could not start');
      });
      return () => {
        cancelAnimationFrame(animationFrame);
        activeContext.onstatechange = null;
        gainRef.current = null;
        contextRef.current = null;
        void activeContext.close().catch((error: unknown) => console.error('Failed to close Audio Viz context', error));
      };
    } catch (error: unknown) {
      setAudioError(error instanceof Error ? error.message : 'Audio visualization is unavailable');
      void context?.close().catch((reason: unknown) => console.error('Failed to close Audio Viz context', reason));
      contextRef.current = null;
      gainRef.current = null;
    }
  }, [snapshot.stream, channel, historyMs]);

  useEffect(() => {
    const gain = gainRef.current;
    if (gain !== null) gain.gain.value = listening ? 0.65 : 0;
  }, [listening, snapshot.stream]);

  const resumeAudio = () => {
    void contextRef.current?.resume().catch((error: unknown) => {
      setAudioError(error instanceof Error ? error.message : 'Audio playback could not start');
    });
  };

  const effectiveKind = source === '' || audioError !== null ? 'error' : snapshot.kind;
  const errorMessage = source === ''
    ? 'Audio input is not connected'
    : audioError ?? (snapshot.kind === 'error' ? snapshot.message : '');

  return <div className={`presentation-audio ${compact ? 'presentation-audio-compact' : ''}`} data-audio-source={source}>
    <audio ref={audioRef} autoPlay muted playsInline />
    <canvas ref={canvasRef} width={640} height={200} data-testid="audio-viz-canvas" aria-label="Audio visualization" />
    <div className="presentation-audio-toolbar nodrag nowheel">
      <div className="segment" role="tablist" aria-label="Audio visualization">
        <button type="button" role="tab" aria-selected={mode === 'waveform'} className={mode === 'waveform' ? 'selected' : ''} onClick={() => { setMode('waveform'); resumeAudio(); }}>Wave</button>
        <button type="button" role="tab" aria-selected={mode === 'spectrum'} className={mode === 'spectrum' ? 'selected' : ''} onClick={() => { setMode('spectrum'); resumeAudio(); }}>Spectrum</button>
      </div>
      <button type="button" className="icon-button" aria-label={listening ? 'Mute browser audio' : 'Listen in browser'}
        title={listening ? 'Mute browser audio' : 'Listen in browser'} aria-pressed={listening}
        onClick={() => { resumeAudio(); setListening((current) => !current); }}>
        {listening ? <Volume2 size={15} /> : <VolumeX size={15} />}
      </button>
    </div>
    <div className={`presentation-audio-status status-${effectiveKind}`} role="status">
      {effectiveKind === 'connecting' && 'Connecting'}
      {effectiveKind === 'playing' && (audioSuspended ? 'Click Wave or Spectrum to start preview' : hasSignal ? 'Live' : 'No audio signal')}
      {effectiveKind === 'error' && <><span>{errorMessage}</span>{source !== '' && <button type="button" className="icon-button" aria-label="Reconnect audio" title="Reconnect audio" onClick={retry}><RefreshCw size={15} /></button>}</>}
    </div>
  </div>;
}
