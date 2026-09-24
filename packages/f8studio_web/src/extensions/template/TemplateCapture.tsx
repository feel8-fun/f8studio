import { Check, Crop, ImageDown } from 'lucide-react';
import { useCallback, useRef, useState } from 'react';

import { invokeRuntimeCommand, setRuntimeState } from '../../api/client';
import type { JsonValue } from '../../api/contracts';

interface Point { readonly x: number; readonly y: number }
interface Selection { readonly start: Point; readonly end: Point }

function objectValue(value: JsonValue | undefined): Readonly<Record<string, JsonValue>> | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  return value as Readonly<Record<string, JsonValue>>;
}

export function TemplateCapture() {
  const [serviceId, setServiceId] = useState('template_match');
  const [imageUrl, setImageUrl] = useState('');
  const [selection, setSelection] = useState<Selection | null>(null);
  const [status, setStatus] = useState('Ready');
  const imageRef = useRef<HTMLImageElement>(null);
  const dragRef = useRef<Point | null>(null);

  const capture = useCallback(async () => {
    setStatus('Capturing');
    try {
      const response = await invokeRuntimeCommand(serviceId.trim(), 'captureTemplateFrame', {
        format: 'jpeg', quality: 90, maxBytes: 4_000_000, maxWidth: 1920, maxHeight: 1080,
      });
      const direct = objectValue(response);
      const result = objectValue(direct?.result) ?? direct;
      const image = objectValue(result?.image);
      const b64 = typeof image?.b64 === 'string' ? image.b64 : '';
      const format = typeof image?.format === 'string' ? image.format : 'jpeg';
      if (!b64) throw new Error('Capture response does not include image.b64');
      setImageUrl(`data:image/${format};base64,${b64}`);
      setSelection(null);
      setStatus(`Captured frame ${String(result?.frameId ?? '')}`);
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Capture failed');
    }
  }, [serviceId]);

  const pointFromEvent = (event: React.PointerEvent<HTMLDivElement>): Point => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const imageBounds = imageRef.current?.getBoundingClientRect();
    const minX = imageBounds === undefined ? 0 : imageBounds.left - bounds.left;
    const minY = imageBounds === undefined ? 0 : imageBounds.top - bounds.top;
    const maxX = imageBounds === undefined ? bounds.width : imageBounds.right - bounds.left;
    const maxY = imageBounds === undefined ? bounds.height : imageBounds.bottom - bounds.top;
    return {
      x: Math.max(minX, Math.min(maxX, event.clientX - bounds.left)),
      y: Math.max(minY, Math.min(maxY, event.clientY - bounds.top)),
    };
  };

  const apply = useCallback(async () => {
    const image = imageRef.current;
    if (image === null || selection === null) return;
    const bounds = image.getBoundingClientRect();
    const stage = image.parentElement?.getBoundingClientRect();
    if (stage === undefined) return;
    const left = Math.min(selection.start.x, selection.end.x);
    const top = Math.min(selection.start.y, selection.end.y);
    const width = Math.abs(selection.end.x - selection.start.x);
    const height = Math.abs(selection.end.y - selection.start.y);
    if (width < 2 || height < 2) { setStatus('Select a larger region'); return; }
    const scaleX = image.naturalWidth / bounds.width;
    const scaleY = image.naturalHeight / bounds.height;
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width * scaleX));
    canvas.height = Math.max(1, Math.round(height * scaleY));
    const context = canvas.getContext('2d');
    if (context === null) { setStatus('Canvas is unavailable'); return; }
    const imageX = left - (bounds.left - stage.left);
    const imageY = top - (bounds.top - stage.top);
    context.drawImage(image, imageX * scaleX, imageY * scaleY, width * scaleX, height * scaleY, 0, 0, canvas.width, canvas.height);
    const b64 = canvas.toDataURL('image/png').split(',')[1] ?? '';
    try {
      await setRuntimeState(serviceId.trim(), serviceId.trim(), 'templateImagePngB64', b64);
      setStatus(`Applied ${canvas.width} x ${canvas.height} template`);
    } catch (error: unknown) {
      setStatus(error instanceof Error ? error.message : 'Apply failed');
    }
  }, [selection, serviceId]);

  const rect = selection === null ? null : {
    left: Math.min(selection.start.x, selection.end.x), top: Math.min(selection.start.y, selection.end.y),
    width: Math.abs(selection.end.x - selection.start.x), height: Math.abs(selection.end.y - selection.start.y),
  };

  return <div className="template-workspace">
    <div className="template-toolbar">
      <label className="source-field"><Crop size={15} /><input value={serviceId} onChange={(event) => setServiceId(event.target.value)} aria-label="Template match service id" /></label>
      <button className="command-button" type="button" onClick={() => void capture()}><ImageDown size={15} />Capture</button>
      <button className="command-button primary" type="button" disabled={rect === null} onClick={() => void apply()}><Check size={15} />Apply ROI</button>
      <span className="tool-status" role="status">{status}</span>
    </div>
    <div
      className="template-stage"
      onPointerDown={(event) => { const point = pointFromEvent(event); dragRef.current = point; setSelection({ start: point, end: point }); event.currentTarget.setPointerCapture(event.pointerId); }}
      onPointerMove={(event) => { const start = dragRef.current; if (start !== null) setSelection({ start, end: pointFromEvent(event) }); }}
      onPointerUp={() => { dragRef.current = null; }}
    >
      {imageUrl ? <img ref={imageRef} src={imageUrl} alt="Captured template source" draggable={false} /> : <div className="empty-state centered">Capture a frame, then drag a region</div>}
      {rect !== null && <div className="roi-selection" style={rect} />}
    </div>
  </div>;
}
