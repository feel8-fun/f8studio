import { ChevronDown, Search } from 'lucide-react';
import { useMemo, useState, type ReactNode } from 'react';

import type { CatalogSnapshot, OperatorSpec, ServiceSpec } from '../api/contracts';

type CatalogSpec = ServiceSpec | OperatorSpec;
type GroupMode = 'service' | 'category';

interface CatalogGroup {
  readonly key: string;
  readonly label: string;
  readonly specs: readonly OperatorSpec[];
}

const CATEGORY_LABELS: Readonly<Record<string, string>> = {
  canvas: 'Canvas', control: 'Controls', debug: 'Debug', execution: 'Execution',
  expr: 'Expressions', flow: 'Flow', input: 'Inputs', motion: 'Motion',
  output: 'Outputs', routing: 'Routing', signal: 'Signal', viz: 'Visualization',
};

function categoryLabel(category: string): string {
  const suffix = category.split('.').at(-1) ?? category;
  return CATEGORY_LABELS[suffix] ?? suffix.replaceAll('_', ' ');
}

function operatorGroups(operators: readonly OperatorSpec[], services: readonly ServiceSpec[], mode: GroupMode): readonly CatalogGroup[] {
  const serviceLabels = new Map(services.map((service) => [service.serviceClass, service.label]));
  const groups = new Map<string, OperatorSpec[]>();
  for (const operator of operators) {
    const key = mode === 'service' ? operator.serviceClass : (operator.paletteCategory || 'other').split('.').at(-1) ?? 'other';
    const members = groups.get(key) ?? [];
    members.push(operator);
    groups.set(key, members);
  }
  return [...groups.entries()].map(([key, specs]) => ({
    key,
    label: mode === 'service' ? serviceLabels.get(key) ?? key : categoryLabel(key),
    specs: [...specs].sort((left, right) => left.label.localeCompare(right.label)),
  })).sort((left, right) => left.label.localeCompare(right.label));
}

function CatalogFold({ label, count, initiallyOpen, children }: {
  readonly label: string;
  readonly count: number;
  readonly initiallyOpen: boolean;
  readonly children: ReactNode;
}) {
  const [open, setOpen] = useState(initiallyOpen);
  return <details className="catalog-group" open={open}>
    <summary onClick={(event) => { event.preventDefault(); setOpen((current) => !current); }}>
      <ChevronDown size={13} /><span>{label}</span><small>{count}</small>
    </summary>
    {children}
  </details>;
}

export function NodeCatalog({ catalog, projectServiceClasses, canAdd, onAdd }: {
  readonly catalog: CatalogSnapshot | null;
  readonly projectServiceClasses: ReadonlySet<string>;
  readonly canAdd: boolean;
  readonly onAdd: (spec: CatalogSpec) => void;
}) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<GroupMode>('service');
  const normalizedQuery = query.trim().toLowerCase();
  const serviceLabels = useMemo(() => new Map((catalog?.services ?? []).map((service) => [service.serviceClass, service.label])), [catalog]);
  const services = useMemo(() => (catalog?.services ?? []).filter((spec) => !spec.hiddenInPalette &&
    `${spec.label} ${spec.serviceClass} ${(spec.tags ?? []).join(' ')}`.toLowerCase().includes(normalizedQuery)), [catalog, normalizedQuery]);
  const operators = useMemo(() => (catalog?.operators ?? []).filter((spec) => !spec.hiddenInPalette &&
    `${spec.label} ${spec.operatorClass} ${spec.serviceClass} ${serviceLabels.get(spec.serviceClass) ?? ''} ${spec.paletteCategory ?? ''} ${(spec.tags ?? []).join(' ')}`.toLowerCase().includes(normalizedQuery)), [catalog, normalizedQuery, serviceLabels]);
  const groups = useMemo(() => operatorGroups(operators, catalog?.services ?? [], mode), [operators, catalog, mode]);

  const operatorButton = (spec: OperatorSpec) => {
    const available = projectServiceClasses.has(spec.serviceClass) || spec.serviceClass === 'f8.pystudio';
    return <button key={`${spec.serviceClass}:${spec.operatorClass}`} type="button" disabled={!canAdd || !available}
      title={available ? spec.description : `Requires ${spec.serviceClass}`} onClick={() => onAdd(spec)}>
      <strong>{spec.label}</strong><span>{spec.operatorClass}</span>
    </button>;
  };

  return <>
    <label className="catalog-search"><Search size={15} />
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search nodes" aria-label="Search nodes" />
    </label>
    <div className="catalog-mode segment" role="tablist" aria-label="Group operators by">
      <button type="button" role="tab" aria-selected={mode === 'service'} className={mode === 'service' ? 'selected' : ''} onClick={() => setMode('service')}>Service</button>
      <button type="button" role="tab" aria-selected={mode === 'category'} className={mode === 'category' ? 'selected' : ''} onClick={() => setMode('category')}>Category</button>
    </div>
    <div className="catalog-list">
      <CatalogFold label="Services" count={services.length} initiallyOpen key={`services:${normalizedQuery !== ''}`}>
        {services.map((spec) => <button key={spec.serviceClass} type="button" disabled={!canAdd} onClick={() => onAdd(spec)}>
          <strong>{spec.label}</strong><span>{spec.serviceClass}</span>
        </button>)}
      </CatalogFold>
      <h2>Operators</h2>
      {groups.map((group) => <CatalogFold label={group.label} count={group.specs.length} key={`${mode}:${group.key}:${normalizedQuery !== ''}`}
        initiallyOpen={normalizedQuery !== ''}>
        {group.specs.map(operatorButton)}
      </CatalogFold>)}
      {groups.length === 0 && <p className="catalog-empty">No matching operators</p>}
    </div>
  </>;
}
