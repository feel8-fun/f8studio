declare module 'osr-emu' {
  export type OSRModel = 'OSR2' | 'SR6' | 'SSR1';

  export class OSREmulator {
    constructor(target: HTMLElement | string, options?: { readonly model?: OSRModel });
    write(line: string): void;
    destroy(): void;
  }
}
