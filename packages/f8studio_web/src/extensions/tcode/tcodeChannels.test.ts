import { expect, test } from 'vitest';

import { mergeTCodeChannels, parseTCodeChannels } from './tcodeChannels';

test('parses TCode axes without treating timing suffixes as channels', () => {
  expect(parseTCodeChannels('L05000I500 R09999S300 V02500 A01000\n')).toEqual({
    L0: 5000, R0: 9999, V0: 2500, A0: 1000,
  });
  expect(mergeTCodeChannels({ L0: 5000, R0: 1000 }, 'R09999I500')).toEqual({ L0: 5000, R0: 9999 });
});
