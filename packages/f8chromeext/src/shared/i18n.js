export const getBrowserLanguage = () => {
  if (typeof chrome !== 'undefined' && chrome?.i18n?.getUILanguage) {
    return chrome.i18n.getUILanguage();
  }
  if (typeof navigator !== 'undefined' && navigator.language) {
    return navigator.language;
  }
  return 'en';
};

export const t = (key, substitutions) => {
  const normalizedSubs =
    substitutions == null
      ? undefined
      : Array.isArray(substitutions) || typeof substitutions === 'string'
        ? substitutions
        : [String(substitutions)];
  if (typeof chrome !== 'undefined' && chrome?.i18n?.getMessage) {
    const message = chrome.i18n.getMessage(key, normalizedSubs);
    if (message) {
      return message;
    }
  }
  if (normalizedSubs === undefined) {
    return key;
  }
  if (Array.isArray(normalizedSubs)) {
    return `${key}: ${normalizedSubs.join(' ')}`;
  }
  return `${key}: ${normalizedSubs}`;
};
