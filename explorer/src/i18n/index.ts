import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

import { en } from './en';
import { zhCN } from './zh-CN';

export const LANGUAGE_STORAGE_KEY = 'semantica.explorer.language';

export const SUPPORTED_LANGUAGES = ['en', 'zh-CN'] as const;
export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

function readStoredLanguage(): SupportedLanguage | null {
  try {
    const stored = window.localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return stored === 'en' || stored === 'zh-CN' ? stored : null;
  } catch {
    return null;
  }
}

function detectLanguage(): SupportedLanguage {
  const stored = readStoredLanguage();
  if (stored) return stored;
  return navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en';
}

export function setLanguage(language: SupportedLanguage) {
  try {
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // Storage can be unavailable (private windows, blocked site data); the choice then lasts for this page only.
  }
  void i18n.changeLanguage(language);
}

void i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    'zh-CN': { translation: zhCN },
  },
  lng: detectLanguage(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
});

export default i18n;
