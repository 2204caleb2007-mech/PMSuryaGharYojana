/// <reference types="vite/client" />

/**
 * Global ambient type extensions.
 * This file is an ambient script (no imports/exports), so Window interface
 * augmentations are written directly — no `declare global { }` wrapper needed.
 */

// ArcGIS AMD require loader (injected via <script> in index.html)
interface Window {
  require: any;

  // Google Identity Services + Maps JavaScript API
  google?: {
    accounts: {
      id: {
        initialize: (config: {
          client_id: string;
          callback: (response: { credential: string }) => void;
          auto_select?: boolean;
          cancel_on_tap_outside?: boolean;
        }) => void;
        prompt: () => void;
        renderButton: (element: HTMLElement, options: object) => void;
        disableAutoSelect: () => void;
      };
    };
    maps?: {
      importLibrary: (name: string) => Promise<any>;
      [key: string]: any;
    };
  };
}
