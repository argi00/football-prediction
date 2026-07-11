import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// En développement (`npm run dev`), le frontend tourne sur son propre port
// (5173 par défaut). On proxy les appels /api/* vers le backend FastAPI
// lancé séparément (`uvicorn main:app --port 8000`), pour que le code du
// frontend puisse toujours appeler des chemins relatifs comme fetch('/api/teams')
// sans se soucier du port, aussi bien en dev qu'une fois déployé.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
