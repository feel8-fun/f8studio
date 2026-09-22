import React from 'react';
import ReactDOM from 'react-dom/client';

import { App } from './app/App';
import './styles.css';

const root = document.getElementById('root');
if (root === null) {
  throw new Error('Missing #root application host');
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
