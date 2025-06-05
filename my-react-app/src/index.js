import React from 'react';
import ReactDOM from 'react-dom/client'; // Use createRoot for React 18+
import App from './App';
import './index.css'; // Import Tailwind CSS

const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(<App />);