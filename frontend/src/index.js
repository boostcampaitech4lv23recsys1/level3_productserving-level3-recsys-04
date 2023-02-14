import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import SignIn from './pages/signin.js';
import Album from './pages/album.js';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

const rootElement = document.getElementById('root');
const root = createRoot(rootElement);

function Page() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={'/signin'} element={<SignIn />} />
        <Route path={'/album'} element={<Album />} />
        <Route path={'/*'} element={<Navigate replace to={'/signin'} />} />
      </Routes>
    </BrowserRouter>
  );
}

root.render(
  <StrictMode>
    <Page />
  </StrictMode>
);
