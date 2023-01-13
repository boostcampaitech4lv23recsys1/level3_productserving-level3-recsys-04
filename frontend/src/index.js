import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import SignIn from './pages/signin.js';
import MainPage from './pages/rec.js';
import HowUse from './pages/howuse.js';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

const rootElement = document.getElementById('root');
const root = createRoot(rootElement);

function Page() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={'/signin'} element={<SignIn/>} />
        <Route path={'/howuse'} element={<HowUse/>} />
        <Route path={'/main'} element={<MainPage/>} />
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
