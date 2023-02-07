const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'http://118.67.132.25:30001',
      changeOrigin: true,
    })
  );
};