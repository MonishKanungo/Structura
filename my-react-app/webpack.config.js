const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  // Set the mode to development or production
  mode: 'development', // or 'production' for optimized build
  // Entry point of your application
  entry: './src/index.js',
  // Output configuration
  output: {
    path: path.resolve(__dirname, 'dist'), // Output to a 'dist' folder
    filename: 'bundle.js', // Name of the bundled JavaScript file
    publicPath: '/', // Base path for all assets
  },
  // Module rules for handling different file types
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/, // Apply this rule to .js and .jsx files
        exclude: /node_modules/, // Don't process files in node_modules
        use: {
          loader: 'babel-loader', // Use babel-loader
          options: {
            presets: ['@babel/preset-env', '@babel/preset-react'], // Use these Babel presets
          },
        },
      },
      {
        test: /\.css$/, // Rule for CSS files
        use: ['style-loader', 'css-loader', 'postcss-loader'], // Use style-loader, css-loader, and postcss-loader
      },
    ],
  },
  // Resolve extensions for imports
  resolve: {
    extensions: ['.js', '.jsx'],
  },
  // Plugins used by Webpack
  plugins: [
    new HtmlWebpackPlugin({
      template: './public/index.html', // Use this HTML file as a template
      filename: 'index.html', // Output HTML file name
    }),
  ],
  // Development server configuration
  devServer: {
    static: {
      directory: path.join(__dirname, 'public'), // Serve static files from 'public'
    },
    compress: true, // Enable gzip compression
    port: 3000, // Port to run the dev server on
    historyApiFallback: true, // Fallback to index.html for HTML5 History API based routing
  },
};