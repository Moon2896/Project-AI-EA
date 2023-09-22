const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const fs = require('fs');
const cors = require('cors');
const app = express();
const port = 3000;
app.use(cors());

// Configure the proxy middleware to forward requests to the Python server
// const pythonServerProxy = createProxyMiddleware({
//     target: 'http://localhost:8000', // Specify the URL of the Python server
//     changeOrigin: true, // Set this to true to change the origin to match the target server
//   });

// app.use('/data', pythonServerProxy);


app.get('/data', (req, res) => {
  // Read the CSV file and send it as a response
  fs.readFile('output.csv', 'utf8', (err, data) => {
    if (err) {
      console.error(err);
      res.status(500).send('Error reading CSV file');
    } else {
      res.send(data);
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});