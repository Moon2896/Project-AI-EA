const express = require('express');
const readline = require('readline');
const fs = require('fs');

const app = express();
const port = 3000;

const cors = require('cors');
app.use(cors());

app.get('/data', (req, res) => {
  // Specify the path to your CSV file
  const csvFilePath = './python_implem/output.csv';

  // Create a readable stream for reading the CSV file
  const fileStream = fs.createReadStream(csvFilePath);

  // Create a readline interface
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity, // To handle line endings on different platforms
  });

  let lastLine = null;

  // Read the file line by line
  rl.on('line', (line) => {
    lastLine = line; // Store the current line as the last line
  });

  // When the entire file has been read
  rl.on('close', () => {
    if (lastLine !== null) {
      // Send the last line as the response
      res.send(lastLine);
    } else {
      console.log('The file is empty.');
      res.status(404).send('CSV file is empty');
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
