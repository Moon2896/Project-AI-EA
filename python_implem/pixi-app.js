// Import Pixi.js
// import * as PIXI from './pixi.js';

// Create a Pixi.js application
const app = new PIXI.Application({ width: 800, height: 600 });

// Add the Pixi.js canvas to the HTML container
document.getElementById('pixi-container').appendChild(app.view);

// Your Pixi.js visualization code goes here
//Step 3: Display Python Algorithm Results
//In your pixi-app.js file, you can fetch data from your Python program (located in the python directory) and use Pixi.js to display it in the canvas. You can use techniques like AJAX requests or WebSocket communication to transfer data from your Python code to your Pixi.js application.
document.body.appendChild(app.view);

const container = new PIXI.Container();

app.stage.addChild(container);

// Fetch CSV data from the Express.js server
fetch('/data')
  .then((response) => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.text();
  })
  .then((csvData) => {
    // Process the CSV data and use it in your Pixi.js visualization
    console.log(csvData);
    // You can parse and use the CSV data as needed
  })
  .catch((error) => {
    console.error('Fetch error:', error);
  });