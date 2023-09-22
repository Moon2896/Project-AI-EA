// Import Pixi.js
// import * as PIXI from './pixi.js';

// Create a Pixi.js application
const app = new PIXI.Application({ width: 900, height: 900 });
document.body.appendChild(app.view);

const container = new PIXI.Container();
app.stage.addChild(container);


// Function to convert a fetch to json
async function convertFetched(inputString) {

  if (typeof inputString !== 'string') {
    console.error('Input is not a string.');
    return;
  }
  // Split the input string by commas
  const values = inputString.split(';');

  // Extract values and convert to appropriate types
  const iteration = parseInt(values[0]);
  const noi = parseInt(values[1]); // Number Of Individuals

  const bestIndividualString = values[2];
  const bestIndividual = bestIndividualString.slice(1, -1).split(',').map(Number);
  
  const allScoresString = values[3];
  const allScores = allScoresString.slice(1, -1).split(',').map(Number);
  
  const numSameIndividuals = parseInt(values[4]);
  const numSharedPatterns = parseInt(values[5]);
  const score = parseFloat(values[5+1]);
  const alpha = parseFloat(values[6+1]);
  const beta = parseFloat(values[7+1]);
  const gamma = parseFloat(values[8+1]);
  const median = parseFloat(values[9+1]);
  const q1 = parseFloat(values[10+1]);
  const q3 = parseFloat(values[11+1]);
  const max = parseFloat(values[12+1]);

  // Create the JSON object
  const jsonObject = {
    Iteration: iteration,
    'Best Individual': bestIndividual,
    'Number of Same Individuals': numSameIndividuals,
    'Number of Shared Patterns': numSharedPatterns,
    Score: score,
    'All Scores': allScores,
    alpha: alpha,
    beta: beta,
    gamma: gamma,
    Median: median,
    Q1: q1,
    Q3: q3,
    Max: max,
  };
  return jsonObject;
};

// Function to fetch data from the server
async function fetchData() {
  try {
    const response = await fetch('http://localhost:3000/data'); // Assuming your server is serving data at '/data'
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.text();
    return data;
  } catch (error) {
    console.error('Fetch error:', error);
    // text.text = 'Error fetching data';
  }
}

// Function to update the displayed values
async function updateDisplayedValues(data) {
  try {
     document.getElementById('iteration').textContent = `Iteration: ${data.Iteration}`;
     document.getElementById('best-score').textContent = `Best Score: ${data.Score}`;
     document.getElementById('shared-patterns').textContent = `Number of Shared Patterns: ${data['Number of Shared Patterns']}`;
  } catch (error) {
     console.error('Error fetching data:', error);
  }
}

// Use setInterval to periodically update displayed values
const updateInterval = 1000; // Update every 1 second (adjust as needed)
setInterval(() => {
  // updateDisplayedValues();
}, updateInterval);

// Use Pixi's ticker to update data every frame
app.ticker.add(async () => {
  // Update the displayed values
  var data = await fetchData(); // Wait for the data promise to resolve
  data = await convertFetched(data);
  updateDisplayedValues(data);
});

// Start fetching data immediately
fetchData();