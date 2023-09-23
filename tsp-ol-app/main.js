import './style.css';
import {Map, View} from 'ol';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import LineString from 'ol/geom/LineString';
import { Vector as VectorLayer } from 'ol/layer';
import { Vector as VectorSource } from 'ol/source';
import { Icon, Stroke, Style } from 'ol/style';
import { fromLonLat } from 'ol/proj';


const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({
      source: new OSM(),
    }),
  ],
  view: new View({
    center: fromLonLat([2.3522, 46]), // Paris coordinates in Web Mercator projection
    zoom: 6,
    projection: 'EPSG:3857', // Set the projection to Web Mercator
  }),
});

// Fetch and convertion functions from Express.js
// Function to convert a fetch to json
function convertFetched(inputString) {

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
  const bestIndividual = bestIndividualString.slice(1,-1).split(',').map(Number);
  // console.log(bestIndividual, typeof bestIndividual);
  
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
    'Number of Individuals': noi,
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
  // console.log(jsonObject);
  return jsonObject;
};

// Function to fetch data from the server
async function fetchData() {
  try {
    const response = await fetch('http://localhost:3000/data'); // Assuming your server is serving data at '/data'
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const dataText = await response.text(); // Await the response.text() call
    // console.log(dataText);
    const jsonObject = convertFetched(dataText); // Pass the text data to the conversion function
    return jsonObject;
  } catch (error) {
    console.error('Fetch error:', error);
    // Handle the error as needed, e.g., return an error object
  }
}

// Function to fetch data from the server
async function fetchCities() {
  try {
    const response = await fetch('http://localhost:3000/cities'); // Assuming your server is serving data at '/data'
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const dataText = await response.text(); // Await the response.text() call
    // console.log(dataText, typeof dataText);
    // Parse the JSON data
    let jsonData;
    try {
      jsonData = JSON.parse(dataText);
    } catch (parseError) {
      console.error('Error parsing JSON:', parseError);
      res.status(500).send('Internal Server Error');
      return;
    }
    // console.log(jsonData, typeof jsonData);
    return jsonData;

  } catch (error) {
    console.error('Fetch error:', error);
    // Handle the error as needed, e.g., return an error object
  }
}

var cityData= await fetchCities();

console.log(cityData, typeof cityData);

// Convert data coordinates to the map's projection
cityData.forEach((city) => {
  const [longitude, latitude] = fromLonLat([city.lng, city.lat]);
  city.lat = latitude;
  city.lng = longitude;
});

// Create a vector source and layer for features
const vectorSource = new VectorSource();
const vectorLayer = new VectorLayer({
  source: vectorSource,
  dataProjection: 'EPSG:3006'
});

// Add the vector layer to the map
map.addLayer(vectorLayer);

function addFirstToEnd(arr) {
  if (arr.length >= 1) {
    const firstElement = arr[0];
    arr.push(firstElement);
  }
}

// Function to create a path between selected cities
function createPathBetweenCities(selectedIndexes) {
  // console.log(selectedIndexes);
  addFirstToEnd(selectedIndexes);
  const pathCoordinates = selectedIndexes.map((index) => {
    const city = cityData[index];
    return [city.lng, city.lat];
  });

  if (pathCoordinates.length >= 2) {
    // Create a LineString for the path
    const pathLine = new LineString(pathCoordinates);
    const pathFeature = new Feature({
      geometry: pathLine,
      // Style for the path
      style: new Style({
        stroke: new Stroke({
          color: 'blue', // Customize the color as needed
          width: 3, // Customize the line width as needed
        }),
      }),
    });

    // Create point features for selected cities and their icons
    const cityFeatures = selectedIndexes.map((index) => {
      const city = cityData[index];
      return new Feature({
        geometry: new Point([city.lng, city.lat]),
        // Style for city points
        style: new Style({
          image: new Icon({
            src: 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Google_Maps_pin.svg/1170px-Google_Maps_pin.svg.png', // URL to an icon (customize as needed)
            scale: 5, // Icon size

          }),
        }),
      });
    });

    // Add the path feature and city point features to the vector source
    vectorSource.addFeature(pathFeature);
    cityFeatures.forEach((cityFeature) => {
      vectorSource.addFeature(cityFeature);
    });
  }
};

// Update coordinates and redraw features periodically
setInterval(async () => {
  try {
    
    // First fetch the data
    const data = await fetchData();
    
    // Clear the existing features from the vector source
    vectorSource.clear();
    
    // Dispatch it to the createPathBetweenCities function
    createPathBetweenCities(data['Best Individual']);
    
    const iteration = data['Iteration'];
    const bestScore = data['Score'];
    const numberOfIndividuals = parseInt(data["Number of Individuals"]);
    const numberOfSharedPatterns = parseInt(data['Number of Shared Patterns']);
    const numberOfSameIndividuals = parseInt(data['Number of Same Individuals']);
    const scores = data['All Scores'];
    const alpha = data['alpha'];
    const beta = data['beta'];
    const gamma = data["gamma"];
    const median = data['Median'];
    const q1 = data['Q1'];
    const q3 = data['Q3'];
    const max = data['Max'];
    
    const ppNOSI = Math.floor(numberOfSameIndividuals/numberOfIndividuals * 100);
    
    const valueBox = document.getElementById('value-box');

    valueBox.innerHTML = `
    <div class="score-section">
    <h1>Iteration: ${iteration}</h1>
    <h2>Score</h2>
    <p>Best Score: ${bestScore}</p>
    <p>Shared Patterns: ${numberOfSharedPatterns}</p>
    <p>Same Individuals: ${numberOfSameIndividuals} (${ppNOSI}%)</p>
    </div>
    <div class="parameter-section">
    <h2>Parameters</h2>
    <p>Alpha: ${alpha}</p>
    <p>Beta: ${beta}</p>
    <p>Gamma: ${gamma}</p>
    </div>
    
    `;
    //   <div class="box-parameters-section">
    //   <h2>Box Parameters</h2>
    //   <p>Min: ${bestScore}</p>
    //   <p>Q1: ${q1}</p>
    //   <p>Median: ${median}</p>
    //   <p>Q3: ${q3}</p>
    //   <p>Max: ${max}</p>
    // </div>
    
    const plot_data = [
      {
        y: scores,
        type: 'box',
        name: 'Scores',
        boxpoints: 'outliers', // Display outliers
        marker: {
          opacity: 0.7, // Set opacity
          color: 'blue', // Customize the box color
        },
        boxmean: true, // Display mean line
        line: {
          width: 1, // Adjust the border width
        },
      }
    ];

    const layout = {
      title: 'Scores Box Plot ',
      xaxis: {
        title: `Iteration ${iteration}`,
      },
      yaxis: {
        title: 'Scores',
      },
      plot_bgcolor: 'rgba(0,0,0,0)', // Make the plot background transparent
      paper_bgcolor: 'rgba(0,0,0,0.1)', // Make the paper (surrounding) background transparent
    };

  Plotly.newPlot('box-plot', plot_data, layout);

} catch (error) {
  console.error('Error fetching or processing data:', error);
}
}, 100); // Update every 0.1 second