const cron = require('node-cron');
const { fetchTodayResults, fetchHistoricalData } = require('./lotteryFetcher');
const axios = require('axios');

const ML_SERVICE_URL = process.env.ML_SERVICE_URL;

async function triggerTraining() {
  try {
    console.log('Triggering ML model training...');
    const response = await axios.post(`${ML_SERVICE_URL}/train`, {}, { timeout: 120000 });
    console.log('Training complete:', response.data);
    return response.data;
  } catch (err) {
    console.error('Training failed:', err.message);
    return null;
  }
}

async function triggerPredictions() {
  const slots = ['1pm', '6pm', '8pm'];
  const predictions = [];

  for (const slot of slots) {
    try {
      const response = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { time_slot: slot },
        { timeout: 30000 }
      );
      predictions.push(response.data);
      console.log(`Prediction for ${slot}:`, response.data);
    } catch (err) {
      console.error(`Prediction failed for ${slot}:`, err.message);
    }
  }

  return predictions;
}

function startCronJobs() {
  // Daily at 12:30 AM IST (7:00 PM UTC previous day)
  cron.schedule('0 19 * * *', async () => {
    console.log('Running daily cron job...');
    try {
      await fetchTodayResults();
      await triggerTraining();
      await triggerPredictions();
      console.log('Daily cron job complete.');
    } catch (err) {
      console.error('Daily cron job failed:', err.message);
    }
  });

  console.log('Cron jobs scheduled (daily at 12:30 AM IST).');
}

async function runStartupTasks() {
  console.log('Running startup tasks...');

  // Fetch historical data
  await fetchHistoricalData();

  // Train model
  await triggerTraining();

  // Generate predictions
  await triggerPredictions();

  console.log('Startup tasks complete.');
}

module.exports = { startCronJobs, runStartupTasks, triggerTraining, triggerPredictions };
