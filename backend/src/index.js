require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const pool = require('./config/database');
const lotteryRoutes = require('./routes/lottery');
const predictionRoutes = require('./routes/predictions');
const { startCronJobs, runStartupTasks } = require('./services/cronService');

const app = express();
const PORT = process.env.BACKEND_PORT || 5000;

app.use(cors());
app.use(express.json());

// Routes
app.use('/api/lottery', lotteryRoutes);
app.use('/api/predictions', predictionRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

async function initDatabase() {
  const schemaPath = path.join(__dirname, 'models', 'schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  await pool.query(schema);
  console.log('Database schema initialized.');
}

async function start() {
  try {
    // Initialize database
    await initDatabase();

    // Start Express server
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Backend server running on port ${PORT}`);
    });

    // Start cron jobs
    startCronJobs();

    // Run startup tasks (fetch data, train, predict) in background
    runStartupTasks().catch((err) => {
      console.error('Startup tasks error:', err.message);
    });
  } catch (err) {
    console.error('Failed to start server:', err);
    process.exit(1);
  }
}

start();
