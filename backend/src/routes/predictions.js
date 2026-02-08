const express = require('express');
const axios = require('axios');
const pool = require('../config/database');

const router = express.Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL;

// GET /api/predictions - Get stored predictions
router.get('/', async (req, res) => {
  try {
    const { date, time_slot, limit = 30 } = req.query;

    let query = 'SELECT * FROM predictions WHERE 1=1';
    const params = [];
    let paramIndex = 1;

    if (date) {
      query += ` AND prediction_date = $${paramIndex++}`;
      params.push(date);
    }
    if (time_slot) {
      query += ` AND time_slot = $${paramIndex++}`;
      params.push(time_slot);
    }

    query += ` ORDER BY created_at DESC LIMIT $${paramIndex++}`;
    params.push(parseInt(limit, 10));

    const result = await pool.query(query, params);
    res.json({ data: result.rows });
  } catch (err) {
    console.error('Error fetching predictions:', err.message);
    res.status(500).json({ error: 'Failed to fetch predictions' });
  }
});

// GET /api/predictions/latest - Get latest predictions for each slot
router.get('/latest', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT DISTINCT ON (time_slot) *
      FROM predictions
      ORDER BY time_slot, created_at DESC
    `);
    res.json({ data: result.rows });
  } catch (err) {
    console.error('Error fetching latest predictions:', err.message);
    res.status(500).json({ error: 'Failed to fetch latest predictions' });
  }
});

// POST /api/predictions/generate - Generate new predictions
router.post('/generate', async (req, res) => {
  try {
    const { time_slot } = req.body;
    const slots = time_slot ? [time_slot] : ['1pm', '6pm', '8pm'];
    const results = [];

    for (const slot of slots) {
      const response = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { time_slot: slot },
        { timeout: 30000 }
      );

      const prediction = response.data;

      // Store prediction
      await pool.query(
        `INSERT INTO predictions (prediction_date, time_slot, predicted_numbers, confidence, model_used, analysis)
         VALUES ($1, $2, $3, $4, $5, $6)`,
        [
          prediction.prediction_date || new Date().toISOString().split('T')[0],
          slot,
          JSON.stringify(prediction.predicted_numbers),
          prediction.confidence || 0,
          prediction.model_used || 'ensemble',
          JSON.stringify(prediction.analysis || {}),
        ]
      );

      results.push(prediction);
    }

    res.json({ data: results });
  } catch (err) {
    console.error('Error generating predictions:', err.message);
    res.status(500).json({ error: 'Failed to generate predictions' });
  }
});

// POST /api/predictions/train - Trigger model retraining
router.post('/train', async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/train`, {}, { timeout: 120000 });
    res.json(response.data);
  } catch (err) {
    console.error('Error training model:', err.message);
    res.status(500).json({ error: 'Training failed' });
  }
});

module.exports = router;
