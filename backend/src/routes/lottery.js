const express = require('express');
const pool = require('../config/database');
const { fetchTodayResults } = require('../services/lotteryFetcher');

const router = express.Router();

// GET /api/lottery/results - Get all results with optional filters
router.get('/results', async (req, res) => {
  try {
    const { time_slot, start_date, end_date, limit = 50, offset = 0 } = req.query;

    let query = 'SELECT * FROM draw_results WHERE 1=1';
    const params = [];
    let paramIndex = 1;

    if (time_slot) {
      query += ` AND time_slot = $${paramIndex++}`;
      params.push(time_slot);
    }
    if (start_date) {
      query += ` AND draw_date >= $${paramIndex++}`;
      params.push(start_date);
    }
    if (end_date) {
      query += ` AND draw_date <= $${paramIndex++}`;
      params.push(end_date);
    }

    query += ` ORDER BY draw_date DESC, time_slot ASC LIMIT $${paramIndex++} OFFSET $${paramIndex++}`;
    params.push(parseInt(limit, 10), parseInt(offset, 10));

    const result = await pool.query(query, params);

    // Get total count
    let countQuery = 'SELECT COUNT(*) FROM draw_results WHERE 1=1';
    const countParams = [];
    let countIndex = 1;

    if (time_slot) {
      countQuery += ` AND time_slot = $${countIndex++}`;
      countParams.push(time_slot);
    }
    if (start_date) {
      countQuery += ` AND draw_date >= $${countIndex++}`;
      countParams.push(start_date);
    }
    if (end_date) {
      countQuery += ` AND draw_date <= $${countIndex++}`;
      countParams.push(end_date);
    }

    const countResult = await pool.query(countQuery, countParams);

    res.json({
      data: result.rows,
      total: parseInt(countResult.rows[0].count, 10),
      limit: parseInt(limit, 10),
      offset: parseInt(offset, 10),
    });
  } catch (err) {
    console.error('Error fetching results:', err.message);
    res.status(500).json({ error: 'Failed to fetch results' });
  }
});

// GET /api/lottery/results/latest - Get latest results for each slot
router.get('/results/latest', async (req, res) => {
  try {
    const result = await pool.query(`
      SELECT DISTINCT ON (time_slot) *
      FROM draw_results
      ORDER BY time_slot, draw_date DESC
    `);
    res.json({ data: result.rows });
  } catch (err) {
    console.error('Error fetching latest results:', err.message);
    res.status(500).json({ error: 'Failed to fetch latest results' });
  }
});

// GET /api/lottery/stats - Get basic stats
router.get('/stats', async (req, res) => {
  try {
    const totalResult = await pool.query('SELECT COUNT(*) FROM draw_results');
    const dateRange = await pool.query(
      'SELECT MIN(draw_date) as earliest, MAX(draw_date) as latest FROM draw_results'
    );
    const slotCounts = await pool.query(
      'SELECT time_slot, COUNT(*) as count FROM draw_results GROUP BY time_slot ORDER BY time_slot'
    );

    res.json({
      total_records: parseInt(totalResult.rows[0].count, 10),
      date_range: dateRange.rows[0],
      slot_counts: slotCounts.rows,
    });
  } catch (err) {
    console.error('Error fetching stats:', err.message);
    res.status(500).json({ error: 'Failed to fetch stats' });
  }
});

// POST /api/lottery/sync - Manually trigger data sync
router.post('/sync', async (req, res) => {
  try {
    const count = await fetchTodayResults();
    res.json({ message: 'Sync complete', records_fetched: count });
  } catch (err) {
    console.error('Error during sync:', err.message);
    res.status(500).json({ error: 'Sync failed' });
  }
});

module.exports = router;
