const axios = require('axios');
const pool = require('../config/database');

const API_BASE = process.env.LOTTERY_API_BASE;
const TIME_SLOTS = ['1pm', '6pm', '8pm'];

async function fetchLatest(timeSlot) {
  try {
    const response = await axios.get(`${API_BASE}/latest`, {
      params: { time: timeSlot },
      timeout: 10000,
    });
    return response.data;
  } catch (err) {
    console.error(`Error fetching latest for ${timeSlot}:`, err.message);
    return null;
  }
}

async function fetchByDate(date, timeSlot) {
  try {
    const response = await axios.get(`${API_BASE}/by-date`, {
      params: { date, time: timeSlot },
      timeout: 10000,
    });
    return response.data;
  } catch (err) {
    console.error(`Error fetching ${date} ${timeSlot}:`, err.message);
    return null;
  }
}

async function storeResult(data, timeSlot) {
  if (!data || !data.date) return false;

  try {
    const prizes = data.prizes || {};
    await pool.query(
      `INSERT INTO draw_results (draw_no, draw_date, time_slot, prizes)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (draw_date, time_slot) DO UPDATE
       SET draw_no = EXCLUDED.draw_no, prizes = EXCLUDED.prizes`,
      [data.no || null, data.date, timeSlot, JSON.stringify(prizes)]
    );
    return true;
  } catch (err) {
    console.error(`Error storing result for ${data.date} ${timeSlot}:`, err.message);
    return false;
  }
}

async function getLatestStoredDate() {
  const result = await pool.query(
    'SELECT MAX(draw_date) as latest_date FROM draw_results'
  );
  return result.rows[0]?.latest_date || null;
}

function formatDate(date) {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

async function fetchHistoricalData() {
  const latestDate = await getLatestStoredDate();
  const today = new Date();
  let startDate;

  if (latestDate) {
    startDate = new Date(latestDate);
    startDate.setDate(startDate.getDate() + 1);
    console.log(`Resuming fetch from ${formatDate(startDate)}`);
  } else {
    startDate = new Date();
    startDate.setDate(startDate.getDate() - 90);
    console.log(`First run: fetching from ${formatDate(startDate)} (90 days back)`);
  }

  let fetched = 0;
  const current = new Date(startDate);

  while (current <= today) {
    const dateStr = formatDate(current);

    for (const slot of TIME_SLOTS) {
      const data = await fetchByDate(dateStr, slot);
      if (data) {
        const stored = await storeResult(data, slot);
        if (stored) fetched++;
      }
      // Small delay to avoid rate limiting
      await new Promise((r) => setTimeout(r, 300));
    }

    current.setDate(current.getDate() + 1);
  }

  console.log(`Historical fetch complete. Stored ${fetched} new records.`);
  return fetched;
}

async function fetchTodayResults() {
  let fetched = 0;
  const today = formatDate(new Date());

  for (const slot of TIME_SLOTS) {
    const data = await fetchByDate(today, slot);
    if (data) {
      const stored = await storeResult(data, slot);
      if (stored) fetched++;
    }
    await new Promise((r) => setTimeout(r, 300));
  }

  console.log(`Today's fetch complete. Stored ${fetched} records.`);
  return fetched;
}

module.exports = {
  fetchLatest,
  fetchByDate,
  storeResult,
  fetchHistoricalData,
  fetchTodayResults,
  getLatestStoredDate,
  TIME_SLOTS,
};
