import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
});

export async function getResults(params = {}) {
  const response = await api.get('/lottery/results', { params });
  return response.data;
}

export async function getLatestResults() {
  const response = await api.get('/lottery/results/latest');
  return response.data;
}

export async function getStats() {
  const response = await api.get('/lottery/stats');
  return response.data;
}

export async function syncData() {
  const response = await api.post('/lottery/sync');
  return response.data;
}

export async function getLatestPredictions() {
  const response = await api.get('/predictions/latest');
  return response.data;
}

export async function getPredictions(params = {}) {
  const response = await api.get('/predictions', { params });
  return response.data;
}

export async function generatePredictions(timeSlot) {
  const response = await api.post('/predictions/generate', {
    time_slot: timeSlot,
  });
  return response.data;
}

export async function trainModel() {
  const response = await api.post('/predictions/train');
  return response.data;
}

export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
