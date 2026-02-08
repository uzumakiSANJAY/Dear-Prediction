import { useState, useEffect } from 'react';
import PredictionCard from '../components/PredictionCard';
import ResultsTable from '../components/ResultsTable';
import {
  getLatestPredictions,
  getLatestResults,
  getStats,
  generatePredictions,
  syncData,
  healthCheck,
} from '../services/api';

export default function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [latestResults, setLatestResults] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');

  const slots = ['1pm', '6pm', '8pm'];

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    try {
      const health = await healthCheck();
      setBackendStatus(health.status === 'ok' ? 'connected' : 'error');
    } catch {
      setBackendStatus('disconnected');
    }

    try {
      const [predData, resultData, statsData] = await Promise.all([
        getLatestPredictions().catch(() => ({ data: [] })),
        getLatestResults().catch(() => ({ data: [] })),
        getStats().catch(() => null),
      ]);

      setPredictions(predData.data || []);
      setLatestResults(resultData.data || []);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to load data:', err);
    }
    setLoading(false);
  }

  async function handleSync() {
    setSyncing(true);
    try {
      await syncData();
      await loadData();
    } catch (err) {
      console.error('Sync failed:', err);
    }
    setSyncing(false);
  }

  async function handleGenerate() {
    setGenerating(true);
    try {
      await generatePredictions();
      await loadData();
    } catch (err) {
      console.error('Generate failed:', err);
    }
    setGenerating(false);
  }

  function getPredictionForSlot(slot) {
    return predictions.find((p) => p.time_slot === slot) || { time_slot: slot };
  }

  const statusColor =
    backendStatus === 'connected'
      ? 'bg-green-500'
      : backendStatus === 'checking'
        ? 'bg-yellow-500'
        : 'bg-red-500';

  return (
    <div className="space-y-8">
      {/* Status Bar */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
          <span className="text-sm text-gray-400">
            Backend: {backendStatus}
          </span>
          {stats && (
            <span className="text-sm text-gray-500 ml-4">
              {stats.total_records} records stored
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleSync}
            disabled={syncing}
            className="px-4 py-2 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg disabled:opacity-50 transition-colors"
          >
            {syncing ? 'Syncing...' : 'Sync Data'}
          </button>
          <button
            onClick={handleGenerate}
            disabled={generating}
            className="px-4 py-2 text-sm bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg disabled:opacity-50 transition-colors"
          >
            {generating ? 'Generating...' : 'New Predictions'}
          </button>
        </div>
      </div>

      {/* Predictions */}
      <div>
        <h2 className="text-xl font-bold text-white mb-4">Today's Predictions</h2>
        {loading ? (
          <div className="text-gray-400">Loading predictions...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {slots.map((slot) => (
              <PredictionCard key={slot} prediction={getPredictionForSlot(slot)} />
            ))}
          </div>
        )}
      </div>

      {/* Latest Results */}
      <div>
        <h2 className="text-xl font-bold text-white mb-4">Latest Draw Results</h2>
        <div className="bg-gray-800 rounded-xl border border-gray-700 p-4">
          <ResultsTable results={latestResults} loading={loading} />
        </div>
      </div>
    </div>
  );
}
