import { useState, useEffect } from 'react';
import ResultsTable from '../components/ResultsTable';
import { getResults } from '../services/api';

export default function History() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [timeSlot, setTimeSlot] = useState('');
  const limit = 20;

  useEffect(() => {
    loadResults();
  }, [page, timeSlot]);

  async function loadResults() {
    setLoading(true);
    try {
      const params = { limit, offset: page * limit };
      if (timeSlot) params.time_slot = timeSlot;

      const data = await getResults(params);
      setResults(data.data || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error('Failed to load results:', err);
    }
    setLoading(false);
  }

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h2 className="text-xl font-bold text-white">Historical Results</h2>
        <div className="flex gap-2">
          {['', '1pm', '6pm', '8pm'].map((slot) => (
            <button
              key={slot}
              onClick={() => {
                setTimeSlot(slot);
                setPage(0);
              }}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                timeSlot === slot
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {slot || 'All'}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 p-4">
        <ResultsTable results={results} loading={loading} />

        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-700">
            <span className="text-sm text-gray-400">
              Showing {page * limit + 1}-{Math.min((page + 1) * limit, total)} of {total}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 text-white rounded-lg disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
