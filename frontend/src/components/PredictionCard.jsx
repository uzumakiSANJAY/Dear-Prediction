import { useState } from 'react';

const slotLabels = {
  '1pm': '1:00 PM',
  '6pm': '6:00 PM',
  '8pm': '8:00 PM',
};

const slotColors = {
  '1pm': 'from-amber-500 to-orange-600',
  '6pm': 'from-blue-500 to-indigo-600',
  '8pm': 'from-purple-500 to-pink-600',
};

const mainPrizes = [
  { key: '1st', label: '1st Prize' },
  { key: '2nd', label: '2nd Prize' },
  { key: '3rd', label: '3rd Prize' },
  { key: '4th', label: '4th Prize' },
  { key: '5th', label: '5th Prize' },
];

const allPrizes = [
  { key: 'mc', label: 'MC' },
  { key: '1st', label: '1st Prize' },
  { key: 'cons', label: 'Consolation' },
  { key: '2nd', label: '2nd Prize' },
  { key: '3rd', label: '3rd Prize' },
  { key: '4th', label: '4th Prize' },
  { key: '5th', label: '5th Prize' },
];

function formatPrizeValue(val) {
  if (!val) return '----';
  if (typeof val === 'string') return val;
  if (Array.isArray(val)) {
    if (val.length <= 3) return val.join(', ');
    return `${val.slice(0, 3).join(', ')} +${val.length - 3}`;
  }
  return String(val);
}

function NumberGrid({ numbers, color }) {
  if (!numbers || !Array.isArray(numbers) || numbers.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {numbers.map((n, i) => (
        <span
          key={i}
          className={`px-1.5 py-0.5 rounded text-xs font-mono font-medium ${color}`}
        >
          {n}
        </span>
      ))}
    </div>
  );
}

export default function PredictionCard({ prediction }) {
  const [expanded, setExpanded] = useState(false);
  const slot = prediction?.time_slot || '1pm';
  const numbers = prediction?.predicted_numbers || {};
  const confidence = prediction?.confidence || 0;
  const modelUsed = prediction?.model_used || 'N/A';
  const analysis = prediction?.analysis || {};

  const confidencePercent = Math.round(confidence * 100);
  const confidenceColor =
    confidencePercent > 70
      ? 'text-green-400'
      : confidencePercent > 45
        ? 'text-yellow-400'
        : 'text-orange-400';

  return (
    <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
      <div className={`bg-gradient-to-r ${slotColors[slot] || slotColors['1pm']} px-5 py-3`}>
        <h3 className="text-lg font-bold text-white">{slotLabels[slot] || slot} Draw</h3>
        <p className="text-sm text-white/80">Model: {modelUsed}</p>
      </div>

      <div className="p-5 space-y-2.5">
        {/* Main prizes: show single number or summary */}
        {mainPrizes.map(({ key, label }) => (
          <div key={key} className="flex justify-between items-center">
            <span className="text-sm text-gray-400">{label}</span>
            <span className="font-mono text-lg font-bold text-white tracking-wider">
              {formatPrizeValue(numbers[key])}
            </span>
          </div>
        ))}

        {/* Confidence */}
        <div className="border-t border-gray-700 pt-3 mt-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Confidence</span>
            <span className={`text-sm font-semibold ${confidenceColor}`}>
              {confidencePercent}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
            <div
              className={`h-2 rounded-full bg-gradient-to-r ${slotColors[slot] || slotColors['1pm']}`}
              style={{ width: `${confidencePercent}%` }}
            />
          </div>
        </div>

        {/* View All toggle */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full mt-3 py-2 text-sm text-indigo-400 hover:text-indigo-300 border border-gray-700 hover:border-gray-600 rounded-lg transition-colors"
        >
          {expanded ? 'Hide Details' : 'View All'}
        </button>

        {expanded && (
          <div className="mt-3 space-y-4 border-t border-gray-700 pt-4">
            {/* All prizes with full number lists */}
            {allPrizes.map(({ key, label }) => {
              const val = numbers[key];
              const isArray = Array.isArray(val);
              return (
                <div key={key}>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-semibold text-gray-300">{label}</span>
                    {!isArray && (
                      <span className="font-mono text-base font-bold text-white tracking-wider">
                        {val || '----'}
                      </span>
                    )}
                    {isArray && (
                      <span className="text-xs text-gray-500">{val.length} numbers</span>
                    )}
                  </div>
                  {isArray && (
                    <NumberGrid numbers={val} color="bg-gray-700/60 text-gray-200" />
                  )}
                </div>
              );
            })}

            {/* Hot numbers */}
            {analysis.hot_numbers && analysis.hot_numbers.length > 0 && (
              <div className="border-t border-gray-700 pt-3">
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Hot Numbers (frequent recent)
                </h4>
                <NumberGrid numbers={analysis.hot_numbers} color="bg-red-500/20 text-red-300" />
              </div>
            )}

            {/* Cold numbers */}
            {analysis.cold_numbers && analysis.cold_numbers.length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Cold Numbers (rare)
                </h4>
                <NumberGrid numbers={analysis.cold_numbers} color="bg-blue-500/20 text-blue-300" />
              </div>
            )}

            {/* Last digit frequency */}
            {analysis.last_digit_freq && Object.keys(analysis.last_digit_freq).length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Last Digit Frequency
                </h4>
                <div className="grid grid-cols-5 gap-1.5">
                  {Object.entries(analysis.last_digit_freq).map(([digit, count]) => (
                    <div key={digit} className="text-center bg-gray-700/50 rounded py-1.5">
                      <div className="font-mono text-sm text-white font-bold">{digit}</div>
                      <div className="text-xs text-gray-400">{count}x</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Per-model breakdown */}
            {analysis.model_breakdown && Object.keys(analysis.model_breakdown).length > 0 && (
              <div>
                <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
                  Per-Model Predictions
                </h4>
                {Object.entries(analysis.model_breakdown).map(([model, preds]) => (
                  <div key={model} className="mb-2">
                    <span className="text-xs text-gray-500 capitalize">{model.replace(/_/g, ' ')}</span>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {Object.entries(preds).map(([prize, num]) => (
                        <span key={prize} className="px-2 py-0.5 bg-gray-700/50 text-gray-300 rounded text-xs font-mono">
                          {prize}: {num}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
