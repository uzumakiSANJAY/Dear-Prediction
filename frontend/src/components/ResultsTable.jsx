import { useState } from 'react';

const prizeKeys = ['mc', '1st', 'cons', '2nd', '3rd', '4th', '5th'];

function formatPrize(val) {
  if (!val) return '-';
  if (Array.isArray(val)) {
    if (val.length === 0) return '-';
    if (val.length <= 3) return val.join(', ');
    return `${val.slice(0, 3).join(', ')} +${val.length - 3} more`;
  }
  return String(val);
}

function ExpandedRow({ prizes }) {
  return (
    <tr>
      <td colSpan={10} className="px-3 py-3 bg-gray-900/50">
        <div className="grid grid-cols-1 gap-3">
          {prizeKeys.map((key) => {
            const val = prizes?.[key];
            if (!val || (Array.isArray(val) && val.length === 0)) return null;
            const items = Array.isArray(val) ? val : [val];
            if (items.length <= 3) return null;
            return (
              <div key={key}>
                <span className="text-xs font-semibold text-gray-500 uppercase">{key}</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {items.map((n, i) => (
                    <span key={i} className="px-1.5 py-0.5 bg-gray-700 text-gray-300 rounded text-xs font-mono">
                      {n}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </td>
    </tr>
  );
}

export default function ResultsTable({ results, loading }) {
  const [expandedId, setExpandedId] = useState(null);

  if (loading) {
    return <div className="text-center py-12 text-gray-400">Loading results...</div>;
  }

  if (!results || results.length === 0) {
    return <div className="text-center py-12 text-gray-400">No results found.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-3 px-3 text-gray-400 font-medium">Date</th>
            <th className="text-left py-3 px-3 text-gray-400 font-medium">Slot</th>
            <th className="text-left py-3 px-3 text-gray-400 font-medium">Draw No</th>
            {prizeKeys.map((key) => (
              <th key={key} className="text-left py-3 px-3 text-gray-400 font-medium uppercase">
                {key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row) => {
            const prizes = typeof row.prizes === 'string' ? JSON.parse(row.prizes) : row.prizes;
            const isExpanded = expandedId === row.id;
            const hasExpandable = prizeKeys.some((k) => Array.isArray(prizes?.[k]) && prizes[k].length > 3);

            return (
              <>
                <tr
                  key={row.id}
                  className={`border-b border-gray-800 hover:bg-gray-800/50 ${hasExpandable ? 'cursor-pointer' : ''}`}
                  onClick={() => hasExpandable && setExpandedId(isExpanded ? null : row.id)}
                >
                  <td className="py-2 px-3 text-gray-300 whitespace-nowrap">
                    {new Date(row.draw_date).toLocaleDateString()}
                  </td>
                  <td className="py-2 px-3">
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-indigo-500/20 text-indigo-300">
                      {row.time_slot}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-gray-300 font-mono">{row.draw_no || '-'}</td>
                  {prizeKeys.map((key) => (
                    <td key={key} className="py-2 px-3 font-mono text-white text-xs max-w-[120px] truncate">
                      {formatPrize(prizes?.[key])}
                    </td>
                  ))}
                </tr>
                {isExpanded && <ExpandedRow key={`exp-${row.id}`} prizes={prizes} />}
              </>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
