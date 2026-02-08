import { Link, useLocation } from 'react-router-dom';

export default function Header() {
  const location = useLocation();

  const linkClass = (path) =>
    `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
      location.pathname === path
        ? 'bg-indigo-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;

  return (
    <header className="bg-gray-800 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
        <Link to="/" className="text-xl font-bold text-white">
          Lottery Prediction
        </Link>
        <nav className="flex gap-2">
          <Link to="/" className={linkClass('/')}>
            Dashboard
          </Link>
          <Link to="/history" className={linkClass('/history')}>
            History
          </Link>
        </nav>
      </div>
    </header>
  );
}
