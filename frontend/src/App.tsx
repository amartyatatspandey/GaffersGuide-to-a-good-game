import { Routes, Route, Link } from "react-router-dom";
import Datasets from "./pages/Datasets";
import Home from "./pages/Home";

function App() {
  const base = import.meta.env.VITE_API_URL || "";
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="border-b bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-6xl items-center gap-6">
          <Link to="/" className="text-lg font-semibold text-gray-900">
            Gaffer's Guide
          </Link>
          <Link to="/datasets" className="text-gray-600 hover:text-gray-900">
            Datasets
          </Link>
        </div>
      </nav>
      <main className="mx-auto max-w-6xl px-4 py-8">
        <Routes>
          <Route path="/" element={<Home baseUrl={base} />} />
          <Route path="/datasets" element={<Datasets baseUrl={base} />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
