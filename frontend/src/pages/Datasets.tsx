import { useEffect, useState } from "react";

interface DatasetInfo {
  name: string;
  split: string;
  num_samples: number;
  root_dir: string;
}

interface DatasetsResponse {
  datasets: DatasetInfo[];
}

interface DatasetsProps {
  baseUrl: string;
}

export default function Datasets({ baseUrl }: DatasetsProps) {
  const [data, setData] = useState<DatasetsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const apiBase = baseUrl ? baseUrl.replace(/\/$/, "") : "";

  useEffect(() => {
    fetch(apiBase ? `${apiBase}/api/datasets` : "/api/datasets")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: DatasetsResponse) => {
        setData(json);
        setError(null);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : "Failed to fetch datasets");
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [apiBase]);

  if (loading) return <p className="text-gray-600">Loading datasets…</p>;
  if (error) return <p className="text-red-600">Error: {error}</p>;
  if (!data?.datasets?.length)
    return <p className="text-gray-600">No datasets found. Run the SoccerNet loader and unzip data.</p>;

  return (
    <div>
      <h1 className="mb-4 text-2xl font-bold text-gray-900">Datasets</h1>
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Name
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Split
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium uppercase text-gray-500">
                Samples
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium uppercase text-gray-500">
                Root
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {data.datasets.map((ds) => (
              <tr key={`${ds.name}-${ds.split}`}>
                <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-gray-900">
                  {ds.name}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-600">{ds.split}</td>
                <td className="whitespace-nowrap px-4 py-3 text-right text-sm text-gray-600">
                  {ds.num_samples}
                </td>
                <td className="px-4 py-3 text-sm text-gray-500">{ds.root_dir}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
