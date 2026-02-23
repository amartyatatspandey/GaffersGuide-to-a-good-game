interface HomeProps {
  baseUrl: string;
}

export default function Home({ baseUrl }: HomeProps) {
  const apiBase = baseUrl ? baseUrl.replace(/\/$/, "") : "";
  const healthUrl = apiBase ? `${apiBase}/health` : "/health";
  const datasetsUrl = apiBase ? `${apiBase}/api/datasets` : "/api/datasets";

  return (
    <div>
      <h1 className="mb-4 text-2xl font-bold text-gray-900">Welcome to Gaffer's Guide</h1>
      <p className="mb-6 text-gray-600">
        Tactical intelligence platform for football CV and analytics.
      </p>
      <div className="space-y-2 text-sm text-gray-500">
        <p>
          <strong>Health:</strong> <code className="rounded bg-gray-200 px-1">{healthUrl}</code>
        </p>
        <p>
          <strong>Datasets API:</strong>{" "}
          <code className="rounded bg-gray-200 px-1">{datasetsUrl}</code>
        </p>
      </div>
    </div>
  );
}
