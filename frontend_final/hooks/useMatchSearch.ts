import { useState, useEffect, useRef } from 'react';
import { searchFixtures } from '@/lib/services/footballApi';
import type { FixtureSearchResult } from '@/types/lineup';

const DEBOUNCE_MS = 300;
const MIN_QUERY_LENGTH = 2;

/**
 * Wraps live fixture search state (query, results, loading, error) with a
 * debounce so we don't hit the backend on every keystroke.
 */
export function useMatchSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<FixtureSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const latestRequestId = useRef(0);

  useEffect(() => {
    const trimmed = query.trim();

    if (trimmed.length < MIN_QUERY_LENGTH) {
      setResults([]);
      setLoading(false);
      setError(null);
      setHasSearched(false);
      return;
    }

    const requestId = ++latestRequestId.current;
    setLoading(true);
    setError(null);

    const timer = setTimeout(() => {
      searchFixtures(trimmed)
        .then((data) => {
          if (requestId !== latestRequestId.current) return; // stale response
          setResults(data);
          setHasSearched(true);
        })
        .catch((err) => {
          if (requestId !== latestRequestId.current) return;
          setError(err instanceof Error ? err.message : 'Failed to search fixtures.');
          setResults([]);
          setHasSearched(true);
        })
        .finally(() => {
          if (requestId !== latestRequestId.current) return;
          setLoading(false);
        });
    }, DEBOUNCE_MS);

    return () => clearTimeout(timer);
  }, [query]);

  return { query, setQuery, results, loading, error, hasSearched };
}
