"use client";
import React, { useEffect, useMemo, useState } from "react";
import { ArrowRight, Calendar, Database, FolderOpen, Search } from "lucide-react";

import { type ReportEntry, getReports } from "@/lib/api";

interface ReportsArchiveProps {
  activeJobId: string | null;
}

export function ReportsArchive({ activeJobId }: ReportsArchiveProps): React.JSX.Element {
  const [query, setQuery] = useState("");
  const [reports, setReports] = useState<ReportEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async (): Promise<void> => {
      try {
        setLoading(true);
        const payload = await getReports();
        if (mounted) {
          setReports(payload.reports);
          setError(null);
        }
      } catch (loadError) {
        if (mounted) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load reports.");
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, []);

  const filteredReports = useMemo(
    () =>
      reports.filter((report) => {
        const haystack = `${report.job_id} ${report.report_filename}`.toLowerCase();
        return haystack.includes(query.toLowerCase());
      }),
    [query, reports],
  );

  return (
    <div className="h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
            <FolderOpen className="text-emerald-500" />
            Telemetry Archive
          </h1>
          <p className="text-sm text-gray-500 mt-2 font-mono">Historical match analysis and tactical debriefs.</p>
        </div>
        
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search archives..." 
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="bg-[#111a12] border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors w-64"
          />
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
        </div>
      </div>

      {loading ? (
        <div className="rounded-xl border border-gray-900 bg-[#111a12] px-5 py-4 text-gray-400">
          Loading reports...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-900 bg-red-950/30 px-5 py-4 text-red-300">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredReports.map((report) => {
            const created = new Date(report.created_at).toLocaleString();
            const active = activeJobId === report.job_id;
            return (
              <div
                key={report.job_id}
                className={`bg-[#111a12] border rounded-xl p-5 transition-all group flex flex-col justify-between min-h-[220px] ${
                  active
                    ? "border-emerald-500/70 shadow-[0_0_24px_rgba(16,185,129,0.22)]"
                    : "border-gray-900 hover:border-emerald-500/50 hover:bg-[#152018]"
                }`}
              >
                <div>
                  <div className="flex justify-between items-start mb-4">
                    <div className="p-2 bg-emerald-500/10 rounded-lg">
                      <Database size={20} className="text-emerald-500" />
                    </div>
                    <span className="text-xs font-bold text-gray-600 font-mono flex items-center gap-1">
                      <Calendar size={12} /> {created}
                    </span>
                  </div>
                  <h3
                    className="font-bold text-gray-300 text-sm mb-1 truncate"
                    title={report.report_filename}
                  >
                    {report.report_filename}
                  </h3>
                  <p className="text-xs font-mono text-gray-500 mb-4">job_id: {report.job_id}</p>
                </div>
                <div className="mt-6 flex items-center justify-between border-t border-gray-900 pt-4">
                  <span className="text-xs font-bold text-emerald-500 uppercase tracking-widest">
                    {active ? "Active Session" : "Analyzed"}
                  </span>
                  <ArrowRight
                    size={16}
                    className="text-gray-600 group-hover:text-emerald-400 transition-colors transform group-hover:translate-x-1"
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
