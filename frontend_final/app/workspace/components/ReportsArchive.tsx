"use client";
import React, { useState, useEffect } from 'react';
import { Database, Search, FolderOpen, Calendar, ArrowRight, Loader2, Trash2, Film, Download } from 'lucide-react';
import { listReports, getReport, deleteReport } from '@/lib/api/jobs';
import { getApiBaseUrl, getAuthHeaders } from '@/lib/apiBase';

export function ReportsArchive({ onOpenReport }: { onOpenReport: (report: any) => void }) {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [downloadingVideo, setDownloadingVideo] = useState<string | null>(null);

  useEffect(() => {
    loadReports();
  }, []);

  async function loadReports() {
    try {
      const data = await listReports();
      setReports(data);
    } catch (error) {
      console.error("Failed to load reports:", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleOpen(reportId: string) {
    try {
      const fullReport = await getReport(reportId);
      onOpenReport(fullReport);
    } catch (error) {
      console.error("Failed to open report:", error);
    }
  }

  async function handleDelete(e: React.MouseEvent, reportId: string) {
    e.stopPropagation(); // Prevent opening the report when clicking delete
    if (!window.confirm("Are you sure you want to delete this report? This cannot be undone.")) {
      return;
    }
    
    try {
      await deleteReport(reportId);
      // Refresh the list after successful deletion
      await loadReports();
    } catch (error) {
      console.error("Failed to delete report:", error);
      alert("Failed to delete the report. Please try again.");
    }
  }

  async function handleDownloadVideo(e: React.MouseEvent, jobId: string) {
    e.stopPropagation(); // Prevent opening the report
    if (!jobId || jobId === 'unknown' || jobId === 'manual') {
      alert("No video is associated with this report.");
      return;
    }

    setDownloadingVideo(jobId);
    try {
      const base = getApiBaseUrl();
      const downloadUrl = `${base}/api/v1/elite/jobs/${jobId}/video/download`;
      const response = await fetch(downloadUrl, { headers: getAuthHeaders() });

      if (!response.ok) {
        throw new Error(`Video download failed: ${response.status}`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.setAttribute("download", `GaffersGuide_TacticalRadar_${jobId}.mp4`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 5000);
    } catch (error) {
      console.error("Failed to download video:", error);
      alert("Failed to download the video. The video may not have been generated for this analysis.");
    } finally {
      setDownloadingVideo(null);
    }
  }

  const filteredReports = reports.filter(r => 
    (r.video_title || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
    (r.quality_profile || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="h-full w-full bg-[#0a0f0a] flex flex-col animate-fade-in font-sans p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-200 tracking-tight flex items-center gap-3">
            <FolderOpen className="text-emerald-500" />
            Tactical Intelligence Repository
          </h1>
          <p className="text-sm text-gray-500 mt-2 font-mono">Persistent tactical reports and historical match analytics.</p>
        </div>
        
        <div className="relative">
          <input 
            type="text" 
            placeholder="Search reports..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-[#111a12] border border-gray-800 rounded-lg pl-10 pr-4 py-2 text-sm text-gray-300 focus:outline-none focus:border-emerald-500 transition-colors w-64"
          />
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600" />
        </div>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="animate-spin text-emerald-500" size={48} />
        </div>
      ) : filteredReports.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-gray-900 rounded-2xl">
          <Database className="text-gray-800 mb-4" size={64} />
          <p className="text-gray-600 font-mono">No tactical reports found in repository.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 overflow-y-auto pr-2 custom-scrollbar">
          {filteredReports.map((report) => (
            <div 
              key={report.id} 
              onClick={() => handleOpen(report.id)}
              className="bg-[#111a12] border border-gray-900 hover:border-emerald-500/50 rounded-xl p-5 cursor-pointer transition-all hover:bg-[#152018] group flex flex-col justify-between min-h-[240px] shadow-sm hover:shadow-emerald-900/10"
            >
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div className="p-2 bg-emerald-500/10 rounded-lg group-hover:bg-emerald-500/20 transition-colors">
                    <Database size={20} className="text-emerald-500" />
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] font-bold text-gray-600 font-mono flex items-center gap-1 uppercase tracking-tighter">
                      <Calendar size={12}/> {new Date(report.saved_at).toLocaleDateString()}
                    </span>
                    {/* Download Video Button */}
                    <button 
                      onClick={(e) => handleDownloadVideo(e, report.job_id)}
                      disabled={downloadingVideo === report.job_id}
                      className="text-gray-600 hover:text-emerald-400 p-1 rounded hover:bg-emerald-500/10 transition-colors disabled:opacity-50"
                      title="Download Tactical Video"
                    >
                      {downloadingVideo === report.job_id ? (
                        <Loader2 size={14} className="animate-spin" />
                      ) : (
                        <Film size={14} />
                      )}
                    </button>
                    {/* Delete Button */}
                    <button 
                      onClick={(e) => handleDelete(e, report.id)}
                      className="text-gray-600 hover:text-red-500 p-1 rounded hover:bg-red-500/10 transition-colors"
                      title="Delete Report"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                
                <h3 className="font-bold text-gray-300 text-sm mb-1 truncate" title={report.video_title}>
                  {report.video_title}
                </h3>
                <p className="text-[10px] font-mono text-gray-500 mb-4 uppercase tracking-widest">
                  {report.quality_profile} • {report.flaw_count} Insights
                </p>
                
                <div className="flex items-center gap-4 mb-4">
                   <div className="flex-1">
                      <p className="text-[8px] text-gray-600 uppercase mb-1">Win Prob (Red)</p>
                      <div className="h-1 bg-gray-900 rounded-full overflow-hidden">
                         <div className="h-full bg-emerald-500" style={{ width: `${report.win_probability?.team_red || 50}%` }} />
                      </div>
                   </div>
                   <div className="flex-1">
                      <p className="text-[8px] text-gray-600 uppercase mb-1">TPI (Red)</p>
                      <p className="text-xs font-mono text-emerald-400 font-bold">{Math.round(report.tactical_power?.red || 0)}</p>
                   </div>
                </div>
              </div>

              <div className="mt-auto flex items-center justify-between border-t border-gray-900 pt-4">
                 <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">View Analysis</span>
                 <ArrowRight size={16} className="text-gray-600 group-hover:text-emerald-400 transition-colors transform group-hover:translate-x-1" />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
