import React from 'react';
import { Save, Download, X, Film, FileText, CheckCircle2, Loader2 } from 'lucide-react';

interface SaveResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSaveBoth: () => void;
  onSaveReportOnly: () => void;
  isSaving: boolean;
  saveStatus: 'idle' | 'success' | 'error' | 'rendering';
}

export function SaveResultsModal({
  isOpen,
  onClose,
  onSaveBoth,
  onSaveReportOnly,
  isSaving,
  saveStatus
}: SaveResultsModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-[#0a0f0a] border border-emerald-500/30 rounded-2xl shadow-[0_0_50px_rgba(16,185,129,0.15)] w-full max-w-md overflow-hidden relative animate-in fade-in zoom-in duration-200">
        
        {/* Header */}
        <div className="p-6 border-b border-gray-900 bg-[#111a12]/50 flex justify-between items-start">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Save Tactical Results</h2>
            <p className="text-sm text-gray-500 mt-1">Analysis complete. What would you like to save?</p>
          </div>
          <button 
            onClick={onClose}
            className="text-gray-500 hover:text-white p-1 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {saveStatus === 'success' ? (
            <div className="flex flex-col items-center justify-center py-8 text-emerald-500 space-y-4">
              <CheckCircle2 size={48} className="animate-in zoom-in duration-300" />
              <p className="font-bold tracking-widest uppercase">Successfully Saved!</p>
            </div>
          ) : saveStatus === 'rendering' ? (
            <div className="flex flex-col items-center justify-center py-8 text-emerald-500 space-y-4">
              <Loader2 size={48} className="animate-spin" />
              <p className="font-bold tracking-widest uppercase text-sm">Rendering Tactical Video...</p>
              <p className="text-xs text-gray-500 text-center max-w-xs">Stitching radar overlay onto your match footage. This may take 30–60 seconds.</p>
            </div>
          ) : (
            <>
              {/* Option 1: Save Both */}
              <button
                disabled={isSaving}
                onClick={onSaveBoth}
                className="w-full flex items-center gap-4 p-4 rounded-xl border border-emerald-500/50 bg-emerald-500/10 hover:bg-emerald-500/20 transition-all group text-left disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="p-3 rounded-full bg-emerald-500 text-black group-hover:scale-110 transition-transform">
                  <Download size={24} />
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-emerald-400">Save Report & Video</h3>
                  <p className="text-xs text-emerald-500/70 mt-1 flex items-center gap-2">
                    <FileText size={12} /> Report <Film size={12} className="ml-2" /> MP4 Video
                  </p>
                </div>
              </button>

              {/* Option 2: Save Report Only */}
              <button
                disabled={isSaving}
                onClick={onSaveReportOnly}
                className="w-full flex items-center gap-4 p-4 rounded-xl border border-gray-800 bg-[#111a12] hover:bg-gray-900 hover:border-gray-700 transition-all group text-left disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="p-3 rounded-full bg-gray-800 text-gray-400 group-hover:text-white transition-colors">
                  <Save size={24} />
                </div>
                <div className="flex-1">
                  <h3 className="font-bold text-gray-300 group-hover:text-white">Save Report Only</h3>
                  <p className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                    <FileText size={12} /> Text data only
                  </p>
                </div>
              </button>

              <button 
                onClick={onClose}
                disabled={isSaving}
                className="w-full py-3 text-sm font-bold text-gray-500 hover:text-white uppercase tracking-widest disabled:opacity-50"
              >
                Skip for now
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
