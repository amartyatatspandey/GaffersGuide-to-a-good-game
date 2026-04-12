"use client";
import React, { useState, useRef, useEffect } from 'react';
import { InsightCard, KeywordConfig } from './InsightCard';
import { VideoHUD } from './VideoHUD';
import { Clock, Send, Bot, User, Menu, ChevronLeft } from 'lucide-react';
import { useStreamingText } from '@/hooks/useStreamingText';

interface TeamInsight {
  payload: string;
  keywords: KeywordConfig[];
}

interface TimelineData {
  time: string;
  title: string;
  minute: string;
  summary: string;
  blueTeam: TeamInsight;
  redTeam: TeamInsight;
}

const TIMELINE_DATA: TimelineData[] = [
  {
    time: '00:00 - 15:00',
    title: 'Midfield Disconnect',
    minute: '12:45',
    summary: '1 Critical insights flagged',
    blueTeam: {
      payload: "The opponent (Red Team) has a massive gap between their defense and midfield. Deploy your striker as a False 9 to drop into this Zone 14 pocket. Instruct your left back to act as an Inverted Fullback, tucking into the central midfield to create a numerical overload.",
      keywords: [
        { text: "False 9", color: "emerald", role: "False 9" },
        { text: "Zone 14 pocket", color: "amber", role: "Zone 14" },
        { text: "Inverted Fullback", color: "emerald", role: "Inverted Fullback" },
        { text: "tucking into the central midfield", color: "cyan", role: "Inverted Fullback" }
      ]
    },
    redTeam: {
      payload: "Blue Team's defensive line is retreating too early, creating a huge gap. Exploit this by pushing our Attacking Midfielder higher to operate between the lines in Zone 14. Have our wingers stay wide to stretch their Inverted Fullbacks.",
      keywords: [
        { text: "Attacking Midfielder", color: "amber", role: "Attacking Mid" },
        { text: "Zone 14", color: "cyan", role: "Zone 14" },
        { text: "Inverted Fullbacks", color: "emerald", role: "Inverted Fullback" }
      ]
    }
  },
  {
    time: '15:00 - 30:00',
    title: 'High Press Exploited',
    minute: '22:15',
    summary: 'High risk transition vulnerability',
    blueTeam: {
      payload: "Our high defensive line is being bypassed by long diagonals. Shift the defensive block to a mid-block temporarily and instruct the holding midfielder to provide cover in the channels.",
      keywords: [
        { text: "mid-block", color: "amber", role: "Defensive Block" },
        { text: "holding midfielder", color: "emerald", role: "Holding Mid" },
        { text: "cover in the channels", color: "cyan", role: "Holding Mid" }
      ]
    },
    redTeam: {
      payload: "Blue team's high press is aggressive. Utilize long diagonals from our Center Backs directly to the wide areas. Keep our wingers high to pin back their fullbacks and exploit the space behind.",
      keywords: [
        { text: "long diagonals", color: "cyan", role: "Playmaker" },
        { text: "pin back their fullbacks", color: "amber", role: "Winger" }
      ]
    }
  },
  {
    time: '30:00 - 45+',
    title: 'Structural Stability',
    minute: '41:00',
    summary: 'No major anomalies structurally',
    blueTeam: {
      payload: "Structural adjustments have neutralized their transition game. Maintain current shape. Wingers should prioritize tracking back to prevent wide overloads before halftime.",
      keywords: []
    },
    redTeam: {
      payload: "Blue team has stabilized their defensive block. Introduce greater rotational movement in the final third to disrupt marking. Consider swapping wingers to change the angle of attack.",
      keywords: []
    }
  }
];

// Sub-component for a streaming AI chat bubble
function StreamingBubble({ text }: { text: string }) {
  const { displayedText } = useStreamingText(text, 25);

  const renderFormattedText = (raw: string) => {
    const parts = raw.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="font-bold text-gray-100">{part.slice(2, -2)}</strong>;
      }
      return <span key={i}>{part}</span>;
    });
  };

  return (
    <div className="flex gap-4">
      <div className="w-8 h-8 rounded-full bg-emerald-900/50 flex items-center justify-center flex-shrink-0 border border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
        <Bot size={16} className="text-emerald-400" />
      </div>
      <div className="flex-1 bg-gray-800/80 border border-gray-700/50 rounded-2xl rounded-tl-none p-4 text-sm text-gray-300 overflow-x-auto [&::-webkit-scrollbar]:h-1.5 [&::-webkit-scrollbar-thumb]:bg-gray-700">
        <div className="min-w-fit pr-2">
          {renderFormattedText(displayedText)}
          <span className="inline-block w-2 bg-emerald-500 animate-pulse h-4 align-middle ml-1 shadow-[0_0_5px_rgba(16,185,129,0.8)]"></span>
        </div>
      </div>
    </div>
  );
}

export function TacticalDashboard() {
  const [activeIndex, setActiveIndex] = useState(0);
  const [isTimelineCollapsed, setIsTimelineCollapsed] = useState(false);
  
  // Chat Interface State
  const [promptInput, setPromptInput] = useState('');
  const [chatHistory, setChatHistory] = useState<{ id: string, role: 'user' | 'ai', text: string }[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isTyping]);

  // Clear chat when timeline changes
  useEffect(() => {
    setChatHistory([]);
    setPromptInput('');
  }, [activeIndex]);

  const handleSendPrompt = (e: React.FormEvent) => {
    e.preventDefault();
    if (!promptInput.trim()) return;

    const newQuery = promptInput.trim();
    setPromptInput('');
    setChatHistory(prev => [...prev, { id: Date.now().toString(), role: 'user', text: newQuery }]);
    setIsTyping(true);

    // Mock LLM Inference Latency
    setTimeout(() => {
      setIsTyping(false);
      setChatHistory(prev => [...prev, { 
        id: (Date.now() + 1).toString(), 
        role: 'ai', 
        text: "Based on the **kinematic tracking data**, dropping a midfielder into the half-space will draw their center-back out of position. This enables the winger to exploit the **blindside run** natively. I recommend activating a **3-2-5 attacking structure**." 
      }]);
    }, 1200);
  };

  return (
    <div className="flex h-full w-full bg-[#0a0f0a] font-sans relative">
      {/* Left Panel: Match Timeline feed */}
      <div className={`${isTimelineCollapsed ? 'w-0 border-r-0 opacity-0' : 'w-[30%] border-r'} transition-all duration-300 ease-in-out border-gray-900 flex flex-col h-full bg-[#111a12]/30 flex-shrink-0 overflow-hidden whitespace-nowrap`}>
        <div className="p-4 border-b border-gray-900 flex items-center justify-between shadow-sm">
          <h2 className="text-xs font-bold text-gray-500 tracking-widest uppercase font-mono">Match Timeline</h2>
          <div className="flex gap-2 items-center">
             <Clock size={16} className="text-gray-600" />
             <button 
               onClick={() => setIsTimelineCollapsed(true)} 
               className="text-gray-500 hover:text-white transition-colors ml-1 p-1 hover:bg-gray-800 rounded">
                 <ChevronLeft size={16} />
             </button>
          </div>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-4 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800 [&::-webkit-scrollbar-track]:bg-transparent">
          {TIMELINE_DATA.map((data, i) => (
            <div 
              key={i} 
              onClick={() => setActiveIndex(i)}
              className={`p-4 border rounded-xl cursor-pointer transition-all ${
                i === activeIndex 
                  ? 'bg-[#111a12] border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.05)]' 
                  : 'bg-[#0a0f0a] border-gray-800 hover:bg-[#111a12]/80'
              }`}>
              <div className={`text-[10px] font-mono mb-1 tracking-widest ${i === activeIndex ? 'text-emerald-500' : 'text-gray-500'}`}>{data.time}</div>
              <div className="text-sm font-bold text-gray-300 font-sans tracking-tight">Phase Analysis Complete</div>
              <div className="text-xs text-gray-500 mt-2 font-sans truncate">{data.summary}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Right Panel: Advanced Insight Viewer & Conversational LLM Interface */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-[#050805]">
        
        {/* Top Half: Video Context */}
        <div className="h-[45%] border-b border-gray-900 p-6 flex flex-col justify-center items-center relative overflow-hidden bg-[#050805] flex-shrink-0">
           <div className="absolute top-4 left-4 text-[10px] font-bold text-gray-600 tracking-widest uppercase font-mono z-30 flex items-center gap-2">
             {isTimelineCollapsed && (
               <button 
                 onClick={() => setIsTimelineCollapsed(false)} 
                 className="p-1 mr-2 rounded hover:bg-gray-800 text-gray-500 hover:text-emerald-400 transition-colors"
               >
                 <Menu size={14} />
               </button>
             )}
             <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
             Live IPC Feed
           </div>
           
           <VideoHUD />
        </div>

        {/* Bottom Half: Telemetry Insight Viewer + Chat */}
        <div className="flex-1 flex flex-row relative h-[55%] bg-[#0a0f0a]">
          
          {/* Left Panel: 60% - Analysis Card */}
          <div className="w-[60%] border-r border-gray-900 flex flex-col relative overflow-hidden bg-[#050805]">
             <div className="flex-1 overflow-y-auto p-8 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800">
               <div className="max-w-3xl mx-auto w-full space-y-6">
                 <div className="text-[10px] font-bold text-gray-500 tracking-widest flex justify-between items-center uppercase font-mono">
                    <span>Proactive Insight Generation</span>
                    <span className="text-emerald-500">Telemetry Synced</span>
                 </div>
                 
                 <InsightCard 
                   key={activeIndex} 
                   title={TIMELINE_DATA[activeIndex].title}
                   minute={TIMELINE_DATA[activeIndex].minute}
                   blueTeam={TIMELINE_DATA[activeIndex].blueTeam}
                   redTeam={TIMELINE_DATA[activeIndex].redTeam}
                 />
               </div>
             </div>
          </div>

          {/* Right Panel: 40% - Gemini Chat Interface */}
          <div className="w-[40%] flex flex-col relative bg-[#0a0f0a]">
            {/* Scrollable Chat Feed */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 pb-24 space-y-6 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:bg-gray-800">
               
               {/* Empty State / Initial Greeting */}
               {chatHistory.length === 0 && (
                 <div className="flex flex-col items-center justify-center h-full text-center opacity-70 space-y-4 mt-8">
                    <div className="w-12 h-12 rounded-full bg-emerald-900/30 flex items-center justify-center border border-emerald-500/30">
                       <Bot size={24} className="text-emerald-500" />
                    </div>
                    <div className="space-y-2">
                       <p className="text-sm font-semibold text-gray-200 font-sans">Tactical Engine Ready</p>
                       <p className="text-xs text-gray-500 max-w-[250px] mx-auto font-sans leading-relaxed">I've analyzed the telemetry. Ask me for tactical variations or player-specific follow-ups.</p>
                    </div>
                 </div>
               )}
               
               {/* Chat Messages */}
               {chatHistory.map((msg) => (
                 msg.role === 'user' ? (
                   <div key={msg.id} className="flex gap-3 justify-end animate-fade-in-up">
                     <div className="max-w-[85%] bg-[#111a12] border border-gray-800 rounded-2xl rounded-tr-none p-3.5 text-sm text-gray-200 font-sans shadow-sm">
                       {msg.text}
                     </div>
                     <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center flex-shrink-0 border border-gray-700">
                       <User size={14} className="text-gray-400" />
                     </div>
                   </div>
                 ) : (
                   <div key={msg.id} className="animate-fade-in-up">
                     <StreamingBubble text={msg.text} />
                   </div>
                 )
               ))}
               
               {/* Typing Indicator */}
               {isTyping && (
                 <div className="flex gap-3 animate-fade-in">
                   <div className="w-8 h-8 rounded-full bg-emerald-900/50 flex items-center justify-center flex-shrink-0 border border-emerald-500/50 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                     <Bot size={14} className="text-emerald-400 animate-pulse" />
                   </div>
                   <div className="bg-gray-800/80 border border-gray-700/50 rounded-2xl rounded-tl-none p-4 text-sm flex items-center gap-1">
                     <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"></span>
                     <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                     <span className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                   </div>
                 </div>
               )}
            </div>

            {/* Chat Input (Gemini Style) */}
            <div className="absolute bottom-0 left-0 right-0 p-5 bg-gradient-to-t from-[#0a0f0a] via-[#0a0f0a] to-transparent">
               <form onSubmit={handleSendPrompt} className="relative flex items-center max-w-lg mx-auto w-full shadow-[0_10px_40px_rgba(0,0,0,0.8)]">
                 <input 
                   type="text" 
                   value={promptInput}
                   onChange={(e) => setPromptInput(e.target.value)}
                   placeholder="Ask a tactical follow-up..."
                   className="w-full bg-[#111a12] border border-gray-800 rounded-[24px] pl-6 pr-14 py-4 text-sm text-gray-200 focus:outline-none focus:border-emerald-500/50 focus:bg-[#1a241c] hover:bg-[#1a241c] transition-all font-sans placeholder-gray-600 shadow-inner"
                 />
                 <button 
                   type="submit"
                   disabled={!promptInput.trim() || isTyping}
                   className="absolute right-2 top-1/2 -translate-y-1/2 p-2.5 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500 hover:text-black rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                 >
                   <Send size={16} className={`transition-transform ${promptInput.trim() && !isTyping ? "group-hover:scale-110" : ""}`} />
                 </button>
               </form>
               <div className="text-center mt-3 text-[10px] text-gray-600 font-sans tracking-wide">
                  Gaffer's Engine can make mistakes. Verify telemetry data.
               </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
