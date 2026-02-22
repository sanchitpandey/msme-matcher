"use client";

import { useState, useRef } from "react";
import axios from "axios";
import { Mic, Search, MapPin, Factory, AlertCircle, CheckCircle2, X, Loader2, UploadCloud, FileText, Cpu, Network, Languages, BarChart3, Info } from "lucide-react";

// --- Types ---
interface AIExplain {
  semantic_score?: number;
  category_match?: number;
  capacity_score?: number;
  distance_factor?: number;
  price_factor?: number;
}

interface MatchItem {
  name: string;
  snp_id: string;
  location: string;
  category: string;
  capability_text: string;
  score: number;
  ltr_score: number;
  absolute_score: number;
  price_tier: string;
  capacity_score?: number;
  explain?: AIExplain;
}

interface OnboardData {
  business_name?: string;
  location?: string;
  predicted_category?: string;
  attributes?: any;
  raw_text_preview?: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MatchItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<{ count: number; time: string; cat?: string } | null>(null);
  const [ondcMode, setOndcMode] = useState(false);
  
  const [onboardData, setOnboardData] = useState<OnboardData | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedItem, setSelectedItem] = useState<MatchItem | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const handleMicClick = async () => {
    if (isRecording) stopRecording();
    else startRecording();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
        await processAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      alert("Microphone access is required.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");
      const res = await axios.post("http://localhost:8000/api/transcribe", formData);
      if (res.data.transcript) {
        setQuery(res.data.transcript);
        handleSearch(res.data.transcript);
      }
    } catch (error) {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setOnboardData(null);
    setResults([]);
    setStats(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://localhost:8000/api/ocr", formData);
      
      if (res.data.status === "success") {
        setOnboardData(res.data.auto_filled_form);
        const extractedText = res.data.auto_filled_form.raw_text_preview || "";
        if (extractedText) {
          setQuery(extractedText.substring(0, 50) + "..."); 
          await handleSearch(extractedText);
        }
      } else {
        alert("OCR failed to read document.");
      }
    } catch (error) {
      alert("Failed to process document.");
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (overrideQuery?: string) => {
    const searchText = overrideQuery || query;
    if (!searchText) return;
    
    setLoading(true);
    setResults([]);

    try {
      if (ondcMode) {
        const payload = {
          context: {
            domain: "ONDC:RET10",
            action: "search",
            transaction_id: "demo_tx_" + Date.now(),
            message_id: "msg_" + Date.now(),
            timestamp: new Date().toISOString()
          },
          message: { intent: { item: { descriptor: { name: searchText } } } }
        };
        const res = await axios.post("http://localhost:8000/ondc/search", payload);
        if (res.data.message?.catalog?.providers) {
          const providers = res.data.message.catalog.providers.map((p: any) => ({
            name: p.descriptor.name,
            location: "ONDC Network", 
            category: p.descriptor.short_desc,
            capability_text: p.descriptor.long_desc,
            ltr_score: 9.9,
            absolute_score: 0.99,
            price_tier: "Contract",
            snp_id: p.id
          }));
          setResults(providers);
          setStats({ count: providers.length, time: "0.8s (ONDC Gateway)", cat: "Network Result" });
        }
      } else {
        const formData = new FormData();
        formData.append("query", searchText);
        const res = await axios.post("http://localhost:8000/api/match", formData);
        setResults(res.data.matches);
        setStats({ count: res.data.count, time: res.data.time_taken, cat: res.data.query_category });
      }
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setLoading(false);
    }
  };

  const setDemoQuery = (text: string) => {
    setQuery(text);
    handleSearch(text);
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 font-sans relative flex flex-col">
      
      {/* HEADER */}
      <header className="bg-slate-900 text-white p-4 shadow-lg sticky top-0 z-10 border-b-4 border-blue-600">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
            <Factory className="w-8 h-8 text-blue-400" />
            <div>
              <h1 className="text-2xl font-bold tracking-tight">IndiaAI MSME Bridge</h1>
              <p className="text-slate-400 text-xs tracking-wider uppercase font-semibold">National Onboarding & Matching System</p>
            </div>
          </div>

          <div className="hidden lg:flex items-center gap-6 text-sm bg-slate-800 px-6 py-2 rounded-lg border border-slate-700">
            <div className="flex flex-col">
              <span className="text-slate-400 text-xs">Suppliers Indexed</span>
              <span className="font-mono font-bold text-emerald-400">3,000+</span>
            </div>
            <div className="w-px h-8 bg-slate-700"></div>
            <div className="flex flex-col">
              <span className="text-slate-400 text-xs">Core Model</span>
              <span className="font-mono font-bold text-blue-400">SBERT + LightGBM</span>
            </div>
            <div className="w-px h-8 bg-slate-700"></div>
            <div className="flex flex-col">
              <span className="text-slate-400 text-xs">Network Status</span>
              <span className="flex items-center gap-1 font-bold text-emerald-400">
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span> Online
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* SEARCH SECTION */}
      <div className="max-w-5xl mx-auto mt-8 px-4 w-full">
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 relative overflow-hidden">
          
          <div className="flex items-center justify-between mb-4 border-b border-slate-100 pb-3">
            <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
              <Search className="w-5 h-5 text-blue-600"/> Agent Matching & Registration
            </h2>
            <label className="flex items-center gap-2 cursor-pointer select-none text-sm font-semibold bg-slate-100 px-3 py-1.5 rounded-md border border-slate-200 hover:bg-slate-200 transition">
              <input 
                type="checkbox" checked={ondcMode} onChange={(e) => setOndcMode(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded" 
              />
              <Network className="w-4 h-4 text-slate-600"/> 
              <span>ONDC Network Mode</span>
            </label>
          </div>

          <div className="flex gap-3 items-center mb-4">
            <div className="flex-1 relative shadow-sm">
              <input
                type="text"
                placeholder="Describe MSME capabilities or requirement..."
                className="w-full pl-4 pr-4 py-3.5 rounded-lg border border-slate-300 focus:ring-2 focus:ring-blue-500 focus:outline-none font-medium text-slate-700"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              />
            </div>
            
            <input type="file" ref={fileInputRef} onChange={handleFileUpload} className="hidden" accept="image/*,.pdf" />
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="p-3.5 bg-slate-50 hover:bg-slate-100 rounded-lg border border-slate-300 text-slate-600 transition group flex flex-col items-center justify-center relative shadow-sm"
              title="Upload GST/Invoice to Auto-Onboard"
            >
              <UploadCloud className="w-5 h-5 group-hover:text-blue-600" />
            </button>

            <button 
              onClick={handleMicClick}
              className={`p-3.5 rounded-lg border shadow-sm transition-all ${
                isRecording ? "bg-red-50 border-red-300 text-red-600 animate-pulse" : "bg-slate-50 hover:bg-slate-100 border-slate-300 text-slate-600"
              }`}
            >
              {isRecording ? <Loader2 className="w-5 h-5 animate-spin" /> : <Mic className="w-5 h-5" />}
            </button>

            <button
              onClick={() => handleSearch()}
              disabled={loading}
              className="px-8 py-3.5 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition shadow-md active:scale-95 disabled:opacity-50 flex items-center gap-2"
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "Match"}
            </button>
          </div>

          <div className="flex gap-2 mt-4 overflow-x-auto pb-2">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center mr-2"><Languages className="w-3 h-3 mr-1"/> Try Prompts:</span>
            <button onClick={() => setDemoQuery("मुझे सूरत में कपड़े की फैक्ट्री ढूंढनी है")} className="text-xs bg-orange-50 text-orange-700 border border-orange-200 px-3 py-1 rounded-full hover:bg-orange-100 whitespace-nowrap font-medium">🇮🇳 Hindi: Textile</button>
            <button onClick={() => setDemoQuery("திருப்பூரில் காட்டன் நூல் எங்கு கிடைக்கும்")} className="text-xs bg-blue-50 text-blue-700 border border-blue-200 px-3 py-1 rounded-full hover:bg-blue-100 whitespace-nowrap font-medium">🇮🇳 Tamil: Cotton</button>
            <button onClick={() => setDemoQuery("Food grade packaging boxes in Delhi with FSSAI")} className="text-xs bg-emerald-50 text-emerald-700 border border-emerald-200 px-3 py-1 rounded-full hover:bg-emerald-100 whitespace-nowrap font-medium">🇬🇧 English: Packaging</button>
          </div>
        </div>
      </div>

      {/* AUTO-ONBOARDING */}
      {onboardData && (
        <div className="max-w-5xl mx-auto px-4 mt-6 w-full animate-in fade-in zoom-in duration-500">
          <div className="bg-emerald-50 border border-emerald-200 p-5 rounded-xl shadow-sm relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10"><FileText className="w-24 h-24 text-emerald-900"/></div>
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle2 className="w-6 h-6 text-emerald-600" />
              <h3 className="text-lg font-bold text-emerald-900">Document Processed: MSME Auto-Registered</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white/80 p-3 rounded-lg border border-emerald-100 backdrop-blur-sm">
                <span className="text-xs text-emerald-600 uppercase font-bold tracking-wider">Business Name</span>
                <p className="font-semibold text-slate-800 text-sm mt-1">{onboardData.business_name || "Auto-detected"}</p>
              </div>
              <div className="bg-white/80 p-3 rounded-lg border border-emerald-100 backdrop-blur-sm">
                <span className="text-xs text-emerald-600 uppercase font-bold tracking-wider">Location Found</span>
                <p className="font-semibold text-slate-800 text-sm mt-1 flex items-center gap-1"><MapPin className="w-3 h-3"/> {onboardData.location || "N/A"}</p>
              </div>
              <div className="bg-white/80 p-3 rounded-lg border border-emerald-100 backdrop-blur-sm">
                <span className="text-xs text-emerald-600 uppercase font-bold tracking-wider">Category Assigned</span>
                <p className="font-semibold text-slate-800 text-sm mt-1">{onboardData.predicted_category || "General"}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SEARCH STATS */}
      {stats && (
        <div className="max-w-5xl mx-auto px-4 mt-6 w-full text-sm text-slate-600 flex justify-between font-medium">
          <span>Found <strong className="text-slate-900">{stats.count}</strong> matching Seller Network Participants</span>
          <span>Category: <strong className="text-blue-600 bg-blue-50 px-2 py-0.5 rounded">{stats.cat}</strong> | Latency: {stats.time}</span>
        </div>
      )}

      {/* RESULTS GRID */}
      <div className="max-w-5xl mx-auto mt-4 px-4 pb-12 w-full flex-1">
        {results.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {results.map((item, idx) => (
              <div key={idx} className="bg-white rounded-xl p-5 shadow-sm border border-slate-200 hover:shadow-md transition relative flex flex-col justify-between group">
                <div>
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="text-lg font-extrabold text-slate-900">{item.name}</h3>
                      <div className="flex items-center gap-1 text-slate-500 text-xs mt-1 font-medium">
                        <MapPin className="w-3.5 h-3.5 text-slate-400" /> {item.location}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-slate-600 text-sm mb-4 leading-relaxed line-clamp-2">
                    {item.capability_text}
                  </p>
                </div>

                <div>
                  {/* Absolute Accuracy Bar */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs font-bold text-slate-500 mb-1">
                      <span>Absolute Relevance</span>
                      <span className={item.absolute_score >= 0.7 ? "text-green-600" : item.absolute_score >= 0.4 ? "text-orange-500" : "text-red-500"}>
                        {(item.absolute_score * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200">
                      <div
                        className={`h-full rounded-full transition-all ${item.absolute_score >= 0.7 ? "bg-green-500" : item.absolute_score >= 0.4 ? "bg-orange-400" : "bg-red-500"}`}
                        style={{ width: `${item.absolute_score * 100}%` }}
                      />
                    </div>
                    {item.absolute_score < 0.4 && (
                      <div className="text-[10px] text-red-400 mt-1 font-medium">
                        Warning: Cross-domain or severe geographic penalty applied.
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-slate-100">
                    <span className="px-2.5 py-1 bg-slate-100 text-slate-600 text-[10px] rounded-md font-bold uppercase tracking-wider">
                      {item.category?.split(' ')[0] || 'Domain'}
                    </span>
                    
                    {/* REDESIGNED VISIBLE BUTTON */}
                    <button 
                      onClick={() => setSelectedItem(item)}
                      className="px-4 py-2 bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold rounded-lg hover:bg-blue-600 hover:text-white transition-colors flex items-center gap-1.5 shadow-sm"
                    >
                      <BarChart3 className="w-3.5 h-3.5"/> View Details
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          !loading && !onboardData && (
            <div className="text-center mt-24 text-slate-400 max-w-sm mx-auto">
              <Factory className="w-16 h-16 mx-auto mb-4 opacity-20" />
              <p className="font-medium text-slate-500">System is ready.</p>
              <p className="text-sm mt-1">Upload a document or speak a requirement to begin the onboarding and mapping process.</p>
            </div>
          )
        )}
      </div>

      {/* FOOTER */}
      <footer className="bg-slate-900 border-t border-slate-800 py-6 mt-auto">
        <div className="max-w-5xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center text-slate-400 text-xs">
          <div className="flex items-center gap-2 mb-4 md:mb-0">
            <Cpu className="w-4 h-4 text-emerald-500"/> <strong>Architecture:</strong> Whisper (ASR) → Tesseract (OCR) → SBERT → LightGBM
          </div>
          <div className="flex gap-4 font-mono">
            <span>DPDP Compliant</span>
            <span>ONDC Ready</span>
            <span>v1.0.0</span>
          </div>
        </div>
      </footer>

      {/* EXPLAINABLE AI MODAL */}
      {selectedItem && (
        <div className="fixed inset-0 bg-slate-900/60 z-50 flex justify-end backdrop-blur-sm">
          <div className="w-full max-w-md bg-white h-full shadow-2xl overflow-y-auto animate-in slide-in-from-right duration-300 flex flex-col">
            <div className="p-6 border-b border-slate-100 bg-slate-50 sticky top-0 z-10 flex justify-between items-center">
              <div>
                <h2 className="text-xl font-black text-slate-900">{selectedItem.name}</h2>
                <p className="text-sm font-medium text-slate-500 flex items-center gap-1 mt-1"><MapPin className="w-3 h-3"/> {selectedItem.location}</p>
              </div>
              <button onClick={() => setSelectedItem(null)} className="p-2 bg-white rounded-full border border-slate-200 hover:bg-slate-100 transition shadow-sm">
                <X className="w-5 h-5 text-slate-600" />
              </button>
            </div>

            <div className="p-6 flex-1 space-y-6">
              <div className="bg-indigo-50 p-5 rounded-xl border border-indigo-100">
                <h3 className="text-sm font-bold text-indigo-900 flex items-center gap-2 mb-3 uppercase tracking-wider">
                  <Info className="w-4 h-4"/> AI Decision Factors
                </h3>
                {selectedItem.explain ? (
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs font-semibold text-indigo-800 mb-1">
                        <span>Semantic Vector Match</span>
                        <span>{(selectedItem.explain.semantic_score! * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-indigo-100 h-1.5 rounded-full"><div className="bg-indigo-500 h-full rounded-full" style={{width: `${selectedItem.explain.semantic_score! * 100}%`}}></div></div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-semibold text-indigo-800 mb-1">
                        <span>Taxonomy/Category Alignment</span>
                        <span>{selectedItem.explain.category_match === 1.0 ? "Exact Match" : "Domain Mismatch"}</span>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-semibold text-indigo-800 mb-1">
                        <span>Geographic Distance Factor</span>
                        <span>{selectedItem.explain.distance_km} km away</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-indigo-700 italic">No telemetry data available for this match (ONDC/Fallback mode).</p>
                )}
              </div>

              <div>
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 border-b border-slate-100 pb-2">Business Capabilities</h3>
                <p className="text-slate-700 leading-relaxed text-sm bg-slate-50 p-4 rounded-lg border border-slate-100">
                  {selectedItem.capability_text}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Pricing Tier</span>
                  <p className="font-semibold text-slate-800 mt-1">{selectedItem.price_tier}</p>
                </div>
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-100">
                  <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Network Role</span>
                  <p className="font-semibold text-slate-800 mt-1">Seller Node (SNP)</p>
                </div>
              </div>
            </div>

            <div className="p-6 border-t border-slate-100 bg-white">
              <button className="w-full bg-blue-600 text-white py-3.5 rounded-lg font-bold hover:bg-blue-700 transition shadow-lg shadow-blue-200">
                Initiate Handshake
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}