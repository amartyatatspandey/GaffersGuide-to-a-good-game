"use client";

import React, { useState, useEffect, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabaseClient';
import { useAuth } from '@/components/AuthProvider';
import { Lock, Mail, AlertTriangle, ArrowRight, Loader2 } from 'lucide-react';

function LoginContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading: authLoading } = useAuth();
  
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // If already logged in, redirect to the next page
  useEffect(() => {
    if (!authLoading && user) {
      const nextParam = searchParams.get('next') || '/workspace';
      router.push(nextParam);
    }
  }, [user, authLoading, router, searchParams]);

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);

    if (!email || !password) {
      setError("Please fill in all fields.");
      setLoading(false);
      return;
    }

    try {
      if (isSignUp) {
        const { data, error: signUpErr } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: typeof window !== 'undefined' ? `${window.location.origin}/workspace` : undefined,
          }
        });

        if (signUpErr) throw signUpErr;

        // In Supabase, if email confirmation is enabled, user session won't be active immediately
        if (data.user && !data.session) {
          setSuccess("Account created! Please check your email for a confirmation link.");
        } else {
          setSuccess("Account created successfully!");
          const nextParam = searchParams.get('next') || '/workspace';
          router.push(nextParam);
        }
      } else {
        const { error: signInErr } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (signInErr) throw signInErr;

        setSuccess("Success! Initializing analysis engine...");
        const nextParam = searchParams.get('next') || '/workspace';
        router.push(nextParam);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An authentication error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setLoading(true);
    setError(null);
    try {
      const { error: oauthErr } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: typeof window !== 'undefined' ? `${window.location.origin}/workspace` : undefined,
        },
      });
      if (oauthErr) throw oauthErr;
    } catch (err: any) {
      console.error(err);
      setError(err.message || "OAuth redirection failed.");
      setLoading(false);
    }
  };

  if (authLoading) {
    return (
      <div className="min-h-screen bg-pitch flex flex-col items-center justify-center font-mono">
        <Loader2 className="animate-spin text-neon mb-4" size={40} />
        <p className="text-chalk/60 text-sm tracking-widest">VERIFYING AUTHENTICATION STATE...</p>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-pitch text-chalk flex flex-col justify-center items-center px-4 relative overflow-hidden font-mono select-none">
      {/* Visual background elements */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-neon/5 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[120px] pointer-events-none"></div>

      <div className="w-full max-w-md relative z-10">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-wider text-chalk">
            GAFFER'S <span className="text-neon">GUIDE</span>
          </h1>
          <p className="text-chalk/50 text-xs sm:text-sm uppercase tracking-widest mt-2">
            AI Tactical Intelligence Engine
          </p>
        </div>

        {/* Card wrapper */}
        <div className="bg-[#070b07]/90 border border-slate-dark/80 rounded-2xl p-6 sm:p-8 backdrop-blur-md shadow-[0_0_50px_rgba(0,255,102,0.03)] hover:border-neon/30 transition-all duration-500">
          <h2 className="text-base sm:text-lg font-bold text-chalk mb-6 border-b border-slate-dark/50 pb-3 flex items-center justify-between">
            <span>{isSignUp ? 'REGISTER_COACH' : 'INITIALIZE_SESSION'}</span>
            <span className="text-[9px] text-neon/60 bg-neon/10 px-2 py-0.5 rounded border border-neon/20">
              v1.0.0
            </span>
          </h2>

          {error && (
            <div className="mb-5 p-3 bg-red-950/40 border border-red-500/30 text-red-400 rounded-lg text-xs flex items-start gap-2.5 animate-pulse">
              <AlertTriangle size={16} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="mb-5 p-3 bg-neon/10 border border-neon/30 text-neon rounded-lg text-xs flex items-start gap-2.5">
              <ArrowRight size={16} className="shrink-0 mt-0.5" />
              <span>{success}</span>
            </div>
          )}

          <form onSubmit={handleEmailAuth} className="space-y-4">
            <div>
              <label className="block text-[10px] text-chalk/40 uppercase tracking-widest mb-1.5 font-bold">
                Email Address
              </label>
              <div className="relative">
                <Mail size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-chalk/35" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="coach@gaffersguide.com"
                  className="w-full bg-[#0b100b] border border-slate-dark text-chalk text-sm rounded-lg pl-10 pr-4 py-2.5 outline-none focus:border-neon/50 focus:ring-1 focus:ring-neon/30 transition-all font-sans"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-[10px] text-chalk/40 uppercase tracking-widest mb-1.5 font-bold">
                Password
              </label>
              <div className="relative">
                <Lock size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-chalk/35" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  className="w-full bg-[#0b100b] border border-slate-dark text-chalk text-sm rounded-lg pl-10 pr-4 py-2.5 outline-none focus:border-neon/50 focus:ring-1 focus:ring-neon/30 transition-all font-sans"
                  required
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 mt-2 bg-neon/10 hover:bg-neon/20 text-neon font-bold border border-neon/40 hover:border-neon rounded-lg transition-all text-xs flex items-center justify-center gap-2 cursor-pointer shadow-[0_0_15px_rgba(0,255,102,0.1)] active:scale-98"
            >
              {loading ? (
                <Loader2 className="animate-spin text-neon" size={16} />
              ) : isSignUp ? (
                'CREATE_COACH_ACCOUNT'
              ) : (
                'BOOT_ANALYSIS_WORKSPACE'
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-6 flex items-center">
            <div className="grow border-t border-slate-dark/30"></div>
            <span className="shrink mx-3 text-[9px] text-chalk/30 uppercase tracking-wider">OR</span>
            <div className="grow border-t border-slate-dark/30"></div>
          </div>

          {/* OAuth button */}
          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full py-2.5 bg-black hover:bg-zinc-900 text-chalk text-xs font-bold border border-zinc-800 rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer active:scale-98"
          >
            {/* Google Icon */}
            <svg className="w-4 h-4 shrink-0" viewBox="0 0 24 24">
              <path
                fill="#EA4335"
                d="M12 5.04c1.62 0 3.08.56 4.22 1.64l3.15-3.15C17.45 1.71 14.93 1 12 1 7.35 1 3.4 3.65 1.5 7.5l3.6 2.8C6.01 7.02 8.78 5.04 12 5.04z"
              />
              <path
                fill="#4285F4"
                d="M23.5 12.25c0-.82-.07-1.6-.21-2.35H12v4.45h6.45c-.28 1.48-1.12 2.73-2.37 3.58l3.67 2.85c2.14-1.98 3.75-4.88 3.75-8.53z"
              />
              <path
                fill="#FBBC05"
                d="M5.1 14.7c-.23-.7-.35-1.44-.35-2.2s.12-1.5.35-2.2l-3.6-2.8C.54 9.17 0 10.53 0 12s.54 2.83 1.5 4.5l3.6-2.8z"
              />
              <path
                fill="#34A853"
                d="M12 23c3.24 0 5.97-1.07 7.96-2.92l-3.67-2.85c-1.02.68-2.33 1.09-4.29 1.09-3.22 0-5.99-1.98-6.9-4.78l-3.6 2.8C3.4 20.35 7.35 23 12 23z"
              />
            </svg>
            SIGN_IN_WITH_GOOGLE
          </button>

          {/* Toggle link */}
          <div className="mt-6 text-center">
            <button
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-[10px] text-chalk/45 hover:text-neon transition-colors"
            >
              {isSignUp
                ? "Already have a license? Switch to SIGN_IN"
                : "New coach? Request license & REGISTER"}
            </button>
          </div>
        </div>
      </div>
    </main>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-pitch flex flex-col items-center justify-center font-mono">
        <Loader2 className="animate-spin text-neon mb-4" size={40} />
        <p className="text-chalk/60 text-sm tracking-widest">LOADING AUTH ENGINE...</p>
      </div>
    }>
      <LoginContent />
    </Suspense>
  );
}
