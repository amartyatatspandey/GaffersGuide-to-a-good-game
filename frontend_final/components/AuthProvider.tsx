"use client";

import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabaseClient';
import type { User, Session } from '@supabase/supabase-js';

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  session: null,
  loading: true,
  signOut: async () => {},
});

// ── Dev bypass ────────────────────────────────────────────────────────────────
// Set NEXT_PUBLIC_BYPASS_AUTH=true in .env.local to skip Supabase auth.
const BYPASS_AUTH = process.env.NEXT_PUBLIC_BYPASS_AUTH === 'true';

const MOCK_USER = {
  id: 'local-dev-user',
  email: 'dev@localhost',
  aud: 'authenticated',
  role: 'authenticated',
  created_at: new Date().toISOString(),
  app_metadata: {},
  user_metadata: { name: 'Local Dev' },
  identities: [],
  factors: [],
} as unknown as User;
// ─────────────────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(BYPASS_AUTH ? MOCK_USER : null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(!BYPASS_AUTH);

  useEffect(() => {
    if (BYPASS_AUTH) return; // Skip Supabase entirely in bypass mode

    // 1. Get initial session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.access_token) {
        localStorage.setItem('gaffer-supabase-token', session.access_token);
      } else {
        localStorage.removeItem('gaffer-supabase-token');
      }
      setLoading(false);
    });

    // 2. Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      if (session?.access_token) {
        localStorage.setItem('gaffer-supabase-token', session.access_token);
      } else {
        localStorage.removeItem('gaffer-supabase-token');
        localStorage.removeItem('gg-profile-name'); // Clear local cached profile name if signed out
      }
      setLoading(false);
    });

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  const signOut = async () => {
    if (BYPASS_AUTH) return;
    setLoading(true);
    try {
      await supabase.auth.signOut();
    } catch (err) {
      console.error("Error during sign out:", err);
    } finally {
      setUser(null);
      setSession(null);
      localStorage.removeItem('gaffer-supabase-token');
      localStorage.removeItem('gg-profile-name');
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider value={{ user, session, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
