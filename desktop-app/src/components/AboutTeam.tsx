"use client";

import React from "react";
import { motion } from "framer-motion";

const teamMembers = [
  {
    name: "A. Turing",
    role: "Lead Math Architect",
    initials: "AT",
    color: "from-cyan-500 to-blue-500",
  },
  {
    name: "J. Carmack",
    role: "Engine Optimization",
    initials: "JC",
    color: "from-emerald-500 to-teal-500",
  },
  {
    name: "K. He",
    role: "Computer Vision Lead",
    initials: "KH",
    color: "from-purple-500 to-indigo-500",
  },
  {
    name: "S. Guardiola",
    role: "Tactical Domain Expert",
    initials: "SG",
    color: "from-amber-500 to-orange-500",
  },
];

export function AboutTeam() {
  return (
    <section className="relative py-24 bg-gray-950 border-t border-white/5">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-4xl md:text-5xl font-bold tracking-tight mb-4"
          >
            Math meets the <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">Beautiful Game.</span>
          </motion.h2>
          <p className="text-xl text-gray-400 font-light max-w-2xl mx-auto">
            Built by engineers from elite quantitative trading firms and tactical analysts from top-flight European clubs.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1 }}
              className="glass-card p-6 border border-white/10 hover:border-emerald-500/30 transition-colors group"
            >
              {/* Abstract Stylized Avatar */}
              <div className={`w-16 h-16 rounded-2xl mb-6 bg-gradient-to-br ${member.color} p-[1px]`}>
                <div className="w-full h-full bg-gray-900 rounded-2xl flex items-center justify-center relative overflow-hidden">
                  <span className="text-xl font-bold text-white relative z-10">{member.initials}</span>
                  {/* Abstract backdrop blur inside avatar */}
                  <div className={`absolute inset-0 bg-gradient-to-tr ${member.color} opacity-20 blur-md`} />
                </div>
              </div>

              <h3 className="text-xl font-bold text-white mb-1 group-hover:text-emerald-400 transition-colors">
                {member.name}
              </h3>
              <p className="text-sm text-gray-400 mb-6">{member.role}</p>

              <a 
                href="#" 
                className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors border border-white/5"
                title={`LinkedIn Profile of ${member.name}`}
              >
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                  <rect x="2" y="9" width="4" height="12"></rect>
                  <circle cx="4" cy="4" r="2"></circle>
                </svg>
              </a>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
