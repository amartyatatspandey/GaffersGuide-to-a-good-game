"use client";

import React, { useRef } from "react";
import { motion, useInView } from "framer-motion";

interface TextRevealProps {
  text: string;
  className?: string;
  style?: React.CSSProperties;
  delay?: number;
}

export function TextReveal({ text, className = "", style, delay = 0 }: TextRevealProps) {
  const ref = useRef<HTMLDivElement>(null);
  // Trigger animation when the component scrolls into view
  const isInView = useInView(ref, { once: true, margin: "-10% 0px" });

  // Split the text into words to preserve layout/wrapping logic,
  // then break the words down into individual characters.
  const words = text.split(" ");

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        delayChildren: delay,
        staggerChildren: 0.05, // 0.05-second delay between each letter
      },
    },
  };

  const charVariants = {
    hidden: {
      y: 50, // Pushed down by 50px
      opacity: 0,
    },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.7,
        ease: [0.16, 1, 0.3, 1] as const,
      },
    },
  };

  return (
    <motion.div
      ref={ref}
      aria-label={text}
      role="text"
      className={`flex flex-wrap ${className}`}
      style={style}
      variants={containerVariants}
      initial="hidden"
      animate={isInView ? "visible" : "hidden"}
    >
      {words.map((word, wordIndex) => (
        // Word wrapper guarantees spaces are maintained correctly via standard CSS margin
        <span key={wordIndex} className="inline-flex whitespace-nowrap mr-[0.25em]">
          {word.split("").map((char, charIndex) => (
            // Overflow: hidden container perfectly clips the text when it is at translateY(100%)
            <span key={charIndex} className="inline-block overflow-hidden relative">
              <motion.span
                variants={charVariants}
                className="inline-block origin-bottom"
              >
                {char}
              </motion.span>
            </span>
          ))}
        </span>
      ))}
    </motion.div>
  );
}
