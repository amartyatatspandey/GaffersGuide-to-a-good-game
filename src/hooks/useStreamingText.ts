import { useState, useEffect } from 'react';

export function useStreamingText(fullText: string, speedMs: number = 20) {
  const [displayedText, setDisplayedText] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!fullText) {
      setDisplayedText('');
      return;
    }

    setIsStreaming(true);
    setDisplayedText('');
    let currentIndex = 0;

    const intervalId = setInterval(() => {
      if (currentIndex < fullText.length - 1) {
        // Use functional state update to guarantee latest state length
        setDisplayedText(prev => prev + fullText[currentIndex]);
        currentIndex++;
      } else {
        // Last character
        setDisplayedText(prev => prev + fullText[currentIndex]);
        setIsStreaming(false);
        clearInterval(intervalId);
      }
    }, speedMs);

    return () => clearInterval(intervalId);
  }, [fullText, speedMs]);

  return { displayedText, isStreaming };
}
