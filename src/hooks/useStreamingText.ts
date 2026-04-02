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
      currentIndex++;
      setDisplayedText(fullText.substring(0, currentIndex));
      if (currentIndex >= fullText.length) {
        setIsStreaming(false);
        clearInterval(intervalId);
      }
    }, speedMs);

    return () => clearInterval(intervalId);
  }, [fullText, speedMs]);

  return { displayedText, isStreaming };
}
