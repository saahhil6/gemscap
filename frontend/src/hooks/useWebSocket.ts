import { useState, useEffect, useRef, useCallback } from 'react';

export default function useWebSocket(url: string) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    const handleOpen = () => setConnected(true);
    const handleClose = () => setConnected(false);
    const handleMessage = (ev: MessageEvent) => setLastMessage(ev.data);

    ws.addEventListener('open', handleOpen);
    ws.addEventListener('close', handleClose);
    ws.addEventListener('message', handleMessage);

    return () => {
      ws.removeEventListener('open', handleOpen);
      ws.removeEventListener('close', handleClose);
      ws.removeEventListener('message', handleMessage);
      ws.close();
      wsRef.current = null;
    };
  }, [url]);

  const send = useCallback((data: any) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    const payload = typeof data === 'string' ? data : JSON.stringify(data);
    ws.send(payload);
    return true;
  }, []);

  const close = useCallback(() => {
    wsRef.current?.close();
  }, []);

  return { connected, lastMessage, send, close } as const;
}

