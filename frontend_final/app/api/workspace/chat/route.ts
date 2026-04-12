import { NextResponse } from "next/server";

const DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434";
const DEFAULT_MODEL = "llama3";

function sanitizeModel(raw: unknown): string {
  if (typeof raw !== "string" || raw.length === 0) return DEFAULT_MODEL;
  const trimmed = raw.trim().slice(0, 80);
  if (!/^[a-zA-Z0-9._:-]+$/.test(trimmed)) return DEFAULT_MODEL;
  return trimmed;
}

export async function POST(req: Request): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const b = body as { messages?: unknown; model?: unknown };
  const messages = Array.isArray(b.messages) ? b.messages : [];
  const model = sanitizeModel(b.model);

  const baseRaw = process.env.OLLAMA_BASE_URL ?? DEFAULT_OLLAMA_BASE;
  const base = baseRaw.replace(/\/$/, "");

  try {
    const res = await fetch(`${base}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages,
        stream: false,
      }),
    });

    const rawText = await res.text();
    if (!res.ok) {
      return NextResponse.json(
        {
          error: "ollama_http_error",
          status: res.status,
          detail: rawText.slice(0, 800),
        },
        { status: 502 },
      );
    }

    let parsed: { message?: { content?: string } };
    try {
      parsed = JSON.parse(rawText) as { message?: { content?: string } };
    } catch {
      return NextResponse.json(
        { error: "ollama_bad_json", detail: rawText.slice(0, 200) },
        { status: 502 },
      );
    }

    const content = parsed.message?.content ?? "";
    return NextResponse.json({ content, model });
  } catch (err) {
    return NextResponse.json(
      { error: "ollama_unreachable", detail: String(err) },
      { status: 503 },
    );
  }
}
