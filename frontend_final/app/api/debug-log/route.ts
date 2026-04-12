import { appendFile, mkdir } from "fs/promises";
import path from "path";
import { NextRequest, NextResponse } from "next/server";

function resolveAgentDebugLogPath(): string {
  const explicit = process.env.AGENT_DEBUG_LOG_ABSOLUTE?.trim();
  if (explicit) {
    return explicit;
  }
  const cwd = process.cwd();
  if (path.basename(cwd) === "frontend_final") {
    return path.join(cwd, "..", ".cursor", "debug-bb63ae.log");
  }
  return path.join(cwd, ".cursor", "debug-bb63ae.log");
}

/** Append one NDJSON line (disabled in production except with AGENT_DEBUG_LOG_ABSOLUTE). */
export async function POST(req: NextRequest): Promise<NextResponse> {
  if (
    process.env.NODE_ENV === "production" &&
    !process.env.AGENT_DEBUG_LOG_ABSOLUTE?.trim()
  ) {
    return new NextResponse(null, { status: 404 });
  }
  const raw = await req.text();
  if (!raw.trim()) {
    return NextResponse.json({ ok: false }, { status: 400 });
  }
  const line = raw.endsWith("\n") ? raw : `${raw}\n`;
  const primary = resolveAgentDebugLogPath();
  const paths = [primary];
  if (path.basename(process.cwd()) === "frontend_final") {
    paths.push(path.join(process.cwd(), ".cursor", "debug-bb63ae.log"));
  }
  for (const logPath of paths) {
    await mkdir(path.dirname(logPath), { recursive: true });
    await appendFile(logPath, line, "utf8");
  }
  return NextResponse.json({ ok: true });
}
