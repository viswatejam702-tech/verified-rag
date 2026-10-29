import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "HEALTHY",
    service: "verified-rag",
    version: "1.0.0",
    timestamp: new Date().toISOString()
  });
}

