import { NextRequest, NextResponse } from "next/server";
import { callVerificationService } from "@/app/lib/utils";

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const {
      answer,
      evidence_passages = [],
      claims,
      threshold_entailment = 0.60,
      threshold_contradiction = 0.50
    } = body;

    if (!answer && (!claims || claims.length === 0)) {
      return NextResponse.json(
        { error: "Either answer string or claims array is required." },
        { status: 400 }
      );
    }

    if (!evidence_passages || evidence_passages.length === 0) {
      return NextResponse.json(
        { error: "At least one evidence passage is required for verification." },
        { status: 400 }
      );
    }

    // Call Python Verification Microservice
    const verificationData = await callVerificationService(
      answer,
      evidence_passages,
      threshold_entailment,
      threshold_contradiction
    );

    return NextResponse.json({
      ...verificationData,
      orchestration_latency_ms: Date.now() - startTime
    });
  } catch (error: any) {
    console.error("Verification route error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to verify claims" },
      { status: 500 }
    );
  }
}
