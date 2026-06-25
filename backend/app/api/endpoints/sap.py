from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any
import os
import itertools
import json
from groq import Groq

router = APIRouter()

# --- Key Rotation ---
api_keys = []
for i in range(1, 7):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        api_keys.append((key, f"GROQ_API_KEY_{i}"))

if not api_keys:
    print("Warning: No GROQ_API_KEY_1...6 found for SAP endpoints.")

key_cycle = itertools.cycle(api_keys) if api_keys else iter([])


# --- System Prompts ---

PERPERSONAL_SYSTEM_PROMPT = """You are an HR email writer. You will receive a combined JSON object
that may contain data from one or more SAP SuccessFactors API responses,
passed as separate named fields. Each field can be a string, object, or array —
treat all of them together as the full employee data context.

Across all fields, look for these values — here is what matters to you:
- salutation: Mr./Ms./Dr. etc — use for greeting
- preferredName: employee's preferred first name — USE THIS over firstName if available
- firstName: employee's first name — fallback if preferredName is null
- lastName: employee's last name
- displayName: full name — use only if both firstName and lastName are null
- middleName: middle name — ignore this
- gender, maritalStatus, nationality — IGNORE these entirely
- createdBy, createdDateTime, lastModifiedBy, operation,
  script, attachmentId, personIdExternal — IGNORE all of these
- Any field names not listed above — IGNORE entirely

Your job is ONLY to write the OPENING of a professional HR email.

Output EXACTLY:
- Line 1: "Hi [salutation] [preferredName or firstName] [lastName],"
- Line 2: empty line
- Line 3: One warm professional paragraph (2-3 sentences)
  welcoming or acknowledging the employee personally

STRICT RULES:
- Do NOT write closing, sign off, or regards
- Do NOT write a subject line
- Do NOT mention dates, process, or manager
- Do NOT ask questions or add commentary
- Do NOT mention any fields you were told to ignore
- Do NOT reference field names or API names in the output
- If name fields are null across all inputs, use displayName
- Output ONLY greeting line + one paragraph, nothing else"""

ONB2PROCESS_SYSTEM_PROMPT = """You are an HR email writer. You will receive a combined JSON object
that may contain data from one or more SAP SuccessFactors API responses,
passed as separate named fields. Each field can be a string, object, or array —
treat all of them together as the full process context for this employee.

Across all fields, look for these values — here is what matters to you:
- processType: "ONB" = onboarding (new joiner), "OFB" = offboarding (exit)
- startDate: employee's first working day — use ONLY for ONB
- endDate: employee's last working day — use ONLY for OFB
- onboardingInternalHire: true = internal transfer, false = external new hire
- manager: reporting manager's name — mention only if not null
- onboardingHireStatus: current status — ignore this
- processStatus: ignore this
- processVariant, processTrigger, targetSystem — IGNORE all
- cancelEventReason, cancelOffboardingReason, cancelOnboardingReason,
  cancellationComment, cancellationDate — IGNORE all
- activitiesConfig, offboardingActivitiesConfig, customDataCollectionConfig,
  bpeProcessInstanceId, onb2MasterId, mdfSystemRecordStatus,
  processId, processRestarted, cancelledDueToRestart,
  createdBy, createdDateTime, lastModifiedBy, lastModifiedDateTime,
  locale, managerPersonId, employeePersonId, user,
  targetDate — IGNORE all of these
- Any field names not listed above — IGNORE entirely

Your job is ONLY to write the CLOSING section of the email.
The opening has already been written separately, do not repeat the greeting.

Output EXACTLY:
- One paragraph (2-3 sentences):
  - If ONB + onboardingInternalHire is false:
    mention startDate, welcome them to the team,
    say manager will reach out soon
  - If ONB + onboardingInternalHire is true:
    acknowledge it as an exciting internal move,
    mention startDate, say manager will connect
  - If OFB:
    mention endDate as their last day,
    thank them for contributions,
    wish them well in next chapter
- Empty line
- Sign off EXACTLY as: "Warm regards,\nKPMG HR Team"

STRICT RULES:
- Do NOT write any greeting or opening line
- Do NOT repeat the employee's full name
- Do NOT write a subject line
- Do NOT ask questions or add commentary
- Do NOT mention any fields you were told to ignore
- Do NOT reference field names or API names in the output
- If startDate or endDate is null across all inputs, skip the date gracefully
- If manager is null across all inputs, skip the manager mention entirely
- Output ONLY the paragraph + sign off, nothing else"""


# --- Request Model ---
# Accepts any number of fields with any names, all of type Any.
# Example: { "personal": {...}, "process": {...}, "extra_info": "..." }

class SAPRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


# --- Helpers ---

def flatten_to_root(data: dict) -> dict:
    merged = {}
    for value in data.values():
        if isinstance(value, dict):
            merged.update(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    merged.update(item)
        else:
            pass
    return merged


def build_content(request: SAPRequest) -> str:
    data = request.model_dump()
    if not data:
        return "{}"
    flat = flatten_to_root(data)
    return json.dumps(flat, ensure_ascii=False, indent=2)


def call_grok(system_prompt: str, request: SAPRequest) -> str:
    if not api_keys:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEYs configured")

    content = build_content(request)
    last_error = None

    for _ in range(len(api_keys)):
        current_key, key_name = next(key_cycle)
        try:
            client = Groq(api_key=current_key)
            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"Key {key_name} failed: {e} — trying next key")
            continue

    raise HTTPException(status_code=500, detail=f"All Groq keys failed. Last error: {last_error}")


# --- Endpoints ---

@router.post("/perpersonal")
async def perpersonal(request: SAPRequest):
    try:
        result = call_grok(PERPERSONAL_SYSTEM_PROMPT, request)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/onb2process")
async def onb2process(request: SAPRequest):
    try:
        result = call_grok(ONB2PROCESS_SYSTEM_PROMPT, request)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
