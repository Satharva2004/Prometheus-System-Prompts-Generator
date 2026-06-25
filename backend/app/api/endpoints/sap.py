from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import os
import itertools
import json
from groq import Groq

router = APIRouter()

# --- Key Rotation (reuse same pattern as ai.py) ---
api_keys = []
for i in range(1, 7):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        api_keys.append((key, f"GROQ_API_KEY_{i}"))

if not api_keys:
    print("Warning: No GROQ_API_KEY_1...6 found for SAP endpoints.")

key_cycle = itertools.cycle(api_keys) if api_keys else iter([])

def get_groq_client():
    if not api_keys:
        raise HTTPException(status_code=500, detail="No GROQ_API_KEYs configured")
    current_key, key_name = next(key_cycle)
    return Groq(api_key=current_key)


# --- System Prompts ---

PERPERSONAL_SYSTEM_PROMPT = """You are an HR email writer. You will receive a JSON collection
from the PerPersonal API of SAP SuccessFactors.

The collection contains these fields — here is what matters to you:
- salutation: Mr./Ms./Dr. etc — use for greeting
- preferredName: employee's preferred first name — USE THIS over firstName if available
- firstName: employee's first name — fallback if preferredName is null
- lastName: employee's last name
- displayName: full name — use only if both firstName and lastName are null
- middleName: middle name — ignore this
- gender, maritalStatus, nationality — IGNORE these entirely
- createdBy, createdDateTime, lastModifiedBy, operation,
  script, attachmentId, personIdExternal — IGNORE all of these

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
- If name fields are null, use displayName
- Output ONLY greeting line + one paragraph, nothing else"""

ONB2PROCESS_SYSTEM_PROMPT = """You are an HR email writer. You will receive a JSON collection
from the ONB2Process API of SAP SuccessFactors.

The collection contains these fields — here is what matters to you:
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
- Sign off EXACTLY as: "Warm regards,\\nKPMG HR Team"

STRICT RULES:
- Do NOT write any greeting or opening line
- Do NOT repeat the employee's full name
- Do NOT write a subject line
- Do NOT ask questions or add commentary
- Do NOT mention any fields you were told to ignore
- If startDate or endDate is null, skip the date gracefully
- If manager is null, skip the manager mention entirely
- Output ONLY the paragraph + sign off, nothing else"""


# --- Request Model ---

class SAPRequest(BaseModel):
    user_prompt: Any


# --- Helpers ---

def serialize_prompt(user_prompt: Any) -> str:
    if isinstance(user_prompt, str):
        return user_prompt
    return json.dumps(user_prompt, ensure_ascii=False, indent=2)


def call_grok(system_prompt: str, user_prompt: Any) -> str:
    client = get_groq_client()
    content = serialize_prompt(user_prompt)
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


# --- Endpoints ---

@router.post("/perpersonal")
async def perpersonal(request: SAPRequest):
    try:
        result = call_grok(PERPERSONAL_SYSTEM_PROMPT, request.user_prompt)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/onb2process")
async def onb2process(request: SAPRequest):
    try:
        result = call_grok(ONB2PROCESS_SYSTEM_PROMPT, request.user_prompt)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
