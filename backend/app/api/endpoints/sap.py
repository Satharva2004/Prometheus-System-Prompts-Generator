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

WELCOME_SYSTEM_PROMPT = """You are a warm and professional HR onboarding assistant.
Your job is to generate personalized, enthusiastic welcome messages for new employees joining the company.
Keep messages to 3-4 sentences. Be genuine, welcoming, and professional.
Do not use generic corporate language. Make the person feel excited about joining.

You will receive a JSON object with employee details. Look for these fields:
- firstName: employee's preferred first name — address them by first name ONLY
- lastName: employee's last name — do NOT use in the message
- jobTitle: the role they are joining in — mention naturally
- startDate or formattedStartDate: their first day — weave it in naturally

Output format — follow this EXACTLY:
- Use emojis naturally (1-2 per message, e.g. 🎉 at the start, 🚀 or 🌟 near the end)
- Use proper spacing: one blank line between the greeting and the body
- Write 3-4 sentences as one flowing paragraph
- End with a warm, genuine closing line wishing them well

STRICT RULES:
- Do NOT mention any follow-up steps, next steps, onboarding tasks, documents to submit, or actions required
- This is a one-time welcome message — do NOT reference any process, checklist, or what comes next
- Do NOT use phrases like "we will send you", "please complete", "you will receive", "look out for"
- Do NOT write a subject line, sign-off, or regards block
- Do NOT mention lastName anywhere in the output
- Do NOT use hollow corporate phrases like "We are pleased to inform you" or "on behalf of the organization"
- Output ONLY the welcome message, nothing else"""

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


# --- Models ---
# Request: accepts any number of fields with any names, all of type Any.
# Example: { "personal": {...}, "process": {...}, "extra_info": "..." }

class SAPRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

class SAPResponse(BaseModel):
    output: str


# --- Helpers ---

def flatten_to_root(data: dict) -> dict:
    merged = {}
    for key, value in data.items():
        if isinstance(value, dict):
            merged.update(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    merged.update(item)
        else:
            merged[key] = value
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

@router.post("/perpersonal", response_model=SAPResponse)
async def perpersonal(request: SAPRequest):
    try:
        result = call_grok(PERPERSONAL_SYSTEM_PROMPT, request)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/onb2process", response_model=SAPResponse)
async def onb2process(request: SAPRequest):
    try:
        result = call_grok(ONB2PROCESS_SYSTEM_PROMPT, request)
        return {"output": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/welcome", response_model=SAPResponse)
async def welcome(request: SAPRequest):
    """Generate a personalized one-time welcome message for a new hire."""
    try:
        data = flatten_to_root(request.model_dump())

        first_name   = data.get("firstName") or ""
        last_name    = data.get("lastName") or ""
        job_title    = data.get("jobTitle") or ""
        business_unit = data.get("businessUnit") or ""
        department   = data.get("department") or ""
        location     = data.get("location") or ""
        work_location = data.get("workLocation") or ""
        start_date   = data.get("startDate") or data.get("formattedStartDate") or ""

        # Build the details block — only include lines where a value exists
        details_lines = []
        if first_name or last_name:
            details_lines.append(f"Name: {(first_name + ' ' + last_name).strip()}")
        if job_title:
            details_lines.append(f"Job Title: {job_title}")
        if business_unit:
            details_lines.append(f"Business Unit: {business_unit}")
        if department:
            details_lines.append(f"Department: {department}")
        if location:
            details_lines.append(f"Location: {location}")
        if work_location:
            details_lines.append(f"Work Location: {work_location}")
        if start_date:
            details_lines.append(f"Start Date: {start_date}")

        if details_lines:
            details_block = "\n".join(details_lines)
            address_note = (
                f"Address them by first name{' (' + first_name + ')' if first_name else ''} only — do NOT use their last name. "
            )
        else:
            details_block = "No specific employee details provided."
            address_note = "No name was provided — open with a warm generic greeting. "

        user_prompt = (
            "Generate a welcome message for a new hire with the following details:\n\n"
            f"{details_block}\n\n"
            f"{address_note}"
            "Weave in the job title, department or business unit, location or work location, and start date naturally — "
            "only mention the ones that are actually provided above, skip any that are missing. "
            "Use 1-2 relevant emojis (e.g. 🎉 near the start, 🚀 or 🌟 near the end) and proper spacing between sentences. "
            "End with a warm, sincere closing line wishing them well. "
            "Do not mention any follow-up items, next steps, tasks, documents, or actions — "
            "this is a one-time welcome message only."
        )

        if not api_keys:
            raise HTTPException(status_code=500, detail="No GROQ_API_KEYs configured")

        last_error = None
        for _ in range(len(api_keys)):
            current_key, key_name = next(key_cycle)
            try:
                client = Groq(api_key=current_key)
                completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": WELCOME_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=1024,
                )
                return {"output": completion.choices[0].message.content}
            except Exception as e:
                last_error = e
                print(f"Key {key_name} failed: {e} — trying next key")
                continue

        raise HTTPException(status_code=500, detail=f"All Groq keys failed. Last error: {last_error}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
