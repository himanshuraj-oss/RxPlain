"""
RxPlain — Reads your doctor's prescription and explains it in plain language.

Three-step pipeline:
  1. AI OCR: Sarvam Vision (sarvam-m) primary → Groq Vision fallback
  2. Human Review: Patient sees & corrects the OCR text before analysis
  3. Analysis: Sarvam-105B primary → Groq (Llama 3.3 70B) fallback for reliability

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

import base64
import json
import os
import tempfile
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

import pickle
import joblib
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import streamlit as st
from sarvamai import SarvamAI
from PIL import Image, ImageEnhance

KERAS_AVAILABLE = False
keras_load_model = None

try:
    from icalendar import Calendar, Event
    from dateutil.rrule import rrule, DAILY
except ImportError:
    print("⚠️  icalendar not installed. Install: pip install icalendar python-dateutil")
    Calendar = None
    Event = None
    rrule = None
    DAILY = None

try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:
    convert_from_path = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except ImportError:
    print("⚠️  reportlab not installed. Install: pip install reportlab")
    SimpleDocTemplate = None

# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY",  "")

# ─────────────────────────────────────────────────────────────────────────────
# Icons (Pure SVG Glyphs)
# ─────────────────────────────────────────────────────────────────────────────
SVG_ICONS = {
    "clock": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>',
    "calendar": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="18" y2="10"/></svg>',
    "pill": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="4.5" y1="16.5" x2="19.5" y2="1.5"/><path d="M12 2A10 10 0 0 1 22 12A10 10 0 0 1 12 22A10 10 0 0 1 2 12A10 10 0 0 1 12 2Z"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "warning": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "cart": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/></svg>',
    "map": '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/><line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/></svg>',
    "scan": '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 7 3 3 7 3"></polyline><polyline points="17 3 21 3 21 7"></polyline><polyline points="21 17 21 21 17 21"></polyline><polyline points="7 21 3 21 3 17"></polyline><line x1="7" y1="12" x2="17" y2="12"></line></svg>',
    "cpu": '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>',
    "whatsapp": '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.272-.099-.47-.148-.669.15-.198.297-.768.966-.941 1.165-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.076 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421-7.403h-.004a9.87 9.87 0 0 0-9.746 9.798c0 2.603.75 5.106 2.171 7.238L2.26 22.25l7.53-2.49c2.02 1.08 4.286 1.65 6.658 1.65 5.459 0 9.95-4.432 9.975-9.914.005-2.647-.99-5.138-2.775-7.01-1.784-1.871-4.155-2.901-6.688-2.901"/></svg>',
    "play": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>',
    "pause": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>',
    "volume": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon></svg>'
}

# ─────────────────────────────────────────────────────────────────────────────
# Prompts & Demo Data
# ─────────────────────────────────────────────────────────────────────────────
OCR_PROMPT = """\
You are an expert medical OCR specialist trained to decode handwritten Indian doctor \
prescriptions — including extremely messy, rushed, abbreviated, and stylised handwriting.

Extract ALL text visible in this prescription. Pay close attention to:
DRUG ABBREVIATIONS, DOSAGES, FREQUENCY, TIMING, and DURATION.

Return ONLY the raw extracted prescription text — no preamble, no commentary.\
"""

ANALYSIS_PROMPT = """\
You are RxPlain — a trusted medical assistant helping Indian patients understand \
their prescriptions.

Analyse the prescription text below and explain EVERY medication clearly. Look up realistic \
retail price estimations in India for both the exact prescribed brand/medicine and its generic equivalent.
If you are unsure of the price, estimate realistically or output "₹—".

CRITICAL INSTRUCTION: Your response must be ONLY a valid JSON object.
• Start with { — end with }
• No markdown fences (no ```json or ```)
• No text before or after the JSON

Required JSON structure:
{
  "medications": [
    {
      "name": "Exact brand name or variant as written on prescription",
      "prescribedPrice": "Approx Indian retail price for this specific prescribed brand e.g. ₹120/strip",
      "genericName": "Salt composition / cheapest available generic equivalent brand in India",
      "genericPrice": "Approx Indian retail price for the generic version e.g. ₹22/strip",
      "purpose": "What this medicine does in 1-2 plain sentences any patient can understand",
      "dosage": "Dosage e.g. 500mg or 10ml",
      "frequency": "How often e.g. Twice daily / 1-0-1 / SOS",
      "timing": "When to take e.g. After meals / 30 min before breakfast",
      "duration": "How long e.g. 5 days / 1 month / As required",
      "sideEffects": ["most common side effect", "second side effect", "third side effect"],
      "warning": "One key safety warning for the patient, or null if none"
    }
  ],
  "interactions": {
    "text": "Notable drug interaction warning in plain language, or null if none",
    "severity": "none | minor | moderate | severe | contraindicated"
  },
  "foodInteractions": [
    {
      "medicine": "Medicine name this applies to",
      "food": "Food/beverage to avoid or be careful with",
      "reason": "Why this interaction matters in 1-2 plain sentences",
      "advice": "Practical guidance e.g. 'Avoid within 2 hours of taking medicine'"
    }
  ],
  "generalAdvice": "1-2 sentences of practical lifestyle advice based on this prescription",
  "doctorNotes": "Any follow-up instructions, special advice from the prescription, or null"
}

CRITICAL: Include foodInteractions array with common food-drug interactions like:
- Avoid dairy with antibiotics like tetracycline
- No grapefruit with BP medicines and statins
- Avoid tea/coffee near iron tablets (1-2 hour gap)
- No high vitamin K foods with warfarin
- Avoid alcohol with certain medications
This prevents dangerous food-drug interactions many patients don't know about.

Prescription text:
"""

DEMO_DATA = {
    "medications": [
        {
            "name": "Augmentin 625 Duo",
            "prescribedPrice": "₹220 / strip",
            "genericName": "Amoxicillin + Clavulanic Acid (Generic)",
            "genericPrice": "₹60 / strip",
            "purpose": "An advanced antibiotic that kills the bacteria causing your infection, engineered to bypass bacterial resistance walls.",
            "dosage": "625mg",
            "frequency": "2× daily",
            "timing": "After meals",
            "duration": "5 days",
            "sideEffects": ["Loose stools", "Nausea", "Mild stomach upset"],
            "warning": "Complete the full course even if you feel completely fine on Day 3.",
        }
    ],
    "interactions": {
        "text": "No dangerous structural drug interactions detected.",
        "severity": "none"
    },
    "foodInteractions": [
        {
            "medicine": "Augmentin 625 Duo",
            "food": "Dairy products",
            "reason": "Dairy can reduce antibiotic absorption in your stomach",
            "advice": "Take antibiotic 2 hours before or 3 hours after consuming milk/yogurt/cheese"
        }
    ],
    "generalAdvice": "Drink plenty of water and stay resting. Avoid heavy, oily food.",
    "doctorNotes": "Review after 5 days or sooner if high fever persists.",
}

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers & JSON Extractor
# ─────────────────────────────────────────────────────────────────────────────
def load_image_b64(file_path: str) -> tuple[str, str]:
    """Load and optimize image for OCR/vision processing"""
    suffix = Path(file_path).suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

    if suffix == ".pdf":
        if convert_from_path is None:
            raise RuntimeError("PDF support requires pdf2image + poppler.")
        import io
        pages = convert_from_path(file_path, first_page=1, last_page=1, dpi=200)
        buf = io.BytesIO()
        pages[0].save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

    try:
        import io
        
        img = Image.open(file_path)
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
            
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        
        # Resize preserving aspect ratio - NOT too aggressive
        max_size = (2000, 2000)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Smart preprocessing for medicine labels
        # Moderate enhancements for text readability without over-processing
        img = ImageEnhance.Contrast(img).enhance(1.4)  # Moderate contrast
        img = ImageEnhance.Sharpness(img).enhance(1.2)  # Subtle sharpness
        img = ImageEnhance.Brightness(img).enhance(1.05)  # Minimal brightness
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)  # Very high quality
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"
        
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode(), mime_map.get(suffix, "image/jpeg")
        except:
            raise

def extract_json(raw: str) -> dict:
    try: return json.loads(raw)
    except Exception: pass
    
    clean = raw.strip()
    for prefix in ("```json\n", "```json", "```\n", "```"):
        if clean.startswith(prefix): clean = clean[len(prefix):]; break
    clean = clean.rstrip("`").strip()
    try: return json.loads(clean)
    except Exception: pass

    first, last = raw.find("{"), raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        try: return json.loads(raw[first : last + 1])
        except Exception: pass
    raise json.JSONDecodeError("JSON format structural parsing failed.", raw, 0)

# TTS functionality removed - All content now delivered in selected language

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Calendar Sync (.ics generation)
# ─────────────────────────────────────────────────────────────────────────────
def generate_ics_calendar(result: dict) -> bytes:
    """Generate iCalendar (.ics) file with medication alarms"""
    if Calendar is None:
        raise RuntimeError("icalendar library not installed. Run: pip install icalendar python-dateutil")
    
    cal = Calendar()
    cal.add('prodid', '-//RxPlain//Medication Schedule//EN')
    cal.add('version', '2.0')
    cal.add('calscale', 'GREGORIAN')
    cal.add('method', 'PUBLISH')
    cal.add('x-wr-calname', 'RxPlain Medication Schedule')
    cal.add('x-wr-timezone', 'Asia/Kolkata')
    
    start_date = datetime.now()
    
    for med in result.get("medications", []):
        # Parse frequency (e.g., "1-0-1" = 3 times, "2x daily" = 2 times, etc.)
        freq_str = med.get("frequency", "1x daily").lower()
        daily_count = 1
        
        if "1-0-1" in freq_str:
            daily_count = 3
            times = ["8:00", "14:00", "20:00"]
        elif "0-1-1" in freq_str:
            daily_count = 2
            times = ["14:00", "20:00"]
        elif "1-0-0" in freq_str:
            daily_count = 1
            times = ["8:00"]
        elif "twice" in freq_str or "2x" in freq_str or "2 times" in freq_str:
            daily_count = 2
            times = ["8:00", "20:00"]
        elif "thrice" in freq_str or "3x" in freq_str or "3 times" in freq_str:
            daily_count = 3
            times = ["8:00", "14:00", "20:00"]
        else:
            daily_count = 1
            times = ["8:00"]
        
        # Parse duration
        duration_str = med.get("duration", "1 day")
        duration_days = 1
        if "day" in duration_str:
            import re
            match = re.search(r'(\d+)', duration_str)
            if match:
                duration_days = int(match.group(1))
        
        # Create events for each dose
        for day_offset in range(duration_days):
            event_date = start_date + timedelta(days=day_offset)
            
            for idx, time_str in enumerate(times[:daily_count]):
                hour, minute = map(int, time_str.split(':'))
                event_datetime = event_date.replace(hour=hour, minute=minute, second=0)
                
                event = Event()
                event.add('summary', f'💊 Take {med.get("name", "Medication")}')
                event.add('description', f'Dosage: {med.get("dosage", "")}\nInstructions: {med.get("timing", "")}')
                event.add('dtstart', event_datetime)
                event.add('dtend', event_datetime + timedelta(minutes=5))
                event.add('dtstamp', datetime.now())
                event.add('uid', f'{med.get("name", "med")}-{day_offset}-{idx}@rxplain')
                event.add('alarmit', True)
                
                # Add alarm 15 minutes before
                event.add_component(Event())
                
                cal.add_component(event)
    
    ics_content = cal.to_ical()
    return ics_content

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Follow-up Chat with Context
# ─────────────────────────────────────────────────────────────────────────────
def chat_followup(user_question: str, ocr_text: str, result: dict) -> str:
    """Process follow-up question with prescription context"""
    context = f"""You are RxPlain chat assistant. The user is asking about this prescription:

OCR Text: {ocr_text}

Analysis Result: {json.dumps(result, indent=2)}

User Question: {user_question}

Provide a clear, empathetic answer in plain language. If medical, keep it educational."""
    
    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        response = client.chat.completions(
            model="sarvam-105b",
            messages=[{"role": "user", "content": context}],
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Chat Error: {e}")
        # Fallback to Groq
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": context}],
                    "max_tokens": 512,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
    return "Unable to process your question. Please try again."

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Printable HTML Summary Card
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf_summary(result: dict, ocr_text: str) -> bytes:
    """Generate PDF summary of prescription analysis using reportlab"""
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab library not installed. Run: pip install reportlab")
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#007AFF'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#333333'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#555555'),
        spaceAfter=6
    )
    
    story = []
    
    # Title
    story.append(Paragraph("💊 RxPlain — Medication Summary", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<font size=9 color='#999999'>Generated on {timestamp}</font>", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Original Prescription
    story.append(Paragraph("Original Prescription Text", heading_style))
    ocr_text_safe = ocr_text.replace('<', '&lt;').replace('>', '&gt;')
    story.append(Paragraph(f"<font size=8 face='Courier'>{ocr_text_safe}</font>", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Medications
    story.append(Paragraph("Medication Details", heading_style))
    for med in result.get("medications", []):
        story.append(Paragraph(f"<b>{med.get('name', '')}</b>", normal_style))
        med_data = [
            ["Dosage:", med.get('dosage', '')],
            ["Frequency:", med.get('frequency', '')],
            ["Timing:", med.get('timing', '')],
            ["Duration:", med.get('duration', '')],
            ["Purpose:", med.get('purpose', '')],
            ["Prescribed Price:", med.get('prescribedPrice', '')],
            ["Generic Alternative:", f"{med.get('genericName', '')} ({med.get('genericPrice', '')})"],
            ["Side Effects:", ", ".join(med.get('sideEffects', []))],
            ["⚠ Warning:", med.get('warning', 'None')],
        ]
        table = Table(med_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 9),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#555555')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2*inch))
    
    # Interactions
    if result.get("interactions"):
        interactions = result.get("interactions", {})
        severity = interactions.get("severity", "none") if isinstance(interactions, dict) else "none"
        text = interactions.get("text", "") if isinstance(interactions, dict) else interactions
        story.append(Paragraph(f"⚠ Safety Alerts [{severity.upper()}]", heading_style))
        story.append(Paragraph(text, normal_style))
        story.append(Spacer(1, 0.2*inch))
    
    # General Advice
    story.append(Paragraph("Lifestyle Guidance", heading_style))
    story.append(Paragraph(result.get('generalAdvice', ''), normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Doctor Notes
    if result.get('doctorNotes'):
        story.append(Paragraph("Doctor's Notes", heading_style))
        story.append(Paragraph(result.get('doctorNotes', ''), normal_style))
    
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def generate_html_summary(result: dict, ocr_text: str) -> str:
    """Generate a printable HTML summary of the prescription analysis"""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    med_cards = ""
    for med in result.get("medications", []):
        med_cards += f"""
    <div style="page-break-inside: avoid; margin-bottom: 20px; border-left: 4px solid #007AFF; padding-left: 16px;">
        <h3 style="margin: 0 0 8px 0; color: #333;">{med.get('name', '')}</h3>
        <table style="width: 100%; font-size: 13px; color: #555;">
            <tr><td style="font-weight: bold; padding: 4px 0;">Dosage:</td><td>{med.get('dosage', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Frequency:</td><td>{med.get('frequency', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Timing:</td><td>{med.get('timing', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Duration:</td><td>{med.get('duration', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Purpose:</td><td>{med.get('purpose', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Prescribed Price:</td><td>{med.get('prescribedPrice', '')}</td></tr>
            <tr><td style="font-weight: bold; padding: 4px 0;">Generic Alternative:</td><td>{med.get('genericName', '')} ({med.get('genericPrice', '')})</td></tr>
        </table>
        <p style="font-size: 12px; margin-top: 8px; color: #d9534f;"><strong>⚠ Warning:</strong> {med.get('warning', 'None')}</p>
    </div>
"""
    
    interactions_html = ""
    if result.get("interactions"):
        interactions = result.get("interactions", {})
        severity = interactions.get("severity", "none") if isinstance(interactions, dict) else "none"
        severity_color = {"none": "#28a745", "minor": "#ffc107", "moderate": "#fd7e14", "severe": "#dc3545", "contraindicated": "#721c24"}.get(severity, "#999")
        interactions_text = interactions.get("text", "") if isinstance(interactions, dict) else interactions
        interactions_html = f"""
    <div style="background: {severity_color}22; border: 1px solid {severity_color}; border-radius: 8px; padding: 12px; margin-bottom: 20px;">
        <p style="margin: 0; color: {severity_color}; font-weight: bold;">⚠ Interaction Alert [{severity.upper()}]</p>
        <p style="margin: 4px 0 0 0; color: #333;">{interactions_text}</p>
    </div>
"""
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #333; line-height: 1.6; margin: 0; padding: 20px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ color: #007AFF; border-bottom: 2px solid #007AFF; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 24px; margin-bottom: 12px; font-size: 18px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
        td {{ padding: 4px 0; }}
        .timestamp {{ color: #999; font-size: 12px; margin: 20px 0 30px 0; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 11px; color: #999; }}
        @media print {{ body {{ margin: 0; padding: 0; }} .no-print {{ display: none; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>💊 RxPlain — Medication Summary</h1>
        <p class="timestamp">Generated on {timestamp}</p>
        
        <h2>Original Prescription Text</h2>
        <div style="background: #f5f5f5; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; color: #555; white-space: pre-wrap; word-break: break-word;">
{ocr_text}
        </div>
        
        <h2>Medication Details</h2>
        {med_cards}
        
        <h2>⚠ Safety Alerts</h2>
        {interactions_html}
        
        <h2>Lifestyle Guidance</h2>
        <p>{result.get('generalAdvice', '')}</p>
        
        {f"<h2>Doctor's Notes</h2><p>{result.get('doctorNotes', '')}</p>" if result.get('doctorNotes') else ''}
        
        <div class="footer">
            <p><strong>Disclaimer:</strong> RxPlain is an educational tool. Always follow your doctor's advice. This is not a medical consultation.</p>
            <p>For drug authenticity verification, visit: <a href="https://sugam.cci.gov.in" target="_blank">India's SUGAM Portal</a> or <a href="https://medsbio.com" target="_blank">MedsBio</a></p>
        </div>
    </div>
</body>
</html>
"""
    return html

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Drug Authenticity Check Links
# ─────────────────────────────────────────────────────────────────────────────
def get_authenticity_check_links(med_name: str) -> dict:
    """Generate links for drug authenticity verification"""
    search_query = urllib.parse.quote(med_name.split()[0])  # Use brand name only
    return {
        "sugam": f"https://sugam.cci.gov.in/Sugam/",
        "1mg": f"https://www.1mg.com/search/all?name={search_query}",
        "google": f"https://www.google.com/search?q={search_query}+medicine+price+India+generic"
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: WhatsApp Share Deep Link
# ─────────────────────────────────────────────────────────────────────────────
def generate_whatsapp_share_link(result: dict, ocr_text: str) -> str:
    """Generate WhatsApp share deep link with prescription summary"""
    message = "💊 *My Prescription Summary from RxPlain*\n\n"
    message += "*Medicines:*\n"
    
    for med in result.get("medications", []):
        message += f"• {med.get('name', '')} - {med.get('dosage', '')}\n"
        message += f"  Frequency: {med.get('frequency', '')}\n"
        message += f"  Duration: {med.get('duration', '')}\n"
        message += f"  💰 Generic saves: {med.get('genericPrice', '₹—')} vs {med.get('prescribedPrice', '₹—')}\n"
    
    if result.get("generalAdvice"):
        message += f"\n*Important Tips:* {result['generalAdvice']}\n"
    
    if result.get("interactions", {}).get("text") and result.get("interactions", {}).get("severity") != "none":
        message += f"\n⚠️ *Alert:* {result.get('interactions', {}).get('text', '')}\n"
    
    message += "\n_Shared via RxPlain - Understand your prescriptions_"
    
    # Encode message for WhatsApp
    encoded_message = urllib.parse.quote(message)
    return f"https://wa.me/?text={encoded_message}"

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Calculate Total Cost Savings
# ─────────────────────────────────────────────────────────────────────────────
def calculate_total_savings(result: dict) -> dict:
    """Calculate total savings from switching to generic medicines"""
    total_prescribed = 0
    total_generic = 0
    savings_per_med = []
    
    for med in result.get("medications", []):
        presc_str = med.get("prescribedPrice", "₹0").replace("₹", "").split("/")[0].strip()
        generic_str = med.get("genericPrice", "₹0").replace("₹", "").split("/")[0].strip()
        
        try:
            presc_price = float(presc_str)
            generic_price = float(generic_str)
            total_prescribed += presc_price
            total_generic += generic_price
            
            savings = presc_price - generic_price
            if savings > 0:
                savings_per_med.append({
                    "name": med.get("name", ""),
                    "savings": savings,
                    "percentage": round((savings / presc_price * 100)) if presc_price > 0 else 0
                })
        except:
            pass
    
    total_savings = total_prescribed - total_generic
    return {
        "total_prescribed": total_prescribed,
        "total_generic": total_generic,
        "total_savings": max(0, total_savings),
        "percentage": round((total_savings / total_prescribed * 100)) if total_prescribed > 0 else 0,
        "per_medicine": savings_per_med
    }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: ABHA/ABDM Integration
# ─────────────────────────────────────────────────────────────────────────────
def get_abha_integration_link(patient_name: str = "", prescription_hash: str = "") -> str:
    """Generate link to save prescription to ABHA Health Record"""
    # Updated ABHA portal - India's national health stack
    abha_base = "https://abdm.gov.in/"
    return abha_base

# ─────────────────────────────────────────────────────────────────────────────
# NEW: OCR Confidence Flagging
# ─────────────────────────────────────────────────────────────────────────────
def flag_uncertain_ocr_words(ocr_text: str) -> tuple[str, list]:
    """Identify potentially uncertain words in OCR text and flag them"""
    # Common medical abbreviations and drugs that OCR might struggle with
    uncertain_patterns = [
        r'[0O]mg',  # 0 vs O
        r'IU',      # Often misread
        r'ml\b',    # ml vs mi
        r'\d+\-\d+\-\d+',  # Frequency patterns
        r'[B8]',    # Often confused
    ]
    
    import re
    flagged_words = []
    highlighted_text = ocr_text
    
    # Simple heuristic: words with numbers and letters mixed are often uncertain
    words = ocr_text.split()
    for i, word in enumerate(words):
        # Flag complex patterns that mixing of case/numbers/special chars
        if len(word) > 3 and sum(c.isdigit() for c in word) > 0 and sum(c.isalpha() for c in word) > 0:
            if word not in ["500mg", "1000mg", "250mg", "10ml", "5ml", "1-0-1", "0-1-1"]:
                if word not in flagged_words:
                    flagged_words.append(word)
    
    return highlighted_text, flagged_words


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: AI OCR Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def ocr_sarvam_vision(image_b64: str, mime: str) -> str:
    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
        json={
            "model": "sarvam-m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": OCR_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}],
            "max_tokens": 1024, 
            "temperature": 0.05,
        },
        timeout=60,
    )
    if resp.status_code != 200: 
        raise RuntimeError(f"Sarvam Vision HTTP {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()

def ocr_groq_vision(image_b64: str, mime: str) -> str:
    if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY not found.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct", 
            "messages": [{"role": "user", "content": [{"type": "text", "text": OCR_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]}],
            "max_tokens": 1024, 
            "temperature": 0.05,
        },
        timeout=60,
    )
    if resp.status_code != 200: 
        raise RuntimeError(f"Groq API Error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()

def run_ocr(file_path: str) -> tuple[str, str]:
    image_b64, mime = load_image_b64(file_path)
    try:
        return ocr_sarvam_vision(image_b64, mime), "Sarvam Vision"
    except Exception as e_sarvam:
        print(f"Sarvam OCR Failed: {e_sarvam}")
        return ocr_groq_vision(image_b64, mime), "Groq Vision"

# ─────────────────────────────────────────────────────────────────────────────
# ML Report Vision OCR — Sarvam → Groq pipeline, auto-fill from lab photos
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_with_prompt(image_b64: str, mime: str, prompt: str) -> str:
    """Run vision OCR with a custom prompt. Tries Sarvam Vision first, falls back to Groq."""
    try:
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
            json={
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                ]}],
                "max_tokens": 1024,
                "temperature": 0.05,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"Sarvam Vision HTTP {resp.status_code}")
    except Exception as e_s:
        if not GROQ_API_KEY:
            raise RuntimeError("Both Sarvam and Groq unavailable.") from e_s
        resp2 = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                ]}],
                "max_tokens": 1024,
                "temperature": 0.05,
            },
            timeout=60,
        )
        if resp2.status_code != 200:
            raise RuntimeError(f"Groq API Error {resp2.status_code}: {resp2.text}")
        return resp2.json()["choices"][0]["message"]["content"].strip()


# Per-model JSON extraction prompts — each returns session_state key → value
ML_OCR_PROMPTS = {
    "diabetes": """\
You are a medical OCR specialist. Extract diabetes/metabolic test values from this lab report image.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "db_preg": <number of pregnancies, integer — use 0 if male or not shown>,
  "db_gluc": <plasma glucose mg/dL, integer>,
  "db_bp":   <diastolic blood pressure mmHg, integer>,
  "db_skin": <triceps skin fold thickness mm, integer — use 0 if not shown>,
  "db_ins":  <2-hour serum insulin µU/mL, integer — use 0 if not shown>,
  "db_bmi":  <BMI kg/m², float>,
  "db_dpf":  <diabetes pedigree function, float — use 0.5 if not shown>,
  "db_age":  <age in years, integer>
}
Set missing fields to null. Extract numeric values only, no units.""",

    "heart": """\
You are a medical OCR specialist. Extract cardiac test values from this ECG or cardiology report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "ht_age":    <age in years, integer>,
  "ht_sex":    <0=female, 1=male>,
  "ht_cp":     <chest pain type: 0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic>,
  "ht_trest":  <resting blood pressure systolic mmHg, integer>,
  "ht_chol":   <serum cholesterol mg/dL, integer>,
  "ht_fbs":    <fasting blood sugar: 1 if greater than 120 mg/dL else 0>,
  "ht_recg":   <resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy>,
  "ht_thal_hr":<maximum heart rate achieved bpm, integer>,
  "ht_exang":  <exercise induced angina: 1=yes, 0=no>,
  "ht_oldp":   <ST depression oldpeak, float>,
  "ht_slope":  <slope of ST segment: 0=upsloping, 1=flat, 2=downsloping>,
  "ht_ca":     <number of major vessels 0-3 on fluoroscopy>,
  "ht_thalv":  <thalassemia: 0=normal, 1=fixed defect, 2=reversable defect, 3=unknown>
}
Set missing fields to null.""",

    "osteoporosis": """\
You are a medical OCR specialist. Extract patient information for osteoporosis risk from this report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "os_age":  <age in years, integer>,
  "os_sex":  <0=female, 1=male>,
  "os_horm": <hormonal changes: 0=normal, 1=postmenopausal>,
  "os_fam":  <family history of osteoporosis: 0=no, 1=yes>,
  "os_race": <ethnicity: 0=African American, 1=Asian, 2=Caucasian>,
  "os_bw":   <body weight: 0=normal, 1=overweight, 2=underweight>,
  "os_calc": <calcium intake: 0=adequate, 1=low>,
  "os_vitd": <vitamin D: 0=sufficient, 1=insufficient>,
  "os_phys": <physical activity: 0=active, 1=sedentary>,
  "os_smok": <smoking: 0=no, 1=yes>,
  "os_alc":  <alcohol: 0=none, 1=moderate>,
  "os_cond": <medical conditions: 0=none, 1=hyperthyroidism, 2=rheumatoid arthritis>,
  "os_meds": <medications: 0=none, 1=corticosteroids>,
  "os_frac": <prior fractures: 0=no, 1=yes>
}
Set missing fields to null.""",

    "stroke": """\
You are a medical OCR specialist. Extract stroke risk factors from this medical report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "sk_gender":  <0=female, 1=male, 2=other>,
  "sk_age":     <age in years, integer>,
  "sk_htn":     <hypertension: 0=no, 1=yes>,
  "sk_heart":   <heart disease history: 0=no, 1=yes>,
  "sk_married": <ever married: 0=no, 1=yes>,
  "sk_work":    <work type: 0=govt job, 1=never worked, 2=private, 3=self-employed, 4=children>,
  "sk_res":     <residence: 0=rural, 1=urban>,
  "sk_glucose": <average glucose level mg/dL, float>,
  "sk_bmi":     <BMI kg/m², float>,
  "sk_smoking": <smoking: 0=unknown, 1=formerly smoked, 2=never smoked, 3=currently smokes>
}
Set missing fields to null.""",

    "liver": """\
You are a medical OCR specialist. Extract liver function test (LFT) values from this lab report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "lv_age":  <age in years, integer>,
  "lv_sex":  <0=female, 1=male>,
  "lv_tbil": <total bilirubin mg/dL, float>,
  "lv_dbil": <direct bilirubin mg/dL, float>,
  "lv_alkp": <alkaline phosphatase ALP IU/L, integer>,
  "lv_alt":  <ALT or SGPT IU/L, integer>,
  "lv_ast":  <AST or SGOT IU/L, integer>,
  "lv_tp":   <total protein g/dL, float>,
  "lv_alb":  <albumin g/dL, float>,
  "lv_agr":  <albumin to globulin A/G ratio, float>
}
Set missing fields to null.""",

    "kidney": """\
You are a medical OCR specialist. Extract kidney function and blood test values from this lab report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "kd_age":  <age in years, integer>,
  "kd_bp":   <blood pressure mmHg, integer>,
  "kd_bgr":  <blood glucose random mg/dL, integer>,
  "kd_bu":   <blood urea mg/dL, integer>,
  "kd_sc":   <serum creatinine mg/dL, float>,
  "kd_sod":  <sodium mEq/L, integer>,
  "kd_pot":  <potassium mEq/L, float>,
  "kd_hemo": <haemoglobin g/dL, float>,
  "kd_pcv":  <packed cell volume percent, integer>,
  "kd_wc":   <WBC white blood cell count cells per microlitre, integer>,
  "kd_rc":   <RBC red blood cell count millions per microlitre, float>,
  "kd_htn":  <hypertension: 0=no, 1=yes>,
  "kd_dm":   <diabetes mellitus: 0=no, 1=yes>,
  "kd_ane":  <anaemia: 0=no, 1=yes>
}
Set missing fields to null.""",

    "breast_cancer": """\
You are a medical OCR specialist. Extract tumor measurements from this pathology/biopsy report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "bc_radius":    <mean radius mm, float>,
  "bc_texture":   <mean texture, float>,
  "bc_perimeter": <mean perimeter mm, float>,
  "bc_area":      <mean area mm squared, float>,
  "bc_smooth":    <mean smoothness, float>,
  "bc_compact":   <mean compactness, float>,
  "bc_concav":    <mean concavity, float>,
  "bc_concpts":   <mean concave points, float>,
  "bc_symm":      <mean symmetry, float>,
  "bc_fractal":   <mean fractal dimension, float>
}
Set missing fields to null.""",

    "parkinsons": """\
You are a medical OCR specialist. Extract voice biomarker measurements from this Parkinson's voice analysis report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "pk_fo":      <MDVP:Fo(Hz) average vocal fundamental frequency, float>,
  "pk_fhi":     <MDVP:Fhi(Hz) maximum vocal fundamental frequency, float>,
  "pk_flo":     <MDVP:Flo(Hz) minimum vocal fundamental frequency, float>,
  "pk_jitter":  <MDVP:Jitter(%) local variation in fundamental frequency, float>,
  "pk_shimmer": <MDVP:Shimmer local variation in amplitude, float>,
  "pk_hnr":     <HNR harmonics-to-noise ratio dB, float>,
  "pk_rpde":    <RPDE recurrence period density entropy, float>,
  "pk_dfa":     <DFA signal fractal scaling exponent, float>,
  "pk_spread1": <spread1 nonlinear measure of fundamental frequency variation, float — usually negative>,
  "pk_ppe":     <PPE pitch period entropy, float>
}
Set missing fields to null.""",

    "thyroid": """\
You are a medical OCR specialist. Extract thyroid function test (TFT) values from this lab report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "th_age":    <age in years, integer>,
  "th_sex":    <0=female, 1=male>,
  "th_tsh":    <TSH value mIU/L, float>,
  "th_t3":     <T3 value nmol/L, float>,
  "th_tt4":    <Total T4 nmol/L, float>,
  "th_t4u":    <T4 Uptake ratio, float>,
  "th_fti":    <Free Thyroxine Index FTI, float>,
  "th_onthyr": <on thyroxine medication: 0=no, 1=yes>,
  "th_antith": <on antithyroid medication: 0=no, 1=yes>,
  "th_sick":   <currently sick: 0=no, 1=yes>,
  "th_preg":   <pregnant: 0=no, 1=yes>,
  "th_surg":   <thyroid surgery history: 0=no, 1=yes>
}
Set missing fields to null.""",

    "anaemia": """\
You are a medical OCR specialist. Extract Complete Blood Count (CBC) values from this lab report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "an_gender": <0=female, 1=male>,
  "an_hgb":    <Haemoglobin g/dL, float>,
  "an_mch":    <MCH pg, float>,
  "an_mchc":   <MCHC g/dL, float>,
  "an_mcv":    <MCV fL, float>
}
Set missing fields to null.""",

    "pcos": """\
You are a medical OCR specialist. Extract hormonal profile and ultrasound values for PCOS assessment.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "pc_age":      <age in years, integer>,
  "pc_bmi":      <BMI kg per m², float>,
  "pc_fsh":      <FSH mIU per mL, float>,
  "pc_lh":       <LH mIU per mL, float>,
  "pc_amh":      <AMH ng per mL, float>,
  "pc_cycle":    <menstrual cycle length days, integer>,
  "pc_fol_r":    <follicle count right ovary on USG, integer>,
  "pc_fol_l":    <follicle count left ovary on USG, integer>,
  "pc_endo":     <endometrium thickness mm, float>,
  "pc_irreg":    <irregular cycle: 0=regular, 1=irregular>,
  "pc_skin":     <skin darkening: 0=no, 1=yes>,
  "pc_hair":     <excessive hair growth: 0=no, 1=yes>,
  "pc_pimple":   <pimples or acne: 0=no, 1=yes>,
  "pc_hairloss": <hair loss: 0=no, 1=yes>,
  "pc_wtgain":   <weight gain: 0=no, 1=yes>
}
Set missing fields to null.""",

    "heart_failure": """\
You are a medical OCR specialist. Extract heart failure clinical values from this cardiology report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "hf_age":       <age in years, integer>,
  "hf_sex":       <0=female, 1=male>,
  "hf_ef":        <ejection fraction percent, integer>,
  "hf_cpk":       <creatinine phosphokinase CPK IU per L, integer>,
  "hf_platelets": <platelet count cells per mL, float>,
  "hf_sc":        <serum creatinine mg per dL, float>,
  "hf_sodium":    <serum sodium mEq per L, integer>,
  "hf_time":      <follow-up period in days, integer — use 100 if not shown>,
  "hf_anaemia":   <anaemia present: 0=no, 1=yes>,
  "hf_diabetes":  <diabetes: 0=no, 1=yes>,
  "hf_hbp":       <high blood pressure: 0=no, 1=yes>,
  "hf_smoking":   <smoking: 0=no, 1=yes>
}
Set missing fields to null.""",

    "cervical": """\
You are a medical OCR specialist. Extract cervical cancer risk factor values from this gynaecology report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "cc_age":        <age in years, integer>,
  "cc_partners":   <number of sexual partners, integer>,
  "cc_first_sex":  <age at first sexual intercourse, integer>,
  "cc_pregnancies":<number of pregnancies, integer>,
  "cc_smokes":     <smokes: 0=no, 1=yes>,
  "cc_smokes_yrs": <years of smoking, float — use 0 if non-smoker>,
  "cc_hc":         <hormonal contraceptives use: 0=no, 1=yes>,
  "cc_hc_yrs":     <years of hormonal contraceptive use, float>,
  "cc_iud":        <IUD use: 0=no, 1=yes>,
  "cc_iud_yrs":    <years of IUD use, float>,
  "cc_stds":       <STDs history: 0=no, 1=yes>,
  "cc_stds_n":     <number of STD diagnoses, integer>,
  "cc_dx_hpv":     <HPV diagnosis: 0=no, 1=yes>,
  "cc_dx_cin":     <CIN diagnosis: 0=no, 1=yes>,
  "cc_dx_cancer":  <prior cancer diagnosis: 0=no, 1=yes>
}
Set missing fields to null.""",

    "hepatitis": """\
You are a medical OCR specialist. Extract hepatitis liver enzyme values from this lab report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "hep_age":  <age in years, integer>,
  "hep_sex":  <0=female, 1=male>,
  "hep_alb":  <albumin ALB g per dL, float>,
  "hep_alp":  <alkaline phosphatase ALP IU per L, float>,
  "hep_alt":  <ALT SGPT IU per L, float>,
  "hep_ast":  <AST SGOT IU per L, float>,
  "hep_bil":  <total bilirubin BIL mg per dL, float>,
  "hep_che":  <cholinesterase CHE kU per L, float>,
  "hep_chol": <cholesterol mmol per L, float>,
  "hep_crea": <creatinine micromol per L, float>,
  "hep_ggt":  <GGT IU per L, float>,
  "hep_prot": <total protein g per dL, float>
}
Set missing fields to null.""",

    "sepsis": """\
You are a medical OCR specialist. Extract ICU patient vitals and lab values from this clinical report.
Return ONLY a valid JSON object — no markdown fences, no extra text.
{
  "sp_hr":     <heart rate beats per minute, float>,
  "sp_o2":     <oxygen saturation percent, float>,
  "sp_temp":   <body temperature Celsius, float>,
  "sp_sbp":    <systolic blood pressure mmHg, float>,
  "sp_map":    <mean arterial pressure mmHg, float>,
  "sp_resp":   <respiration rate breaths per minute, float>,
  "sp_wbc":    <white blood cell count 10 to the power 9 per L, float>,
  "sp_hgb":    <haemoglobin g per dL, float>,
  "sp_bun":    <blood urea nitrogen mg per dL, float>,
  "sp_creat":  <creatinine mg per dL, float>,
  "sp_gluc":   <glucose mg per dL, float>,
  "sp_lact":   <lactate mmol per L, float>,
  "sp_age":    <patient age in years, float>,
  "sp_iculos": <ICU length of stay in hours, float — use 1 if not applicable>
}
Set missing fields to null.""",
}


def ocr_ml_report(image_b64: str, mime: str, model_type: str) -> dict:
    """Extract lab values from a medical report image for a specific ML predictor."""
    prompt = ML_OCR_PROMPTS.get(model_type)
    if not prompt:
        raise ValueError(f"No OCR prompt defined for model type: {model_type}")
    raw = _ocr_with_prompt(image_b64, mime, prompt)
    return extract_json(raw)


def _apply_ml_ocr_to_session(model_type: str, extracted: dict):
    """Map OCR-extracted numeric codes to exact selectbox strings and write session_state."""

    def _set(key, val, opt_map=None):
        if val is None:
            return
        if opt_map and isinstance(val, (int, float)):
            mapped = opt_map.get(int(round(val)))
            if mapped:
                st.session_state[key] = mapped
            return
        st.session_state[key] = val

    _HEART_CP   = {0:"0 – Typical Angina", 1:"1 – Atypical Angina", 2:"2 – Non-anginal Pain", 3:"3 – Asymptomatic"}
    _HEART_ECG  = {0:"0 – Normal", 1:"1 – ST-T Abnormality", 2:"2 – LV Hypertrophy"}
    _HEART_SLP  = {0:"0 – Upsloping", 1:"1 – Flat", 2:"2 – Downsloping"}
    _HEART_THAL = {0:"0 – Normal", 1:"1 – Fixed Defect", 2:"2 – Reversable Defect", 3:"3 – Unknown"}
    _SEX_MF     = {0:"Female (0)", 1:"Male (1)"}
    _YES_NO     = {0:"No (0)", 1:"Yes (1)"}
    _OS_RACE    = {0:"African American (0)", 1:"Asian (1)", 2:"Caucasian (2)"}
    _OS_BW      = {0:"Normal (0)", 1:"Overweight (1)", 2:"Underweight (2)"}
    _OS_HORM    = {0:"Normal (0)", 1:"Postmenopausal (1)"}
    _OS_COND    = {0:"None (0)", 1:"Hyperthyroidism (1)", 2:"Rheumatoid Arthritis (2)"}
    _OS_CALC    = {0:"Adequate (0)", 1:"Low (1)"}
    _OS_VITD    = {0:"Sufficient (0)", 1:"Insufficient (1)"}
    _OS_PHYS    = {0:"Active (0)", 1:"Sedentary (1)"}
    _OS_MEDS    = {0:"None (0)", 1:"Corticosteroids (1)"}
    _OS_ALC     = {0:"None (0)", 1:"Moderate (1)"}
    _SK_GEN     = {0:"Female (0)", 1:"Male (1)", 2:"Other (2)"}
    _SK_WORK    = {0:"Government Job (0)", 1:"Never Worked (1)", 2:"Private (2)", 3:"Self-employed (3)", 4:"Children (4)"}
    _SK_SMOK    = {0:"Unknown (0)", 1:"Formerly Smoked (1)", 2:"Never Smoked (2)", 3:"Smokes (3)"}
    _SK_RES     = {0:"Rural (0)", 1:"Urban (1)"}

    if model_type == "diabetes":
        for k in ["db_preg", "db_gluc", "db_bp", "db_skin", "db_ins", "db_bmi", "db_dpf", "db_age"]:
            _set(k, extracted.get(k))

    elif model_type == "heart":
        _set("ht_age",    extracted.get("ht_age"))
        _set("ht_sex",    extracted.get("ht_sex"),    _SEX_MF)
        _set("ht_cp",     extracted.get("ht_cp"),     _HEART_CP)
        _set("ht_trest",  extracted.get("ht_trest"))
        _set("ht_chol",   extracted.get("ht_chol"))
        _set("ht_fbs",    extracted.get("ht_fbs"),    _YES_NO)
        _set("ht_recg",   extracted.get("ht_recg"),   _HEART_ECG)
        _set("ht_thal_hr",extracted.get("ht_thal_hr"))
        _set("ht_exang",  extracted.get("ht_exang"),  _YES_NO)
        _set("ht_oldp",   extracted.get("ht_oldp"))
        _set("ht_slope",  extracted.get("ht_slope"),  _HEART_SLP)
        _set("ht_ca",     extracted.get("ht_ca"))
        _set("ht_thalv",  extracted.get("ht_thalv"),  _HEART_THAL)

    elif model_type == "osteoporosis":
        _set("os_age",  extracted.get("os_age"))
        _set("os_sex",  extracted.get("os_sex"),  _SEX_MF)
        _set("os_horm", extracted.get("os_horm"), _OS_HORM)
        _set("os_fam",  extracted.get("os_fam"),  _YES_NO)
        _set("os_race", extracted.get("os_race"), _OS_RACE)
        _set("os_bw",   extracted.get("os_bw"),   _OS_BW)
        _set("os_calc", extracted.get("os_calc"), _OS_CALC)
        _set("os_vitd", extracted.get("os_vitd"), _OS_VITD)
        _set("os_phys", extracted.get("os_phys"), _OS_PHYS)
        _set("os_smok", extracted.get("os_smok"), _YES_NO)
        _set("os_alc",  extracted.get("os_alc"),  _OS_ALC)
        _set("os_cond", extracted.get("os_cond"), _OS_COND)
        _set("os_meds", extracted.get("os_meds"), _OS_MEDS)
        _set("os_frac", extracted.get("os_frac"), _YES_NO)

    elif model_type == "stroke":
        _set("sk_gender",  extracted.get("sk_gender"),  _SK_GEN)
        _set("sk_age",     extracted.get("sk_age"))
        _set("sk_htn",     extracted.get("sk_htn"),     _YES_NO)
        _set("sk_heart",   extracted.get("sk_heart"),   _YES_NO)
        _set("sk_married", extracted.get("sk_married"), _YES_NO)
        _set("sk_work",    extracted.get("sk_work"),    _SK_WORK)
        _set("sk_res",     extracted.get("sk_res"),     _SK_RES)
        _set("sk_glucose", extracted.get("sk_glucose"))
        _set("sk_bmi",     extracted.get("sk_bmi"))
        _set("sk_smoking", extracted.get("sk_smoking"), _SK_SMOK)

    elif model_type == "liver":
        _set("lv_age",  extracted.get("lv_age"))
        _set("lv_sex",  extracted.get("lv_sex"),  _SEX_MF)
        _set("lv_tbil", extracted.get("lv_tbil"))
        _set("lv_dbil", extracted.get("lv_dbil"))
        _set("lv_alkp", extracted.get("lv_alkp"))
        _set("lv_alt",  extracted.get("lv_alt"))
        _set("lv_ast",  extracted.get("lv_ast"))
        _set("lv_tp",   extracted.get("lv_tp"))
        _set("lv_alb",  extracted.get("lv_alb"))
        _set("lv_agr",  extracted.get("lv_agr"))

    elif model_type == "kidney":
        _set("kd_age",  extracted.get("kd_age"))
        _set("kd_bp",   extracted.get("kd_bp"))
        _set("kd_bgr",  extracted.get("kd_bgr"))
        _set("kd_bu",   extracted.get("kd_bu"))
        _set("kd_sc",   extracted.get("kd_sc"))
        _set("kd_sod",  extracted.get("kd_sod"))
        _set("kd_pot",  extracted.get("kd_pot"))
        _set("kd_hemo", extracted.get("kd_hemo"))
        _set("kd_pcv",  extracted.get("kd_pcv"))
        _set("kd_wc",   extracted.get("kd_wc"))
        _set("kd_rc",   extracted.get("kd_rc"))
        _set("kd_htn",  extracted.get("kd_htn"), _YES_NO)
        _set("kd_dm",   extracted.get("kd_dm"),  _YES_NO)
        _set("kd_ane",  extracted.get("kd_ane"), _YES_NO)

    elif model_type == "breast_cancer":
        for k in ["bc_radius","bc_texture","bc_perimeter","bc_area","bc_smooth",
                  "bc_compact","bc_concav","bc_concpts","bc_symm","bc_fractal"]:
            _set(k, extracted.get(k))

    elif model_type == "parkinsons":
        for k in ["pk_fo","pk_fhi","pk_flo","pk_jitter","pk_shimmer",
                  "pk_hnr","pk_rpde","pk_dfa","pk_spread1","pk_ppe"]:
            _set(k, extracted.get(k))

    elif model_type == "thyroid":
        _TH_SEX  = {0:"Female (0)", 1:"Male (1)"}
        _TH_BINO = {0:"No (0)", 1:"Yes (1)"}
        _set("th_age",    extracted.get("th_age"))
        _set("th_sex",    extracted.get("th_sex"),    _TH_SEX)
        _set("th_tsh",    extracted.get("th_tsh"))
        _set("th_t3",     extracted.get("th_t3"))
        _set("th_tt4",    extracted.get("th_tt4"))
        _set("th_t4u",    extracted.get("th_t4u"))
        _set("th_fti",    extracted.get("th_fti"))
        _set("th_onthyr", extracted.get("th_onthyr"), _TH_BINO)
        _set("th_antith", extracted.get("th_antith"), _TH_BINO)
        _set("th_sick",   extracted.get("th_sick"),   _TH_BINO)
        _set("th_preg",   extracted.get("th_preg"),   _TH_BINO)
        _set("th_surg",   extracted.get("th_surg"),   _TH_BINO)

    elif model_type == "anaemia":
        _AN_GEN = {0:"Female (0)", 1:"Male (1)"}
        _set("an_gender", extracted.get("an_gender"), _AN_GEN)
        _set("an_hgb",    extracted.get("an_hgb"))
        _set("an_mch",    extracted.get("an_mch"))
        _set("an_mchc",   extracted.get("an_mchc"))
        _set("an_mcv",    extracted.get("an_mcv"))

    elif model_type == "pcos":
        _PC_BINO = {0:"No (0)", 1:"Yes (1)"}
        _PC_CYC  = {0:"Regular (0)", 1:"Irregular (1)"}
        _set("pc_age",      extracted.get("pc_age"))
        _set("pc_bmi",      extracted.get("pc_bmi"))
        _set("pc_fsh",      extracted.get("pc_fsh"))
        _set("pc_lh",       extracted.get("pc_lh"))
        _set("pc_amh",      extracted.get("pc_amh"))
        _set("pc_cycle",    extracted.get("pc_cycle"))
        _set("pc_fol_r",    extracted.get("pc_fol_r"))
        _set("pc_fol_l",    extracted.get("pc_fol_l"))
        _set("pc_endo",     extracted.get("pc_endo"))
        _set("pc_irreg",    extracted.get("pc_irreg"),    _PC_CYC)
        _set("pc_skin",     extracted.get("pc_skin"),     _PC_BINO)
        _set("pc_hair",     extracted.get("pc_hair"),     _PC_BINO)
        _set("pc_pimple",   extracted.get("pc_pimple"),   _PC_BINO)
        _set("pc_hairloss", extracted.get("pc_hairloss"), _PC_BINO)
        _set("pc_wtgain",   extracted.get("pc_wtgain"),   _PC_BINO)

    elif model_type == "heart_failure":
        _YN = {0:"No (0)", 1:"Yes (1)"}
        _SM = {0:"Female (0)", 1:"Male (1)"}
        _set("hf_age",      extracted.get("hf_age"))
        _set("hf_sex",      extracted.get("hf_sex"),      _SM)
        _set("hf_ef",       extracted.get("hf_ef"))
        _set("hf_cpk",      extracted.get("hf_cpk"))
        _set("hf_platelets",extracted.get("hf_platelets"))
        _set("hf_sc",       extracted.get("hf_sc"))
        _set("hf_sodium",   extracted.get("hf_sodium"))
        _set("hf_time",     extracted.get("hf_time"))
        _set("hf_anaemia",  extracted.get("hf_anaemia"),  _YN)
        _set("hf_diabetes", extracted.get("hf_diabetes"), _YN)
        _set("hf_hbp",      extracted.get("hf_hbp"),      _YN)
        _set("hf_smoking",  extracted.get("hf_smoking"),  _YN)

    elif model_type == "cervical":
        _YN = {0:"No (0)", 1:"Yes (1)"}
        _set("cc_age",         extracted.get("cc_age"))
        _set("cc_partners",    extracted.get("cc_partners"))
        _set("cc_first_sex",   extracted.get("cc_first_sex"))
        _set("cc_pregnancies", extracted.get("cc_pregnancies"))
        _set("cc_smokes",      extracted.get("cc_smokes"),      _YN)
        _set("cc_smokes_yrs",  extracted.get("cc_smokes_yrs"))
        _set("cc_hc",          extracted.get("cc_hc"),          _YN)
        _set("cc_hc_yrs",      extracted.get("cc_hc_yrs"))
        _set("cc_iud",         extracted.get("cc_iud"),         _YN)
        _set("cc_iud_yrs",     extracted.get("cc_iud_yrs"))
        _set("cc_stds",        extracted.get("cc_stds"),        _YN)
        _set("cc_stds_n",      extracted.get("cc_stds_n"))
        _set("cc_dx_hpv",      extracted.get("cc_dx_hpv"),      _YN)
        _set("cc_dx_cin",      extracted.get("cc_dx_cin"),      _YN)
        _set("cc_dx_cancer",   extracted.get("cc_dx_cancer"),   _YN)

    elif model_type == "hepatitis":
        _SM = {0:"Female (0)", 1:"Male (1)"}
        _set("hep_age",  extracted.get("hep_age"))
        _set("hep_sex",  extracted.get("hep_sex"),  _SM)
        _set("hep_alb",  extracted.get("hep_alb"))
        _set("hep_alp",  extracted.get("hep_alp"))
        _set("hep_alt",  extracted.get("hep_alt"))
        _set("hep_ast",  extracted.get("hep_ast"))
        _set("hep_bil",  extracted.get("hep_bil"))
        _set("hep_che",  extracted.get("hep_che"))
        _set("hep_chol", extracted.get("hep_chol"))
        _set("hep_crea", extracted.get("hep_crea"))
        _set("hep_ggt",  extracted.get("hep_ggt"))
        _set("hep_prot", extracted.get("hep_prot"))

    elif model_type == "sepsis":
        for k in ["sp_hr","sp_o2","sp_temp","sp_sbp","sp_map","sp_resp",
                  "sp_wbc","sp_hgb","sp_bun","sp_creat","sp_gluc","sp_lact",
                  "sp_age","sp_iculos"]:
            _set(k, extracted.get(k))


def _render_report_scanner(model_type: str, scanner_note: str = "blood test or medical report"):
    """Render the collapsible 'Scan Lab Report' section for any ML predictor tab."""
    with st.expander("📷 Scan a Lab Report — AI auto-fills all fields from photo", expanded=False):
        st.markdown(f"""
<div style="background:linear-gradient(135deg,#1C1C1E,#111111);border:1px solid #2C2C2E;border-radius:12px;padding:14px 16px;margin-bottom:12px;">
<div style="font-size:14px;font-weight:600;color:#FFFFFF;margin-bottom:6px;">📸 Auto-fill from Report Photo</div>
<div style="font-size:13px;color:#A1A1A6;line-height:1.6;">
Snap or upload your <b style="color:#FFFFFF;">{scanner_note}</b>.<br>
The AI (<b style="color:#007AFF;">Sarvam Vision → Groq fallback</b>) will read the values and
pre-fill every field below automatically. You can review and edit before predicting.
</div>
<div style="margin-top:8px;font-size:11px;color:#636366;">
💡 Tip: Ensure the report is well-lit and the text is clearly visible.
</div>
</div>
""", unsafe_allow_html=True)

        report_file = st.file_uploader(
            "Upload Report Image or PDF",
            type=["jpg", "jpeg", "png", "pdf"],
            key=f"report_scan_{model_type}",
            label_visibility="collapsed",
        )

        if report_file:
            col_prev, col_btn = st.columns([2, 3])
            with col_prev:
                st.image(report_file, caption="Report Preview", use_container_width=True)
            with col_btn:
                st.markdown("""
<div style="background:#111111;border:1px solid #2C2C2E;border-radius:10px;padding:12px 14px;margin-bottom:10px;font-size:12px;color:#A1A1A6;line-height:1.7;">
🤖 AI will scan this image and extract all measurable values, then pre-fill the form fields below.<br><br>
<span style="color:#636366;">You can correct any mis-read values before clicking Predict.</span>
</div>
""", unsafe_allow_html=True)
                if st.button(
                    "🔬 Extract & Auto-Fill Fields",
                    type="primary",
                    use_container_width=True,
                    key=f"btn_scan_{model_type}",
                ):
                    with st.spinner("AI scanning report…"):
                        try:
                            suffix = Path(getattr(report_file, "name", "report.jpg")).suffix or ".jpg"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                tmp.write(report_file.read())
                                tmp_path = tmp.name
                            img_b64, mime_type = load_image_b64(tmp_path)
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                            extracted = ocr_ml_report(img_b64, mime_type, model_type)
                            filled = {k: v for k, v in extracted.items() if v is not None}
                            _apply_ml_ocr_to_session(model_type, filled)
                            count = len(filled)
                            st.success(f"✅ Extracted {count} value(s)! Fields are pre-filled — review and edit if needed, then click Predict.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not read report: {e}. Please enter values manually below.")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Deep Clinical Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def analyze_with_sarvam(prompt: str) -> dict:
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    response = client.chat.completions(model="sarvam-105b", messages=[{"role": "user", "content": prompt}])
    content = response.choices[0].message.content
    if not content: raise ValueError("Sarvam returned an empty response (safety filter trigger).")
    return extract_json(content.strip())

def analyze_with_groq(prompt: str) -> dict:
    if not GROQ_API_KEY: raise RuntimeError("GROQ API Key missing for fallback.")
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=60,
    )
    if resp.status_code != 200: raise RuntimeError(f"Groq Analysis Error: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    if not content: raise ValueError("Groq returned empty content.")
    return extract_json(content)

def execute_analysis_pipeline(ocr_text: str, language: str = "English") -> dict:
    prompt = ANALYSIS_PROMPT + ocr_text.strip()
    
    lang_map = {
        "हिंदी": "Hindi",
        "Tamil": "Tamil",
        "Telugu": "Telugu",
        "Bengali": "Bengali",
        "Kannada": "Kannada",
        "Marathi": "Marathi",
    }
    
    if language != "English":
        target_lang = lang_map.get(language, "Hindi")
        prompt += f"\n\nCRITICAL INSTRUCTION: You MUST provide ALL patient-facing content in {target_lang} using native script (Devanagari for Hindi, Tamil script for Tamil, Telugu script for Telugu, Bengali script for Bengali, Kannada script for Kannada, Marathi script for Marathi). This includes: purpose, sideEffects, warning, generalAdvice, doctorNotes, and all text in interactions and foodInteractions arrays. ONLY keep medication/drug names in English for clarity."

    try:
        return analyze_with_sarvam(prompt)
    except Exception as e_sarvam:
        print(f"Sarvam Analysis Failed ({e_sarvam}). Deploying Groq Fallback...")
        try:
            return analyze_with_groq(prompt)
        except Exception as e_groq:
            raise RuntimeError(f"Both analysis engines failed.\nSarvam Error: {e_sarvam}\nGroq Error: {e_groq}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Medicine Bottle Scanner & Verification
# ─────────────────────────────────────────────────────────────────────────────
def extract_medicine_name_from_image(image_b64: str, mime: str) -> str:
    """Extract medicine name from bottle/package image using OCR"""
    prompt = """Look at this medicine bottle, packet, or strip image carefully.

Your task: Find and extract ONLY the main active medicine/drug name - the most prominently displayed ingredient name on the label.

Search order:
1. Large bold text (usually at top of label)
2. Medicine name in caps or bold
3. NOT the manufacturer name, NOT dosage numbers, NOT dates
4. First extract the name in ENGLISH

Examples:
- "Paracetamol 500mg tablets" → "Paracetamol"
- "ASPIRIN BP" → "Aspirin"  
- "IBUPROFEN + PARACETAMOL" → "Ibuprofen"
- Strip showing "CETIRIZINE 10mg" → "Cetirizine"
- "Amoxicillin Trihydrate" → "Amoxicillin"

Return ONLY the medicine name in 1-3 words, in English. If you cannot read any medicine name clearly, respond with "UNCLEAR". NO EXPLANATIONS."""
    
    try:
        # Try Sarvam first
        resp = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
            json={
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                ]}],
                "max_tokens": 50,
                "temperature": 0.05,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            result = resp.json()["choices"][0]["message"]["content"].strip()
            if result and result.upper() != "UNCLEAR" and len(result.strip()) > 1:
                # Clean up response
                result = result.strip('*"\'`.,')
                print(f"✓ Medicine name extracted (Sarvam): {result}")
                return result
    except Exception as e:
        print(f"⚠ Sarvam extraction: {str(e)[:60]}")
    
    # Fallback to Groq with better model
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.2-90b-vision-preview",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}
                ]}],
                "max_tokens": 50,
                "temperature": 0.05,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            result = resp.json()["choices"][0]["message"]["content"].strip()
            if result and result.upper() != "UNCLEAR" and len(result.strip()) > 1:
                result = result.strip('*"\'`.,')
                print(f"✓ Medicine name extracted (Groq): {result}")
                return result
    except Exception as e:
        print(f"⚠ Groq extraction: {str(e)[:60]}")
    
    return None

def get_medicine_info(medicine_name: str, respond_in_hindi: bool = False) -> dict:
    """Get detailed info about a specific medicine including generics and pricing"""
    prompt = f"""You are a trusted Indian pharmacist. A patient has a medicine called "{medicine_name}".

Provide ONLY a valid JSON object with this structure:
{{
  "medications": [
    {{
      "name": "{medicine_name}",
      "prescribedPrice": "Approximate retail price in India e.g. ₹120",
      "genericName": "Generic/salt name e.g. Paracetamol",
      "genericPrice": "Cheapest generic price e.g. ₹20",
      "purpose": "What this medicine does in 1-2 plain sentences",
      "dosage": "Common dosage e.g. 500mg",
      "frequency": "Typical usage e.g. As needed / 3 times daily",
      "timing": "When to take e.g. After meals",
      "sideEffects": ["most common", "second common", "third common"],
      "warning": "One key safety warning or null"
    }}
  ],
  "interactions": {{
    "text": "Common drug interactions to watch for",
    "severity": "minor"
  }},
  "foodInteractions": [
    {{
      "medicine": "{medicine_name}",
      "food": "Food/beverage to avoid if any",
      "reason": "Why this matters",
      "advice": "Practical guidance"
    }}
  ],
  "generalAdvice": "Storage and usage tips specific to this medicine. Include: Store in cool, dry place. Keep away from children.",
  "doctorNotes": "When to stop and see doctor or null"
}}

No markdown, no explanations, ONLY the JSON object."""
    
    if respond_in_hindi:
        prompt += "\n\nWrite purpose, sideEffects, warning, generalAdvice in Hindi (Devanagari). Keep medicine names and metrics in English."
    
    try:
        return analyze_with_sarvam(prompt)
    except Exception as e:
        print(f"Medicine info error (Sarvam): {e}")
        try:
            return analyze_with_groq(prompt)
        except Exception as e2:
            print(f"Medicine info error (Groq): {e2}")
            # Return a template response with the medicine name when AI fails
            return {
                "medications": [{
                    "name": medicine_name,
                    "prescribedPrice": "₹—",
                    "genericName": "Check with pharmacist",
                    "genericPrice": "₹—",
                    "purpose": "Please consult your doctor or pharmacist for detailed information about this medicine.",
                    "dosage": "—",
                    "frequency": "—",
                    "timing": "—",
                    "sideEffects": ["Consult pharmacist for full list"],
                    "warning": "Always read the package insert and follow pharmacist instructions"
                }],
                "interactions": {
                    "text": "Consult pharmacist about possible drug interactions with other medicines",
                    "severity": "minor"
                },
                "foodInteractions": [],
                "generalAdvice": f"Store {medicine_name} as per package instructions. Never share with others.",
                "doctorNotes": "Consult your doctor if symptoms persist"
            }

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Auto-Save Lab Report from Prescription Analysis
# ─────────────────────────────────────────────────────────────────────────────
def auto_save_prescription_as_report(result: dict, ocr_text: str):
    """Automatically save prescription analysis as a lab report entry."""
    if "lab_reports" not in st.session_state:
        st.session_state.lab_reports = []
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = "Self"
    
    # Create report entry with prescription details
    report_entry = {
        "date": datetime.now().isoformat(),
        "type": "Prescription Analysis",
        "test_name": "Medication Prescription",
        "profile": st.session_state.current_profile,
        "medications": [med.get("name", "") for med in result.get("medications", [])],
        "value": f"{len(result.get('medications', []))} medications",
        "reference": "Clinical prescription",
        "notes": f"OCR Source: {st.session_state.get('ocr_source', 'Unknown')}\nSaved via auto-capture system",
        "ocr_text": ocr_text[:500] if ocr_text else ""  # Store first 500 chars
    }
    
    st.session_state.lab_reports.append(report_entry)

# ─────────────────────────────────────────────────────────────────────────────
# UI Card Renderers
# ─────────────────────────────────────────────────────────────────────────────
def _drug_card_html(med: dict, food_interactions: list = None) -> str:
    """Render medicine card with integrated food-drug interactions"""
    presc_query = urllib.parse.quote(med.get("name", ""))
    generic_query = urllib.parse.quote(med.get("genericName", ""))
    order_prescribed_url = f"https://www.1mg.com/search/all?name={presc_query}"
    order_generic_url = f"https://www.1mg.com/search/all?name={generic_query}"
    jan_aushadhi_url = "https://www.google.com/maps/search/?api=1&query=Jan+Aushadhi+Kendra"
    
    # Drug authenticity links
    auth_links = get_authenticity_check_links(med.get("name", ""))

    side_effects_html = "".join(
        f'<span style="font-size:11px;background:#2C2C2E;padding:4px 12px;border-radius:18px;color:#A1A1A6;margin:0 6px 6px 0;display:inline-block;font-weight:500;">{se}</span>'
        for se in (med.get("sideEffects") or [])
    )

    warning_html = ""
    if med.get("warning"):
        warning_html = f'<div style="margin-top:14px;background:rgba(255,214,10,0.08);border:1px solid rgba(255,214,10,0.25);border-radius:10px;padding:11px 13px;font-size:12px;color:#FFD60A;display:flex;gap:8px;align-items:start;"><span style="color:#FFD60A;flex-shrink:0;margin-top:1px;">{SVG_ICONS["warning"]}</span><div style="line-height:1.5;">{med["warning"]}</div></div>'

    # Food-drug interactions for this specific medicine
    food_warning_html = ""
    if food_interactions:
        med_food_interactions = [f for f in food_interactions if f.get("medicine") == med.get("name")]
        if med_food_interactions:
            food_warning_html = '<div style="margin-top:14px;background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.2);border-radius:10px;padding:10px 13px;">'
            for fi in med_food_interactions:
                food_warning_html += f'''<div style="font-size:12px;color:#FFD60A;margin-bottom:6px;"><strong style="color:#FFD60A;">🍽️ {fi.get("food", "")}</strong><div style="font-size:11px;color:#A1A1A6;margin-top:2px;">{fi.get("reason", "")}</div><div style="font-size:11px;color:#FFD60A;font-weight:500;margin-top:2px;">→ {fi.get("advice", "")}</div></div>'''
            food_warning_html += '</div>'

    info_cells = "".join(
        f'<div style="background:#1C1C1E;border-radius:10px;padding:11px 5px;text-align:center;"><div style="color:#A1A1A6;margin-bottom:5px;display:flex;justify-content:center;font-size:14px;">{icon}</div><div style="font-size:8px;color:#8E8E93;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px;">{label}</div><div style="font-size:13px;font-weight:600;color:#FFFFFF;">{val or "—"}</div></div>'
        for icon, label, val in [
            (SVG_ICONS["clock"], "Timing", med.get("timing")),
            (SVG_ICONS["calendar"], "Duration", med.get("duration")),
            (SVG_ICONS["pill"], "Dose", med.get("dosage")),
        ]
    )

    return f"""
<div style="background:#111111;border:1px solid #222222;border-radius:16px;padding:16px;margin-bottom:16px;box-shadow:0 4px 16px rgba(0,0,0,0.2);">
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;gap:10px;">
<div style="min-width:0;flex:1;">
<div style="font-weight:700;color:#FFFFFF;font-size:18px;margin-bottom:4px;">{med.get("name","")}</div>
<div style="font-size:12px;color:#A1A1A6;">₹{med.get("prescribedPrice","—").split('/')[0]} → <span style="color:#30D158;font-weight:600;">₹{med.get("genericPrice","—").split('/')[0]} Generic</span></div>
</div>
<span style="background:#007AFF;color:#FFFFFF;font-size:11px;font-weight:600;padding:5px 11px;border-radius:16px;white-space:nowrap;">{med.get("frequency","")}</span>
</div>
<div style="font-size:13px;color:#E5E5EA;line-height:1.5;margin-bottom:12px;">{med.get("purpose","")}</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">{info_cells}</div>
<div style="margin-bottom:8px;"><div style="font-size:9px;font-weight:700;color:#7a8a9a;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Side Effects</div><div>{side_effects_html}</div></div>
{warning_html}{food_warning_html}
<div style="display:flex;gap:8px;margin-top:12px;flex-direction:column;">
<div style="display:flex;gap:8px;">
<a href="{order_prescribed_url}" target="_blank" style="text-decoration:none;flex:1;color:white !important;"><div style="background:#2C2C2E;color:#000000;text-align:center;padding:10px;border-radius:12px;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;">{SVG_ICONS["cart"]} Order</div></a>
<a href="{order_generic_url}" target="_blank" style="text-decoration:none;flex:1;color:white !important;"><div style="background:#2C2C2E;color:#FFFFFF;text-align:center;padding:10px;border-radius:12px;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;">{SVG_ICONS["cart"]} Generic</div></a>
<a href="{auth_links['1mg']}" target="_blank" style="text-decoration:none;flex:1;color:white !important;"><div style="background:#1C1C1E;border:1px solid #FFD60A;color:#FFD60A;text-align:center;padding:10px;border-radius:12px;font-size:13px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;">{SVG_ICONS["info"]} Check</div></a>
</div>
<a href="{jan_aushadhi_url}" target="_blank" style="text-decoration:none;width:100%;color:white !important;"><div style="background:#1C1C1E;border:1px solid #333333;color:#FFFFFF;text-align:center;padding:10px;border-radius:12px;font-size:12px;font-weight:600;display:flex;align-items:center;justify-content:center;gap:6px;">{SVG_ICONS["map"]} Jan Aushadhi Kendra (70%+ off)</div></a>
</div>
</div>
"""

def render_results(result: dict, ocr_text: str = ""):
    # Interaction warning with severity grading
    if result.get("interactions"):
        interactions = result.get("interactions", {})
        if isinstance(interactions, dict):
            text = interactions.get("text", "")
            severity = interactions.get("severity", "none")
        else:
            text = interactions
            severity = "none"
        
        severity_colors = {
            "none": ("#28a745", "✅ No Critical Interactions"),
            "minor": ("#ffc107", "⚠ Minor Interaction"),
            "moderate": ("#fd7e14", "⚠ Moderate Interaction"),
            "severe": ("#dc3545", "🔴 Severe Interaction"),
            "contraindicated": ("#721c24", "🚫 CONTRAINDICATED")
        }
        
        color, title = severity_colors.get(severity, ("#999999", "⚠ Interaction"))
        
        if text and text != "No dangerous structural drug interactions detected.":
            st.markdown(f"""
<div style="background:rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1);border:2px solid {color};border-radius:16px;padding:16px 18px;margin-bottom:24px;display:flex;gap:12px;align-items:flex-start;">
<span style="color:{color};margin-top:2px;font-size:18px;">{SVG_ICONS["warning"]}</span>
<div><div style="font-weight:700;color:{color};font-size:15px;margin-bottom:4px;">{title}</div><div style="font-size:14px;color:#E5E5EA;line-height:1.6;">{text}</div></div>
</div>
""", unsafe_allow_html=True)

    # Render each medication with integrated food interactions
    food_interactions = result.get("foodInteractions", [])
    for med in result.get("medications", []):
        st.markdown(_drug_card_html(med, food_interactions), unsafe_allow_html=True)
    
    # NEW: Cost Savings Calculator
    savings_info = calculate_total_savings(result)
    if savings_info["total_savings"] > 0:
        st.markdown(f"""
<div style="background:linear-gradient(135deg, #1C1C1E 0%, #111111 100%);border:2px solid #30D158;border-radius:16px;padding:20px;margin-bottom:24px;text-align:center;">
<div style="font-size:13px;color:#A1A1A6;margin-bottom:8px;">💰 SWITCH TO GENERICS AND SAVE</div>
<div style="font-size:32px;font-weight:700;color:#30D158;margin-bottom:2px;">₹{int(savings_info["total_savings"])}</div>
<div style="font-size:12px;color:#A1A1A6;">on this prescription ({savings_info["percentage"]}% discount)</div>
<div style="margin-top:12px;font-size:11px;color:#8E8E93;">Your pharma: ₹{int(savings_info["total_prescribed"])} → Generics: ₹{int(savings_info["total_generic"])}</div>
</div>
""", unsafe_allow_html=True)

    if result.get("generalAdvice"):
        st.markdown(f"""
<div style="background:#111111;border:1px solid #333333;border-radius:16px;padding:20px;margin-bottom:16px;">
<div style="font-weight:600;color:#FFFFFF;margin-bottom:8px;font-size:15px;">Lifestyle Guidance</div><div style="font-size:14px;color:#A1A1A6;line-height:1.6;">{result["generalAdvice"]}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<p style="text-align:center;font-size:12px;color:#6E6E73;padding:16px 20px 0;">
RxPlain is engineered for computational educational lookups only. Always follow physical medical oversight.
</p>
""", unsafe_allow_html=True)

# Centered Modal Overlay for Loading WITH Progress Bar
# NOTE: ALL indentation removed below so Streamlit does not parse it as a code block.
def _loading_card(icon_svg: str, title: str, subtitle: str):
    st.markdown(f"""
<div style="position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.65); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); z-index:99999; display:flex; justify-content:center; align-items:center;">
<div style="background:#111111; border:1px solid #333333; border-radius:24px; padding:40px 32px; text-align:center; max-width:360px; width:90%; box-shadow:0 24px 48px rgba(0,0,0,0.6); animation: fadein 0.3s ease;">
<div style="color:#007AFF; margin-bottom:20px; display:flex; justify-content:center;">
<div style="width:48px; height:48px;">{icon_svg}</div>
</div>
<div style="font-weight:600; color:#FFFFFF; font-size:18px; margin-bottom:8px;">{title}</div>
<div style="font-size:14px; color:#A1A1A6; line-height:1.6; margin-bottom: 24px;">{subtitle}</div>
<div style="width:100%; height:4px; background:#2C2C2E; border-radius:2px; overflow:hidden; position:relative;">
<div style="position:absolute; top:0; left:0; height:100%; width:30%; background:#007AFF; border-radius:2px; animation: indeterminate_progress 1.5s infinite ease-in-out;"></div>
</div>
</div>
</div>
<style>
@keyframes fadein {{ from {{ opacity:0; transform:scale(0.95); }} to {{ opacity:1; transform:scale(1); }} }}
@keyframes indeterminate_progress {{
0% {{ left: -30%; width: 30%; }}
50% {{ width: 50%; }}
100% {{ left: 100%; width: 30%; }}
}}
</style>
""", unsafe_allow_html=True)

def _step_badge(n: int, label: str, active: bool = False) -> str:
    bg = "#007AFF" if active else "#1C1C1E"
    color = "#FFFFFF" if active else "#8E8E93"
    border = "none" if active else "1px solid #333333"
    num_bg = "#FFFFFF" if active else "#333333"
    num_color = "#007AFF" if active else "#FFFFFF"
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;background:{bg};border:{border};border-radius:20px;padding:6px 14px 6px 8px;">'
        f'<span style="background:{num_bg};color:{num_color} !important;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">{n}</span>'
        f'<span style="font-size:13px;color:{color} !important;font-weight:600;">{label}</span>'
        f'</div>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# ML Models — trained on open datasets via Google Colab
# ─────────────────────────────────────────────────────────────────────────────
ML_MODEL_PATHS = {
    "diabetes":            "pickle_model_diabetes.pkl",
    "heart":               "pickle_model_heart.pkl",
    "disease":             "pickle_model_disease.pkl",
    "pneumonia":           "pickle_model_pneumonia.pkl",
    "malaria":             "pickle_model_malaria.pkl",
    "osteoporosis":        "pickle_model_osteoporosis.pkl",
    "stroke":              "pickle_model_stroke.pkl",
    "parkinsons":          "pickle_model_parkinsons.pkl",
    "liver":               "pickle_model_liver.pkl",
    "kidney":              "pickle_model_kidney.pkl",
    "breast_cancer":       "pickle_model_breast_cancer.pkl",
    "tuberculosis":        "pickle_model_tuberculosis.pkl",
    "retinopathy":         "pickle_model_retinopathy.pkl",

    "bone_fracture":       "pickle_model_bone_fracture.pkl",
    "skin_disease":        "pickle_model_skin_disease.pkl",

    # ── New models ────────────────────────────────────────────────────────────
    "thyroid":             "thyroid_rf_model.pkl",
    "anaemia":             "pickle_model_anaemia.pkl",
    "pcos":                "pcos_rf_model.pkl",
    # ── Batch 3 ───────────────────────────────────────────────────────────────
    "heart_failure":       "pickle_model_heart_failure.pkl",
    "cervical":            "pickle_model_cervical.pkl",
    "hepatitis":           "pickle_model_hepatitis.pkl",
    "sepsis":              "pickle_model_sepsis.pkl",
}

# 132-symptom vocabulary used by disease.csv / pickle_model_disease.pkl
DISEASE_SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
    'vomiting','burning_micturition','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
    'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
    'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
    'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck',
    'swelling_joints','movement_stiffness','spinning_movements','loss_of_balance',
    'unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
    'foul_smell_of_urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
    'dischromic_patches','watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration',
    'visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections',
    'coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption',
    'blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking',
    'pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting',
    'small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze'
]

SYMPTOM_DISPLAY = [s.replace('_', ' ').title() for s in DISEASE_SYMPTOMS]
SYMPTOM_MAP     = dict(zip(SYMPTOM_DISPLAY, DISEASE_SYMPTOMS))   # display → raw

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pickle(path: str):
    """Load a scikit-learn model saved with pickle or joblib. Returns (model, error_str)."""
    if not os.path.exists(path):
        return None, f"Model file not found: `{path}`. Place the .pkl file in the same directory as app.py."
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except Exception:
        pass
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, f"Failed to load {path}: {e}"

@st.cache_resource(show_spinner=False)
def _load_keras(path: str):
    """Load a Keras .h5 model. Returns (model, error_str)."""
    if not KERAS_AVAILABLE:
        return None, "TensorFlow / Keras not installed. Run: `pip install tensorflow`"
    if not os.path.exists(path):
        return None, f"Model file not found: `{path}`. Place the Keras model file in the same directory as app.py."
    try:
        return keras_load_model(path), None
    except Exception as e:
        return None, f"Failed to load Keras model {path}: {e}"

def _model_missing_card(name: str, err: str):
    st.markdown(f"""
<div style="background:#1C1C1E;border:1px solid #FF453A;border-radius:14px;padding:18px 20px;margin:16px 0;">
<div style="font-weight:700;color:#FF453A;font-size:15px;margin-bottom:6px;">⚠ {name} Model Not Loaded</div>
<div style="font-size:13px;color:#A1A1A6;line-height:1.6;">{err}</div>
<div style="font-size:12px;color:#636366;margin-top:8px;">
📦 Models trained on open datasets using Google Colab
</div>
</div>
""", unsafe_allow_html=True)

def _result_card(label: str, confidence: float, positive: bool, detail: str = ""):
    color  = "#FF453A" if positive else "#30D158"
    icon   = "🔴" if positive else "✅"
    pct    = f"{confidence * 100:.1f}%"
    st.markdown(f"""
<div style="background:#111111;border:2px solid {color};border-radius:16px;padding:20px;margin:16px 0;text-align:center;">
<div style="font-size:36px;margin-bottom:8px;">{icon}</div>
<div style="font-size:22px;font-weight:700;color:{color};margin-bottom:4px;">{label}</div>
<div style="font-size:13px;color:#A1A1A6;margin-bottom:8px;">Model Confidence: <b style="color:#FFFFFF;">{pct}</b></div>
{"<div style='font-size:12px;color:#636366;line-height:1.5;'>"+detail+"</div>" if detail else ""}
</div>
""", unsafe_allow_html=True)
    st.warning("⚕️ This tool is for **educational/screening purposes only**. Always consult a qualified doctor for diagnosis.")


# ─── ML Predictors ────────────────────────────────────────────────────────────

def predict_diabetes(features: list):
    model, err = _load_pickle(ML_MODEL_PATHS["diabetes"])
    if err: _model_missing_card("Diabetes", err); return
    arr = np.array(features).reshape(1, -1)
    pred  = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]
    confidence = proba[int(pred)]
    positive = int(pred) == 1
    label = "Diabetic" if positive else "Non-Diabetic"
    detail = (
        "High blood glucose detected. Please consult an endocrinologist."
        if positive else
        "No diabetes detected. Maintain healthy diet & regular exercise."
    )
    _result_card(label, confidence, positive, detail)


def predict_heart(features: list):
    model, err = _load_pickle(ML_MODEL_PATHS["heart"])
    if err: _model_missing_card("Heart Disease", err); return
    arr = np.array(features).reshape(1, -1)
    pred  = model.predict(arr)[0]
    proba = model.predict_proba(arr)[0]
    confidence = proba[int(pred)]
    positive = int(pred) == 1
    label = "Heart Disease Detected" if positive else "Heart Disease Not Detected"
    detail = (
        "Cardiac risk factors present. Please see a cardiologist immediately."
        if positive else
        "No major cardiac risk detected. Keep up healthy lifestyle habits."
    )
    _result_card(label, confidence, positive, detail)


def predict_disease(selected_display_symptoms: list):
    model, err = _load_pickle(ML_MODEL_PATHS["disease"])
    if err: _model_missing_card("Disease Predictor", err); return
    # Build a 17-element symptom vector matching how disease.csv was trained:
    # the model likely expects a row of 17 symptom string columns
    # We pad/truncate to 17 entries
    raw_symptoms = [SYMPTOM_MAP.get(s, s.lower().replace(' ', '_'))
                    for s in selected_display_symptoms]
    # Pad with empty string if fewer than 17
    while len(raw_symptoms) < 17:
        raw_symptoms.append("")
    raw_symptoms = raw_symptoms[:17]
    arr = np.array(raw_symptoms).reshape(1, -1)
    try:
        pred = model.predict(arr)[0]
        st.markdown(f"""
<div style="background:#111111;border:2px solid #007AFF;border-radius:16px;padding:20px;margin:16px 0;text-align:center;">
<div style="font-size:32px;margin-bottom:8px;">🩺</div>
<div style="font-size:13px;color:#A1A1A6;margin-bottom:6px;">Predicted Condition</div>
<div style="font-size:22px;font-weight:700;color:#FFFFFF;">{pred}</div>
</div>
""", unsafe_allow_html=True)
        st.warning("⚕️ Educational/screening use only. Always consult a qualified doctor.")
    except Exception as e:
        st.error(f"Prediction error: {e}. The symptom encoding may not match this model's training format.")


def predict_pneumonia(image_file):
    """Run pneumonia X-ray prediction using sklearn Pipeline (PCA + RandomForest)."""
    model, err = _load_pickle(ML_MODEL_PATHS["pneumonia"])
    if err: _model_missing_card("Pneumonia", err); return
    try:
        img = Image.open(image_file).convert("RGB").resize((64, 64))
        arr = np.array(img).flatten().reshape(1, -1)
        pred     = model.predict(arr)[0]
        proba    = model.predict_proba(arr)[0]
        confidence = proba[int(pred)]
        positive = int(pred) == 1
        label    = "PNEUMONIA Detected" if positive else "NORMAL"
        detail   = (
            "Chest X-ray pattern suggests pneumonia. Please consult a pulmonologist or physician."
            if positive else
            "Chest X-ray appears normal. No signs of pneumonia detected."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Pneumonia prediction failed: {e}")


def predict_malaria(image_file):
    """Run malaria cell-image prediction using sklearn Pipeline (PCA + RandomForest)."""
    model, err = _load_pickle(ML_MODEL_PATHS["malaria"])
    if err: _model_missing_card("Malaria", err); return
    try:
        img = Image.open(image_file).convert("RGB").resize((64, 64))
        arr = np.array(img).flatten().reshape(1, -1)
        pred     = model.predict(arr)[0]
        proba    = model.predict_proba(arr)[0]
        confidence = proba[int(pred)]
        positive = int(pred) == 1   # 1 = Parasitized
        label    = "PARASITIZED — Malaria Detected" if positive else "UNINFECTED — No Malaria"
        detail   = (
            "Cell image suggests malaria parasite presence. Seek medical attention immediately."
            if positive else
            "Cell image appears uninfected. No malaria parasite detected."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Malaria prediction failed: {e}")


def predict_osteoporosis(features: list):
    """Osteoporosis risk prediction using Pipeline(StandardScaler + RandomForest).
    Features order must match training:
    [Age, Gender, Hormonal Changes, Family History, Body Weight,
     Calcium Intake, Vitamin D Intake, Physical Activity,
     Smoking, Alcohol Consumption, Prior Fractures]
    """
    model, err = _load_pickle(ML_MODEL_PATHS["osteoporosis"])
    if err:
        _model_missing_card("Osteoporosis", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "High Osteoporosis Risk" if positive else "Low Osteoporosis Risk"
        detail = (
            "Bone density loss risk detected. Please consult an orthopaedic or endocrinologist for a DEXA scan."
            if positive else
            "No significant osteoporosis risk detected. Maintain calcium intake, vitamin D, and weight-bearing exercise."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Osteoporosis prediction failed: {e}")

def predict_stroke(features: list):
    """Stroke risk prediction.
    Features: gender, age, hypertension, heart_disease, ever_married,
              work_type, Residence_type, avg_glucose_level, bmi, smoking_status
    """
    model, err = _load_pickle(ML_MODEL_PATHS["stroke"])
    if err:
        _model_missing_card("Stroke", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "High Stroke Risk" if positive else "Low Stroke Risk"
        detail = (
            "Risk factors suggest elevated stroke probability. Please consult a neurologist immediately."
            if positive else
            "No significant stroke risk indicators detected. Maintain a healthy BP, diet, and active lifestyle."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Stroke prediction failed: {e}")


def predict_parkinsons(features: list):
    """Parkinson's disease detection from voice measurements.
    Features: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%),
              MDVP:Shimmer, HNR, RPDE, DFA, spread1, PPE
    """
    model, err = _load_pickle(ML_MODEL_PATHS["parkinsons"])
    if err:
        _model_missing_card("Parkinson's", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "Parkinson's Likely" if positive else "Parkinson's Unlikely"
        detail = (
            "Voice biomarkers suggest Parkinson's disease patterns. Please consult a neurologist for clinical assessment."
            if positive else
            "Voice biomarkers do not suggest Parkinson's disease. Regular check-ups recommended if symptoms persist."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Parkinson's prediction failed: {e}")


def predict_liver(features: list):
    """Liver disease prediction.
    Features: Age, Gender, Total_Bilirubin, Direct_Bilirubin,
              Alkaline_Phosphotase, Alamine_Aminotransferase,
              Aspartate_Aminotransferase, Total_Protiens, Albumin,
              Albumin_and_Globulin_Ratio
    """
    model, err = _load_pickle(ML_MODEL_PATHS["liver"])
    if err:
        _model_missing_card("Liver Disease", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "Liver Disease Detected" if positive else "No Liver Disease Detected"
        detail = (
            "Blood markers indicate possible liver disease. Please consult a hepatologist or gastroenterologist."
            if positive else
            "Liver function markers appear within acceptable range. Maintain a healthy diet and avoid excess alcohol."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Liver prediction failed: {e}")


def predict_kidney(features: list):
    """Chronic Kidney Disease prediction.
    Features: age, bp, bgr (blood glucose), bu (blood urea), sc (serum creatinine),
              sod (sodium), pot (potassium), hemo (haemoglobin), pcv (packed cell vol),
              wc (WBC count), rc (RBC count), htn, dm, ane
    """
    model, err = _load_pickle(ML_MODEL_PATHS["kidney"])
    if err:
        _model_missing_card("Kidney Disease", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "Chronic Kidney Disease Detected" if positive else "No CKD Detected"
        detail = (
            "Indicators suggest chronic kidney disease. Please see a nephrologist for further evaluation."
            if positive else
            "No signs of chronic kidney disease. Stay hydrated and monitor blood pressure regularly."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Kidney prediction failed: {e}")


def predict_breast_cancer(features: list):
    """Breast cancer prediction from tumor measurements (Wisconsin dataset).
    Features (10 mean values): radius, texture, perimeter, area, smoothness,
                               compactness, concavity, concave_points,
                               symmetry, fractal_dimension
    """
    model, err = _load_pickle(ML_MODEL_PATHS["breast_cancer"])
    if err:
        _model_missing_card("Breast Cancer", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "Malignant (Cancer Likely)" if positive else "Benign (Non-cancerous)"
        detail = (
            "Tumor measurements suggest malignancy. Please consult an oncologist immediately for biopsy and clinical confirmation."
            if positive else
            "Tumor measurements appear benign. Regular screenings are still recommended — consult your doctor."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Breast cancer prediction failed: {e}")


def _predict_image_model(key: str, image_file, labels: list[str], positive_idx: int, name: str, detail_pos: str, detail_neg: str):
    """Generic image predict: resize → flatten → PCA+RF pipeline."""
    model, err = _load_pickle(ML_MODEL_PATHS[key])
    if err:
        _model_missing_card(name, err)
        return
    try:
        img = Image.open(image_file).convert("RGB").resize((64, 64))
        arr = np.array(img).flatten().reshape(1, -1)
        pred       = model.predict(arr)[0]
        proba      = model.predict_proba(arr)[0]
        confidence = proba[int(pred)]
        positive   = int(pred) == positive_idx
        _result_card(labels[int(pred)], confidence, positive, detail_pos if positive else detail_neg)
    except Exception as e:
        st.error(f"{name} prediction failed: {e}")


def predict_tuberculosis(image_file):
    _predict_image_model(
        "tuberculosis", image_file,
        labels=["Normal", "Tuberculosis Detected"],
        positive_idx=1, name="Tuberculosis",
        detail_pos="Chest X-Ray pattern suggests TB. Please consult a pulmonologist immediately for sputum culture confirmation.",
        detail_neg="Chest X-Ray appears clear. No signs of tuberculosis detected.",
    )


def predict_retinopathy(image_file):
    _predict_image_model(
        "retinopathy", image_file,
        labels=["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
        positive_idx=1, name="Diabetic Retinopathy",
        detail_pos="Signs of diabetic retinopathy detected. Please consult an ophthalmologist promptly.",
        detail_neg="No signs of diabetic retinopathy detected. Regular annual fundus screening is still recommended.",
    )



def predict_bone_fracture(image_file):
    _predict_image_model(
        "bone_fracture", image_file,
        labels=["No Fracture", "Fracture Detected"],
        positive_idx=1, name="Bone Fracture",
        detail_pos="X-Ray pattern suggests a bone fracture. Please seek immediate medical attention.",
        detail_neg="No fracture detected in the X-Ray. If pain persists, consult an orthopaedic specialist.",
    )


def predict_skin_disease(image_file):
    model, err = _load_pickle(ML_MODEL_PATHS["skin_disease"])
    if err:
        _model_missing_card("Skin Disease", err)
        return
    SKIN_LABELS = {
        0: "Actinic Keratosis",    1: "Basal Cell Carcinoma",
        2: "Benign Keratosis",     3: "Dermatofibroma",
        4: "Melanoma",             5: "Melanocytic Nevi",
        6: "Vascular Lesion",
    }
    MALIGNANT = {0, 1, 4}
    try:
        img = Image.open(image_file).convert("RGB").resize((64, 64))
        arr = np.array(img).flatten().reshape(1, -1)
        pred       = model.predict(arr)[0]
        proba      = model.predict_proba(arr)[0]
        confidence = proba[int(pred)]
        label      = SKIN_LABELS.get(int(pred), f"Class {pred}")
        positive   = int(pred) in MALIGNANT
        detail = (
            f"Lesion resembles {label}. This may require urgent dermatological evaluation — please consult a dermatologist."
            if positive else
            f"Lesion resembles {label}. Generally benign, but any changing or growing lesion should be evaluated by a dermatologist."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Skin disease prediction failed: {e}")


def predict_thyroid(features: list):
    """Thyroid disease prediction.
    Features: age, sex, TSH, T3, TT4, T4U, FTI,
              on_thyroxine, on_antithyroid_meds, sick, pregnant, thyroid_surgery
    Classes: 0=Normal, 1=Hypothyroid, 2=Hyperthyroid
    """
    model, err = _load_pickle(ML_MODEL_PATHS["thyroid"])
    if err:
        _model_missing_card("Thyroid Disease", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        confidence = proba[int(pred)]
        LABELS = {0: "Normal", 1: "Hypothyroidism Likely", 2: "Hyperthyroidism Likely"}
        label    = LABELS.get(int(pred), f"Class {pred}")
        positive = int(pred) != 0
        detail_map = {
            0: "Thyroid hormone levels appear normal. Maintain regular annual check-ups.",
            1: "Values suggest an underactive thyroid (hypothyroidism). Please consult an endocrinologist for TSH confirmation and treatment.",
            2: "Values suggest an overactive thyroid (hyperthyroidism). Please consult an endocrinologist promptly.",
        }
        detail = detail_map.get(int(pred), "")
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Thyroid prediction failed: {e}")


def predict_anaemia(features: list):
    """Anaemia prediction.
    Features: Gender, Haemoglobin, MCH, MCHC, MCV
    """
    model, err = _load_pickle(ML_MODEL_PATHS["anaemia"])
    if err:
        _model_missing_card("Anaemia", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "Anaemia Detected" if positive else "No Anaemia Detected"
        detail = (
            "Blood parameters suggest anaemia. Please consult a doctor for iron studies, B12, and folate levels."
            if positive else
            "Blood parameters appear within normal range. No anaemia detected. Maintain iron-rich diet."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"Anaemia prediction failed: {e}")


def predict_pcos(features: list):
    """PCOS (Polycystic Ovary Syndrome) prediction.
    Features: Age, BMI, FSH, LH, AMH, cycle_length, follicle_no_R, follicle_no_L,
              endometrium, cycle_irregularity, skin_darkening, hair_growth,
              pimples, hair_loss, weight_gain
    """
    model, err = _load_pickle(ML_MODEL_PATHS["pcos"])
    if err:
        _model_missing_card("PCOS", err)
        return
    try:
        arr  = np.array(features, dtype=float).reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        positive   = int(pred) == 1
        confidence = proba[int(pred)]
        label  = "PCOS Detected" if positive else "No PCOS Detected"
        detail = (
            "Hormonal and ultrasound parameters are consistent with PCOS. Please consult a gynaecologist or endocrinologist for clinical confirmation and management."
            if positive else
            "Parameters do not strongly suggest PCOS at this time. Maintain a healthy weight and regular menstrual tracking. Consult a doctor if symptoms persist."
        )
        _result_card(label, confidence, positive, detail)
    except Exception as e:
        st.error(f"PCOS prediction failed: {e}")


def render_ml_tab():
    """Render the Machine Learning & Deep Learning disease predictor tab."""
    st.markdown("""
<div style="background:#111111;border:1px solid #333333;border-radius:16px;padding:18px 20px;margin-bottom:20px;">
<div style="font-weight:700;color:#FFFFFF;font-size:16px;margin-bottom:6px;">🤖 AI Disease Predictor</div>
<div style="font-size:13px;color:#A1A1A6;line-height:1.6;">
Powered by custom ML models trained on open datasets using Google Colab.
Place the <code style="background:#2C2C2E;padding:2px 6px;border-radius:4px;color:#30D158;">.pkl</code>
model files in the same directory as <code style="background:#2C2C2E;padding:2px 6px;border-radius:4px;color:#30D158;">app.py</code> to activate.
</div>
</div>
""", unsafe_allow_html=True)

    # Sub-tabs for each predictor
    sub_tabs = st.tabs([
        "Diabetes", "Heart", "Disease", "Pneumonia", "Malaria",
        "Osteoporosis", "Stroke", "Parkinson's", "Liver", "Kidney",
        "Breast Cancer", "Tuberculosis", "Retinopathy", "Bone Fracture",
        "Skin Disease", "Thyroid", "Anaemia", "PCOS", "Heart Failure",
        "Cervical Cancer", "Hepatitis", "Sepsis", "About Models"
    ])

    # ── TAB 1: DIABETES ──────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.markdown("#### 🩸 Diabetes Prediction")
        st.caption("Model: `pickle_model_diabetes.pkl` | Dataset: `diabetes.csv` | Algorithm: Random Forest / SVM")
        _render_report_scanner("diabetes", "diabetes blood test / HbA1c report / glucose tolerance test")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            preg    = st.number_input("Pregnancies", 0, 20, 1, key="db_preg")
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120, key="db_gluc")
            bp      = st.number_input("Blood Pressure (mmHg)", 0, 180, 70, key="db_bp")
            skin    = st.number_input("Skin Thickness (mm)", 0, 100, 20, key="db_skin")
        with c2:
            insulin = st.number_input("Insulin (µU/mL)", 0, 1000, 80, key="db_ins")
            bmi     = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1, key="db_bmi")
            dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01, key="db_dpf")
            age     = st.number_input("Age", 15, 100, 30, key="db_age")

        if st.button("🔍 Predict Diabetes", type="primary", use_container_width=True, key="btn_diab"):
            predict_diabetes([preg, glucose, bp, skin, insulin, bmi, dpf, age])

    # ── TAB 2: HEART ─────────────────────────────────────────────────────────
    with sub_tabs[1]:
        st.markdown("#### 🫀 Heart Disease Prediction")
        st.caption("Model: `pickle_model_heart.pkl` | Dataset: `heart.csv` | Algorithm: Logistic Regression / RF")
        _render_report_scanner("heart", "ECG report / cardiac blood test / lipid profile / echo report")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            h_age    = st.number_input("Age", 20, 100, 45, key="ht_age")
            h_sex    = st.selectbox("Sex", ["Male (1)", "Female (0)"], key="ht_sex")
            h_cp     = st.selectbox("Chest Pain Type", [
                "0 – Typical Angina", "1 – Atypical Angina",
                "2 – Non-anginal Pain", "3 – Asymptomatic"
            ], key="ht_cp")
            h_trest  = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 130, key="ht_trest")
            h_chol   = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200, key="ht_chol")
            h_fbs    = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No (0)", "Yes (1)"], key="ht_fbs")
            h_recg   = st.selectbox("Resting ECG", [
                "0 – Normal", "1 – ST-T Abnormality", "2 – LV Hypertrophy"
            ], key="ht_recg")
        with c2:
            h_thal   = st.number_input("Max Heart Rate Achieved", 60, 220, 150, key="ht_thal_hr")
            h_exang  = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"], key="ht_exang")
            h_oldp   = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1, key="ht_oldp")
            h_slope  = st.selectbox("Slope of ST Segment", [
                "0 – Upsloping", "1 – Flat", "2 – Downsloping"
            ], key="ht_slope")
            h_ca     = st.selectbox("Major Vessels (Fluoroscopy)", [0, 1, 2, 3], key="ht_ca")
            h_thalv  = st.selectbox("Thalassemia", [
                "0 – Normal", "1 – Fixed Defect", "2 – Reversable Defect", "3 – Unknown"
            ], key="ht_thalv")

        def _first_int(s): return int(s.split("–")[0].strip().split("(")[0].strip().rstrip(")").strip())

        if st.button("🔍 Predict Heart Disease", type="primary", use_container_width=True, key="btn_heart"):
            features = [
                h_age,
                1 if h_sex.startswith("Male") else 0,
                _first_int(h_cp),
                h_trest,
                h_chol,
                1 if h_fbs.startswith("Yes") else 0,
                _first_int(h_recg),
                h_thal,
                1 if h_exang.startswith("Yes") else 0,
                h_oldp,
                _first_int(h_slope),
                int(h_ca),
                _first_int(h_thalv),
            ]
            predict_heart(features)

    # ── TAB 3: DISEASE ───────────────────────────────────────────────────────
    with sub_tabs[2]:
        st.markdown("#### 🦠 General Disease Prediction")
        st.caption("Model: `pickle_model_disease.pkl` | Dataset: `disease.csv` | Algorithm: Decision Tree / RF")
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Select <b style="color:#FFFFFF;">up to 17 symptoms</b> you are currently experiencing. The model will predict the most likely condition.
</div>
""", unsafe_allow_html=True)

        selected_symptoms = st.multiselect(
            "Select Symptoms",
            options=SYMPTOM_DISPLAY,
            max_selections=17,
            placeholder="Search and select symptoms…",
            key="dis_symptoms"
        )
        if selected_symptoms:
            st.caption(f"✅ {len(selected_symptoms)} symptom(s) selected")

        if st.button("🔍 Predict Disease", type="primary", use_container_width=True, key="btn_disease",
                     disabled=len(selected_symptoms) == 0):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                predict_disease(selected_symptoms)

    # ── TAB 4: PNEUMONIA ─────────────────────────────────────────────────────
    with sub_tabs[3]:
        st.markdown("#### 🫁 Pneumonia Detection from Chest X-Ray")
        st.caption("Model: `pickle_model_pneumonia.pkl` | Dataset: Kaggle Chest X-Ray | Algorithm: PCA + Random Forest")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">Chest X-Ray image</b> (JPG/PNG). The model predicts
<span style="color:#FF453A;font-weight:600;">PNEUMONIA</span> or <span style="color:#30D158;font-weight:600;">NORMAL</span>.
<br><br>
📊 Dataset:
<a href="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia" target="_blank" style="color:#007AFF;">
Chest X-Ray Images (Pneumonia) — Kaggle
</a>
</div>
""", unsafe_allow_html=True)

        pneu_file = st.file_uploader(
            "Upload Chest X-Ray", type=["jpg", "jpeg", "png"],
            key="pneu_upload"
        )
        if pneu_file:
            col_img, col_btn = st.columns([1, 1])
            with col_img:
                st.image(pneu_file, caption="Uploaded X-Ray", use_container_width=True)
            with col_btn:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                if st.button("🔍 Analyse X-Ray", type="primary", use_container_width=True, key="btn_pneu"):
                    with st.spinner("Running CNN analysis…"):
                        predict_pneumonia(pneu_file)

    # ── TAB 5: MALARIA ───────────────────────────────────────────────────────
    with sub_tabs[4]:
        st.markdown("#### 🦟 Malaria Detection from Cell Image")
        st.caption("Model: `pickle_model_malaria.pkl` | Dataset: Kaggle Malaria Cell Images | Algorithm: PCA + Random Forest")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">blood cell microscope image</b> (JPG/PNG). The model predicts
<span style="color:#FF453A;font-weight:600;">PARASITIZED</span> or <span style="color:#30D158;font-weight:600;">UNINFECTED</span>.
<br><br>
📊 Dataset:
<a href="https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria" target="_blank" style="color:#007AFF;">
Cell Images for Detecting Malaria — Kaggle
</a>
</div>
""", unsafe_allow_html=True)

        mal_file = st.file_uploader(
            "Upload Cell Image", type=["jpg", "jpeg", "png"],
            key="mal_upload"
        )
        if mal_file:
            col_img, col_btn = st.columns([1, 1])
            with col_img:
                st.image(mal_file, caption="Uploaded Cell Image", use_container_width=True)
            with col_btn:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                if st.button("🔍 Analyse Cell", type="primary", use_container_width=True, key="btn_mal"):
                    with st.spinner("Running analysis…"):
                        predict_malaria(mal_file)

    # ── TAB 6: OSTEOPOROSIS ──────────────────────────────────────────────────
    with sub_tabs[5]:
        st.markdown("#### 🦴 Osteoporosis Risk Prediction")
        st.caption("Model: `pickle_model_osteoporosis.pkl` | Dataset: Kaggle Lifestyle Factors | 14 features | RF Pipeline")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Assess bone density loss risk based on <b style="color:#FFFFFF;">14 lifestyle and clinical factors</b>.
📊 <a href="https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis"
target="_blank" style="color:#007AFF;">Kaggle — Lifestyle Factors Influencing Osteoporosis</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("osteoporosis", "DEXA scan report / bone density test / patient history form")

        c1, c2 = st.columns(2)
        with c1:
            os_age   = st.number_input("Age", 20, 100, 50, key="os_age")
            os_sex   = st.selectbox("Gender", ["Female (0)", "Male (1)"], key="os_sex")
            os_horm  = st.selectbox("Hormonal Changes", ["Normal (0)", "Postmenopausal (1)"], key="os_horm")
            os_fam   = st.selectbox("Family History", ["No (0)", "Yes (1)"], key="os_fam")
            os_race  = st.selectbox("Race / Ethnicity", [
                "African American (0)", "Asian (1)", "Caucasian (2)"
            ], key="os_race")
            os_bw    = st.selectbox("Body Weight", [
                "Normal (0)", "Overweight (1)", "Underweight (2)"
            ], key="os_bw")
            os_calc  = st.selectbox("Calcium Intake", ["Adequate (0)", "Low (1)"], key="os_calc")
        with c2:
            os_vitd  = st.selectbox("Vitamin D Intake", ["Sufficient (0)", "Insufficient (1)"], key="os_vitd")
            os_phys  = st.selectbox("Physical Activity", ["Active (0)", "Sedentary (1)"], key="os_phys")
            os_smok  = st.selectbox("Smoking", ["No (0)", "Yes (1)"], key="os_smok")
            os_alc   = st.selectbox("Alcohol Consumption", ["None (0)", "Moderate (1)"], key="os_alc")
            os_cond  = st.selectbox("Medical Conditions", [
                "None (0)", "Hyperthyroidism (1)", "Rheumatoid Arthritis (2)"
            ], key="os_cond")
            os_meds  = st.selectbox("Medications", ["None (0)", "Corticosteroids (1)"], key="os_meds")
            os_frac  = st.selectbox("Prior Fractures", ["No (0)", "Yes (1)"], key="os_frac")

        def _os_val(s): return int(s.rstrip(")").split("(")[-1])

        if st.button("🔍 Predict Osteoporosis Risk", type="primary", use_container_width=True, key="btn_osteo"):
            features = [
                os_age,
                _os_val(os_sex),
                _os_val(os_horm),
                _os_val(os_fam),
                _os_val(os_race),
                _os_val(os_bw),
                _os_val(os_calc),
                _os_val(os_vitd),
                _os_val(os_phys),
                _os_val(os_smok),
                _os_val(os_alc),
                _os_val(os_cond),
                _os_val(os_meds),
                _os_val(os_frac),
            ]
            predict_osteoporosis(features)

    # ── TAB 7: STROKE ────────────────────────────────────────────────────────
    with sub_tabs[6]:
        st.markdown("#### 🧠 Stroke Risk Prediction")
        st.caption("Model: `pickle_model_stroke.pkl` | Dataset: Kaggle Healthcare Stroke | 10 features | StandardScaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Predict stroke risk based on <b style="color:#FFFFFF;">10 clinical and lifestyle factors</b>.
📊 <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"
target="_blank" style="color:#007AFF;">Kaggle — Stroke Prediction Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("stroke", "health checkup report / blood test / clinical summary")

        def _str_val(s): return int(s.rstrip(")").split("(")[-1])

        c1, c2 = st.columns(2)
        with c1:
            sk_gender   = st.selectbox("Gender", ["Female (0)", "Male (1)", "Other (2)"], key="sk_gender")
            sk_age      = st.number_input("Age", 1, 100, 45, key="sk_age")
            sk_htn      = st.selectbox("Hypertension", ["No (0)", "Yes (1)"], key="sk_htn")
            sk_heart    = st.selectbox("Heart Disease", ["No (0)", "Yes (1)"], key="sk_heart")
            sk_married  = st.selectbox("Ever Married", ["No (0)", "Yes (1)"], key="sk_married")
        with c2:
            sk_work     = st.selectbox("Work Type", [
                "Government Job (0)", "Never Worked (1)", "Private (2)",
                "Self-employed (3)", "Children (4)"
            ], key="sk_work")
            sk_res      = st.selectbox("Residence Type", ["Rural (0)", "Urban (1)"], key="sk_res")
            sk_glucose  = st.number_input("Avg Glucose Level (mg/dL)", 50.0, 300.0, 100.0, step=0.1, key="sk_glucose")
            sk_bmi      = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1, key="sk_bmi")
            sk_smoking  = st.selectbox("Smoking Status", [
                "Unknown (0)", "Formerly Smoked (1)", "Never Smoked (2)", "Smokes (3)"
            ], key="sk_smoking")

        if st.button("🔍 Predict Stroke Risk", type="primary", use_container_width=True, key="btn_stroke"):
            features = [
                _str_val(sk_gender), sk_age, _str_val(sk_htn), _str_val(sk_heart),
                _str_val(sk_married), _str_val(sk_work), _str_val(sk_res),
                sk_glucose, sk_bmi, _str_val(sk_smoking),
            ]
            predict_stroke(features)

    # ── TAB 8: PARKINSON'S ───────────────────────────────────────────────────
    with sub_tabs[7]:
        st.markdown("#### 🎙️ Parkinson's Disease Detection")
        st.caption("Model: `pickle_model_parkinsons.pkl` | Dataset: UCI Parkinson's | 10 voice features | StandardScaler + SVC")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Detect Parkinson's disease from <b style="color:#FFFFFF;">10 voice biomarkers</b> extracted from sustained phonation recordings.
📊 <a href="https://archive.ics.uci.edu/ml/datasets/parkinsons"
target="_blank" style="color:#007AFF;">UCI — Parkinson's Disease Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("parkinsons", "voice analysis report (MDVP / Kay Pentax output)")

        c1, c2 = st.columns(2)
        with c1:
            pk_fo      = st.number_input("MDVP:Fo(Hz) — Avg vocal freq", 80.0, 270.0, 154.2, step=0.1, key="pk_fo")
            pk_fhi     = st.number_input("MDVP:Fhi(Hz) — Max vocal freq", 100.0, 600.0, 197.1, step=0.1, key="pk_fhi")
            pk_flo     = st.number_input("MDVP:Flo(Hz) — Min vocal freq", 60.0, 240.0, 116.3, step=0.1, key="pk_flo")
            pk_jitter  = st.number_input("MDVP:Jitter(%) — Frequency variation", 0.0, 1.0, 0.006, step=0.001, format="%.4f", key="pk_jitter")
            pk_shimmer = st.number_input("MDVP:Shimmer — Amplitude variation", 0.0, 1.0, 0.030, step=0.001, format="%.4f", key="pk_shimmer")
        with c2:
            pk_hnr     = st.number_input("HNR — Harmonics-to-Noise Ratio", 0.0, 35.0, 21.9, step=0.1, key="pk_hnr")
            pk_rpde    = st.number_input("RPDE — Recurrence period density", 0.0, 1.0, 0.50, step=0.001, format="%.4f", key="pk_rpde")
            pk_dfa     = st.number_input("DFA — Signal fractal scaling", 0.0, 1.0, 0.72, step=0.001, format="%.4f", key="pk_dfa")
            pk_spread1 = st.number_input("spread1 — Nonlinear measure", -10.0, 0.0, -5.68, step=0.01, key="pk_spread1")
            pk_ppe     = st.number_input("PPE — Pitch period entropy", 0.0, 1.0, 0.21, step=0.001, format="%.4f", key="pk_ppe")

        if st.button("🔍 Predict Parkinson's", type="primary", use_container_width=True, key="btn_park"):
            features = [pk_fo, pk_fhi, pk_flo, pk_jitter, pk_shimmer,
                        pk_hnr, pk_rpde, pk_dfa, pk_spread1, pk_ppe]
            predict_parkinsons(features)

    # ── TAB 9: LIVER ─────────────────────────────────────────────────────────
    with sub_tabs[8]:
        st.markdown("#### 🫀 Liver Disease Prediction")
        st.caption("Model: `pickle_model_liver.pkl` | Dataset: ILPD (Indian Liver Patient) | 10 features | Imputer + Scaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Assess liver disease risk from <b style="color:#FFFFFF;">10 blood test biomarkers</b>.
📊 <a href="https://www.kaggle.com/datasets/uciml/indian-liver-patient-records"
target="_blank" style="color:#007AFF;">Kaggle — Indian Liver Patient Records (ILPD)</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("liver", "LFT (Liver Function Test) / bilirubin / SGPT / SGOT report")

        c1, c2 = st.columns(2)
        with c1:
            lv_age   = st.number_input("Age", 4, 100, 45, key="lv_age")
            lv_sex   = st.selectbox("Gender", ["Female (0)", "Male (1)"], key="lv_sex")
            lv_tbil  = st.number_input("Total Bilirubin (mg/dL)", 0.1, 75.0, 0.9, step=0.1, key="lv_tbil")
            lv_dbil  = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 20.0, 0.2, step=0.1, key="lv_dbil")
            lv_alkp  = st.number_input("Alkaline Phosphotase (IU/L)", 60, 2110, 180, key="lv_alkp")
        with c2:
            lv_alt   = st.number_input("Alamine Aminotransferase / ALT (IU/L)", 5, 2000, 25, key="lv_alt")
            lv_ast   = st.number_input("Aspartate Aminotransferase / AST (IU/L)", 10, 5000, 30, key="lv_ast")
            lv_tp    = st.number_input("Total Proteins (g/dL)", 2.0, 10.0, 6.8, step=0.1, key="lv_tp")
            lv_alb   = st.number_input("Albumin (g/dL)", 0.5, 6.0, 3.4, step=0.1, key="lv_alb")
            lv_agr   = st.number_input("Albumin/Globulin Ratio", 0.1, 3.0, 1.0, step=0.01, key="lv_agr")

        def _lv_val(s): return int(s.rstrip(")").split("(")[-1])

        if st.button("🔍 Predict Liver Disease", type="primary", use_container_width=True, key="btn_liver"):
            features = [
                lv_age, _lv_val(lv_sex), lv_tbil, lv_dbil, lv_alkp,
                lv_alt, lv_ast, lv_tp, lv_alb, lv_agr,
            ]
            predict_liver(features)

    # ── TAB 10: KIDNEY ───────────────────────────────────────────────────────
    with sub_tabs[9]:
        st.markdown("#### 🫘 Chronic Kidney Disease Prediction")
        st.caption("Model: `pickle_model_kidney.pkl` | Dataset: UCI CKD | 14 features | Imputer + Scaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Predict Chronic Kidney Disease from <b style="color:#FFFFFF;">14 clinical lab measurements</b>.
📊 <a href="https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease"
target="_blank" style="color:#007AFF;">UCI — Chronic Kidney Disease Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("kidney", "kidney function test / CBC / urine report / electrolytes report")

        c1, c2 = st.columns(2)
        with c1:
            kd_age  = st.number_input("Age (years)", 2, 100, 48, key="kd_age")
            kd_bp   = st.number_input("Blood Pressure (mmHg)", 50, 180, 80, key="kd_bp")
            kd_bgr  = st.number_input("Blood Glucose Random (mg/dL)", 70, 500, 120, key="kd_bgr")
            kd_bu   = st.number_input("Blood Urea (mg/dL)", 1, 400, 36, key="kd_bu")
            kd_sc   = st.number_input("Serum Creatinine (mg/dL)", 0.4, 15.0, 1.2, step=0.1, key="kd_sc")
            kd_sod  = st.number_input("Sodium (mEq/L)", 100, 165, 137, key="kd_sod")
            kd_pot  = st.number_input("Potassium (mEq/L)", 2.5, 10.0, 4.6, step=0.1, key="kd_pot")
        with c2:
            kd_hemo = st.number_input("Haemoglobin (g/dL)", 3.0, 20.0, 13.5, step=0.1, key="kd_hemo")
            kd_pcv  = st.number_input("Packed Cell Volume (%)", 9, 55, 41, key="kd_pcv")
            kd_wc   = st.number_input("WBC Count (cells/µL)", 2000, 26000, 8000, key="kd_wc")
            kd_rc   = st.number_input("RBC Count (millions/µL)", 2.0, 8.0, 4.7, step=0.1, key="kd_rc")
            kd_htn  = st.selectbox("Hypertension", ["No (0)", "Yes (1)"], key="kd_htn")
            kd_dm   = st.selectbox("Diabetes Mellitus", ["No (0)", "Yes (1)"], key="kd_dm")
            kd_ane  = st.selectbox("Anaemia", ["No (0)", "Yes (1)"], key="kd_ane")

        def _kd_val(s): return int(s.rstrip(")").split("(")[-1])

        if st.button("🔍 Predict Kidney Disease", type="primary", use_container_width=True, key="btn_kidney"):
            features = [
                kd_age, kd_bp, kd_bgr, kd_bu, kd_sc, kd_sod, kd_pot,
                kd_hemo, kd_pcv, kd_wc, kd_rc,
                _kd_val(kd_htn), _kd_val(kd_dm), _kd_val(kd_ane),
            ]
            predict_kidney(features)

    # ── TAB 11: BREAST CANCER ────────────────────────────────────────────────
    with sub_tabs[10]:
        st.markdown("#### 🎗️ Breast Cancer Prediction")
        st.caption("Model: `pickle_model_breast_cancer.pkl` | Dataset: Wisconsin Breast Cancer | 10 features | StandardScaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Classify tumors as <b style="color:#30D158;">Benign</b> or <b style="color:#FF453A;">Malignant</b> from
<b style="color:#FFFFFF;">10 mean tumor measurements</b> from the Wisconsin Breast Cancer dataset.
📊 <a href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data"
target="_blank" style="color:#007AFF;">Kaggle — Breast Cancer Wisconsin (Diagnostic)</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("breast_cancer", "pathology report / biopsy report / ultrasound fine needle aspiration (FNA) report")

        c1, c2 = st.columns(2)
        with c1:
            bc_radius      = st.number_input("Mean Radius (mm)", 6.0, 30.0, 14.1, step=0.01, key="bc_radius")
            bc_texture     = st.number_input("Mean Texture", 9.0, 40.0, 19.3, step=0.01, key="bc_texture")
            bc_perimeter   = st.number_input("Mean Perimeter (mm)", 40.0, 200.0, 92.0, step=0.1, key="bc_perimeter")
            bc_area        = st.number_input("Mean Area (mm²)", 140.0, 2600.0, 654.9, step=1.0, key="bc_area")
            bc_smoothness  = st.number_input("Mean Smoothness", 0.05, 0.20, 0.096, step=0.001, format="%.4f", key="bc_smooth")
        with c2:
            bc_compactness = st.number_input("Mean Compactness", 0.01, 0.40, 0.104, step=0.001, format="%.4f", key="bc_compact")
            bc_concavity   = st.number_input("Mean Concavity", 0.0, 0.50, 0.089, step=0.001, format="%.4f", key="bc_concav")
            bc_concave_pts = st.number_input("Mean Concave Points", 0.0, 0.20, 0.049, step=0.001, format="%.4f", key="bc_concpts")
            bc_symmetry    = st.number_input("Mean Symmetry", 0.1, 0.4, 0.181, step=0.001, format="%.4f", key="bc_symm")
            bc_fractal     = st.number_input("Mean Fractal Dimension", 0.04, 0.10, 0.062, step=0.001, format="%.4f", key="bc_fractal")

        if st.button("🔍 Predict Breast Cancer", type="primary", use_container_width=True, key="btn_bc"):
            features = [
                bc_radius, bc_texture, bc_perimeter, bc_area, bc_smoothness,
                bc_compactness, bc_concavity, bc_concave_pts, bc_symmetry, bc_fractal,
            ]
            predict_breast_cancer(features)

    # ── TAB 12: ABOUT MODELS ─────────────────────────────────────────────────
    # ── TAB 12: TUBERCULOSIS ─────────────────────────────────────────────────
    with sub_tabs[11]:
        st.markdown("#### 🫁 Tuberculosis Detection from Chest X-Ray")
        st.caption("Model: `pickle_model_tuberculosis.pkl` | Dataset: Kaggle TB X-Ray | Algorithm: PCA + Random Forest")
        st.markdown("""<div style="background:#1C1C1E;border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">Chest X-Ray image</b> (JPG/PNG). The model predicts
<span style="color:#FF453A;font-weight:600;">TUBERCULOSIS</span> or <span style="color:#30D158;font-weight:600;">NORMAL</span>.
</div>""", unsafe_allow_html=True)
        tb_file = st.file_uploader("Upload Chest X-Ray", type=["jpg","jpeg","png"], key="tb_upload")
        if tb_file:
            col_i, col_b = st.columns([1,1])
            with col_i: st.image(tb_file, caption="Uploaded X-Ray", use_container_width=True)
            with col_b:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button("🔍 Analyse X-Ray", type="primary", use_container_width=True, key="btn_tb"):
                    with st.spinner("Analysing…"): predict_tuberculosis(tb_file)

    # ── TAB 13: DIABETIC RETINOPATHY ─────────────────────────────────────────
    with sub_tabs[12]:
        st.markdown("#### 👁️ Diabetic Retinopathy Detection")
        st.caption("Model: `pickle_model_retinopathy.pkl` | Dataset: Kaggle APTOS 2019 | Algorithm: PCA + Random Forest")
        st.markdown("""<div style="background:#1C1C1E;border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">retinal fundus photo</b> (JPG/PNG). The model grades diabetic retinopathy from
<span style="color:#30D158;font-weight:600;">No DR</span> to <span style="color:#FF453A;font-weight:600;">Proliferative DR</span>.
</div>""", unsafe_allow_html=True)
        dr_file = st.file_uploader("Upload Fundus Photo", type=["jpg","jpeg","png"], key="dr_upload")
        if dr_file:
            col_i, col_b = st.columns([1,1])
            with col_i: st.image(dr_file, caption="Fundus Image", use_container_width=True)
            with col_b:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button("🔍 Analyse Retina", type="primary", use_container_width=True, key="btn_dr"):
                    with st.spinner("Analysing…"): predict_retinopathy(dr_file)

    # ── TAB 14: BONE FRACTURE ────────────────────────────────────────────────
    with sub_tabs[13]:
        st.markdown("#### 🩻 Bone Fracture Detection from X-Ray")
        st.caption("Model: `pickle_model_bone_fracture.pkl` | Dataset: Kaggle Fracture Detection | Algorithm: PCA + Random Forest")
        st.markdown("""<div style="background:#1C1C1E;border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">bone X-Ray image</b> (JPG/PNG). The model detects the presence of a
<span style="color:#FF453A;font-weight:600;">Fracture</span> or confirms <span style="color:#30D158;font-weight:600;">No Fracture</span>.
</div>""", unsafe_allow_html=True)
        bf_file = st.file_uploader("Upload Bone X-Ray", type=["jpg","jpeg","png"], key="bf_upload")
        if bf_file:
            col_i, col_b = st.columns([1,1])
            with col_i: st.image(bf_file, caption="Bone X-Ray", use_container_width=True)
            with col_b:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button("🔍 Analyse X-Ray", type="primary", use_container_width=True, key="btn_bf"):
                    with st.spinner("Analysing…"): predict_bone_fracture(bf_file)

    # ── TAB 15: SKIN DISEASE ─────────────────────────────────────────────────
    with sub_tabs[14]:
        st.markdown("#### 🔬 Skin Disease Classification")
        st.caption("Model: `pickle_model_skin_disease.pkl` | Dataset: HAM10000 | Algorithm: PCA + Random Forest")
        st.markdown("""<div style="background:#1C1C1E;border-radius:10px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:#A1A1A6;">
Upload a <b style="color:#FFFFFF;">skin lesion photo</b> (JPG/PNG). Classifies into 7 categories including
<span style="color:#FF453A;font-weight:600;">Melanoma</span>, <span style="color:#FF6B35;font-weight:600;">Basal Cell Carcinoma</span>,
and <span style="color:#30D158;font-weight:600;">benign</span> conditions.
<br><br>⚠️ <b style="color:#FFD60A;">Always consult a dermatologist</b> for any skin concern — this is a screening tool only.
</div>""", unsafe_allow_html=True)
        sk_file = st.file_uploader("Upload Skin Lesion Photo", type=["jpg","jpeg","png"], key="sk_upload")
        if sk_file:
            col_i, col_b = st.columns([1,1])
            with col_i: st.image(sk_file, caption="Skin Lesion", use_container_width=True)
            with col_b:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                if st.button("🔍 Classify Lesion", type="primary", use_container_width=True, key="btn_sk2"):
                    with st.spinner("Analysing…"): predict_skin_disease(sk_file)

    # ── TAB 16: THYROID ──────────────────────────────────────────────────────
    with sub_tabs[15]:
        st.markdown("#### 🦋 Thyroid Disease Prediction")
        st.caption("Model: `thyroid_rf_model.pkl` | Dataset: UCI Thyroid / Kaggle | 12 features | Scaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Classify thyroid function as <b style="color:#30D158;">Normal</b>,
<span style="color:#FF6B35;font-weight:600;">Hypothyroid</span>, or
<span style="color:#FF453A;font-weight:600;">Hyperthyroid</span> from TFT lab values.
📊 <a href="https://www.kaggle.com/datasets/kapoorprakhar/thyroid-disease-dataset"
target="_blank" style="color:#007AFF;">Kaggle — Thyroid Disease Dataset (UCI)</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("thyroid", "Thyroid Function Test (TFT) / TSH / T3 / T4 lab report")

        def _th_bino(s): return int(s.rstrip(")").split("(")[-1])

        c1, c2 = st.columns(2)
        with c1:
            th_age    = st.number_input("Age", 1, 100, 40, key="th_age")
            th_sex    = st.selectbox("Gender", ["Female (0)", "Male (1)"], key="th_sex")
            th_tsh    = st.number_input("TSH (mIU/L)", 0.0, 600.0, 2.5, step=0.1, key="th_tsh")
            th_t3     = st.number_input("T3 (nmol/L)", 0.0, 10.0, 1.8, step=0.1, key="th_t3")
            th_tt4    = st.number_input("Total T4 (nmol/L)", 0.0, 300.0, 100.0, step=1.0, key="th_tt4")
            th_t4u    = st.number_input("T4 Uptake", 0.0, 3.0, 1.0, step=0.01, key="th_t4u")
        with c2:
            th_fti    = st.number_input("Free Thyroxine Index (FTI)", 0.0, 500.0, 100.0, step=1.0, key="th_fti")
            th_onthyr = st.selectbox("On Thyroxine Medication", ["No (0)", "Yes (1)"], key="th_onthyr")
            th_antith = st.selectbox("On Antithyroid Medication", ["No (0)", "Yes (1)"], key="th_antith")
            th_sick   = st.selectbox("Currently Sick", ["No (0)", "Yes (1)"], key="th_sick")
            th_preg   = st.selectbox("Pregnant", ["No (0)", "Yes (1)"], key="th_preg")
            th_surg   = st.selectbox("Thyroid Surgery History", ["No (0)", "Yes (1)"], key="th_surg")

        if st.button("🔍 Predict Thyroid Function", type="primary", use_container_width=True, key="btn_thyroid"):
            features = [
                th_age, _th_bino(th_sex), th_tsh, th_t3, th_tt4, th_t4u, th_fti,
                _th_bino(th_onthyr), _th_bino(th_antith),
                _th_bino(th_sick), _th_bino(th_preg), _th_bino(th_surg),
            ]
            predict_thyroid(features)

    # ── TAB 17: ANAEMIA ──────────────────────────────────────────────────────
    with sub_tabs[16]:
        st.markdown("#### 🩺 Anaemia Detection")
        st.caption("Model: `pickle_model_anaemia.pkl` | Dataset: Kaggle Anaemia Dataset | 5 features | RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Detect anaemia from <b style="color:#FFFFFF;">CBC blood test values</b>: Haemoglobin, MCH, MCHC, MCV.
📊 <a href="https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset"
target="_blank" style="color:#007AFF;">Kaggle — Anaemia Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("anaemia", "CBC (Complete Blood Count) / haemoglobin / blood test report")

        def _an_bino(s): return int(s.rstrip(")").split("(")[-1])

        c1, c2 = st.columns(2)
        with c1:
            an_gender = st.selectbox("Gender", ["Female (0)", "Male (1)"], key="an_gender")
            an_hgb    = st.number_input("Haemoglobin (g/dL)", 1.0, 20.0, 13.5, step=0.1, key="an_hgb")
            an_mch    = st.number_input("MCH (pg)", 10.0, 50.0, 28.0, step=0.1, key="an_mch")
        with c2:
            an_mchc   = st.number_input("MCHC (g/dL)", 20.0, 40.0, 33.0, step=0.1, key="an_mchc")
            an_mcv    = st.number_input("MCV (fL)", 50.0, 130.0, 88.0, step=0.5, key="an_mcv")

        if st.button("🔍 Detect Anaemia", type="primary", use_container_width=True, key="btn_anaemia"):
            features = [_an_bino(an_gender), an_hgb, an_mch, an_mchc, an_mcv]
            predict_anaemia(features)

    # ── TAB 18: PCOS ─────────────────────────────────────────────────────────
    with sub_tabs[17]:
        st.markdown("#### 🔵 PCOS Detection")
        st.caption("Model: `pcos_rf_model.pkl` | Dataset: Kaggle PCOS Dataset | 15 features | Scaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Detect <b style="color:#FFFFFF;">Polycystic Ovary Syndrome (PCOS)</b> from hormonal levels, ultrasound findings, and symptoms.
📊 <a href="https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos"
target="_blank" style="color:#007AFF;">Kaggle — Polycystic Ovary Syndrome (PCOS) Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("pcos", "hormonal profile / ultrasound report / gynaecology report")

        def _pc_bino(s): return int(s.rstrip(")").split("(")[-1])

        c1, c2 = st.columns(2)
        with c1:
            pc_age    = st.number_input("Age", 15, 55, 27, key="pc_age")
            pc_bmi    = st.number_input("BMI (kg/m²)", 10.0, 50.0, 24.0, step=0.1, key="pc_bmi")
            pc_fsh    = st.number_input("FSH (mIU/mL)", 0.0, 50.0, 5.0, step=0.1, key="pc_fsh")
            pc_lh     = st.number_input("LH (mIU/mL)", 0.0, 100.0, 6.0, step=0.1, key="pc_lh")
            pc_amh    = st.number_input("AMH (ng/mL)", 0.0, 20.0, 3.0, step=0.1, key="pc_amh")
            pc_cycle  = st.number_input("Cycle Length (days)", 15, 90, 29, key="pc_cycle")
            pc_fol_r  = st.number_input("Follicle Count — Right Ovary", 0, 40, 5, key="pc_fol_r")
            pc_fol_l  = st.number_input("Follicle Count — Left Ovary", 0, 40, 5, key="pc_fol_l")
        with c2:
            pc_endo   = st.number_input("Endometrium Thickness (mm)", 0.0, 20.0, 8.0, step=0.1, key="pc_endo")
            pc_irreg  = st.selectbox("Menstrual Cycle", ["Regular (0)", "Irregular (1)"], key="pc_irreg")
            pc_skin   = st.selectbox("Skin Darkening", ["No (0)", "Yes (1)"], key="pc_skin")
            pc_hair   = st.selectbox("Excessive Hair Growth", ["No (0)", "Yes (1)"], key="pc_hair")
            pc_pimple = st.selectbox("Pimples / Acne", ["No (0)", "Yes (1)"], key="pc_pimple")
            pc_hairloss=st.selectbox("Hair Loss", ["No (0)", "Yes (1)"], key="pc_hairloss")
            pc_wtgain = st.selectbox("Weight Gain", ["No (0)", "Yes (1)"], key="pc_wtgain")

        if st.button("🔍 Detect PCOS", type="primary", use_container_width=True, key="btn_pcos"):
            features = [
                pc_age, pc_bmi, pc_fsh, pc_lh, pc_amh, pc_cycle,
                pc_fol_r, pc_fol_l, pc_endo,
                _pc_bino(pc_irreg), _pc_bino(pc_skin), _pc_bino(pc_hair),
                _pc_bino(pc_pimple), _pc_bino(pc_hairloss), _pc_bino(pc_wtgain),
            ]
            predict_pcos(features)

    # ── TAB 19: ABOUT MODELS ─────────────────────────────────────────────────
    # ── TAB 19: HEART FAILURE ────────────────────────────────────────────────
    with sub_tabs[18]:
        st.markdown("#### 💔 Heart Failure Prediction")
        st.caption("Model: `pickle_model_heart_failure.pkl` | Dataset: Kaggle Heart Failure Clinical | 12 features | GBM")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Predict heart failure mortality risk from <b style="color:#FFFFFF;">12 clinical features</b>.
📊 <a href="https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data" target="_blank" style="color:#007AFF;">Kaggle — Heart Failure Clinical Data</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("heart_failure", "cardiology / echo / ejection fraction / serum creatinine report")
        def _hf_yn(v): return 1 if "Yes" in v else 0
        c1, c2 = st.columns(2)
        with c1:
            hf_age      = st.number_input("Age", 1, 100, 60, key="hf_age")
            hf_sex      = st.selectbox("Sex", ["Female (0)", "Male (1)"], key="hf_sex")
            hf_ef       = st.number_input("Ejection Fraction (%)", 10, 80, 38, key="hf_ef")
            hf_cpk      = st.number_input("Creatinine Phosphokinase CPK (IU/L)", 0, 8000, 250, key="hf_cpk")
            hf_platelets= st.number_input("Platelets (×10³/mL)", 0.0, 900000.0, 250000.0, step=1000.0, key="hf_platelets")
            hf_sc       = st.number_input("Serum Creatinine (mg/dL)", 0.0, 10.0, 1.1, step=0.01, key="hf_sc")
        with c2:
            hf_sodium   = st.number_input("Serum Sodium (mEq/L)", 100, 150, 137, key="hf_sodium")
            hf_time     = st.number_input("Follow-up Period (days)", 1, 300, 100, key="hf_time")
            hf_anaemia  = st.selectbox("Anaemia?", ["No (0)", "Yes (1)"], key="hf_anaemia")
            hf_diabetes = st.selectbox("Diabetes?", ["No (0)", "Yes (1)"], key="hf_diabetes")
            hf_hbp      = st.selectbox("High Blood Pressure?", ["No (0)", "Yes (1)"], key="hf_hbp")
            hf_smoking  = st.selectbox("Smoking?", ["No (0)", "Yes (1)"], key="hf_smoking")
        if st.button("🔬 Predict Heart Failure Risk", type="primary", use_container_width=True, key="btn_hf"):
            model, err = _load_pickle(ML_MODEL_PATHS["heart_failure"])
            if model:
                feats = [[hf_age, _hf_yn(hf_anaemia), hf_cpk, _hf_yn(hf_diabetes), hf_ef,
                          _hf_yn(hf_hbp), hf_platelets, hf_sc, hf_sodium,
                          1 if "Male" in hf_sex else 0, _hf_yn(hf_smoking), hf_time]]
                pred = model.predict(feats)[0]
                if pred == 1:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#FF453A;'>🔴 High Mortality Risk — Urgent Cardiology Review Needed</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#30D158;'>🟢 Lower Risk — Continue Monitoring</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Model not found: {err}")

    # ── TAB 20: CERVICAL CANCER ──────────────────────────────────────────────
    with sub_tabs[19]:
        st.markdown("#### 🔴 Cervical Cancer Risk Prediction")
        st.caption("Model: `pickle_model_cervical.pkl` | Dataset: UCI Cervical Cancer | 15 features | GBM")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Predict biopsy-positive cervical cancer risk from <b style="color:#FFFFFF;">reproductive history & lifestyle risk factors</b>.
📊 <a href="https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29" target="_blank" style="color:#007AFF;">UCI — Cervical Cancer Risk Factors</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("cervical", "gynaecology report / pap smear / colposcopy report")
        def _cc_yn(v): return 1 if "Yes" in v else 0
        c1, c2 = st.columns(2)
        with c1:
            cc_age         = st.number_input("Age", 10, 90, 30, key="cc_age")
            cc_partners    = st.number_input("Number of Sexual Partners", 0, 30, 2, key="cc_partners")
            cc_first_sex   = st.number_input("Age at First Intercourse", 10, 30, 18, key="cc_first_sex")
            cc_pregnancies = st.number_input("Number of Pregnancies", 0, 15, 1, key="cc_pregnancies")
            cc_smokes      = st.selectbox("Smokes?", ["No (0)", "Yes (1)"], key="cc_smokes")
            cc_smokes_yrs  = st.number_input("Smoking Years", 0.0, 50.0, 0.0, step=0.5, key="cc_smokes_yrs")
            cc_hc          = st.selectbox("Hormonal Contraceptives?", ["No (0)", "Yes (1)"], key="cc_hc")
            cc_hc_yrs      = st.number_input("HC Years of Use", 0.0, 30.0, 0.0, step=0.5, key="cc_hc_yrs")
        with c2:
            cc_iud         = st.selectbox("IUD?", ["No (0)", "Yes (1)"], key="cc_iud")
            cc_iud_yrs     = st.number_input("IUD Years of Use", 0.0, 20.0, 0.0, step=0.5, key="cc_iud_yrs")
            cc_stds        = st.selectbox("STDs History?", ["No (0)", "Yes (1)"], key="cc_stds")
            cc_stds_n      = st.number_input("Number of STD Diagnoses", 0, 10, 0, key="cc_stds_n")
            cc_dx_hpv      = st.selectbox("HPV Diagnosis?", ["No (0)", "Yes (1)"], key="cc_dx_hpv")
            cc_dx_cin      = st.selectbox("CIN Diagnosis?", ["No (0)", "Yes (1)"], key="cc_dx_cin")
            cc_dx_cancer   = st.selectbox("Prior Cancer?", ["No (0)", "Yes (1)"], key="cc_dx_cancer")
        if st.button("🔬 Predict Cervical Cancer Risk", type="primary", use_container_width=True, key="btn_cervical"):
            model, err = _load_pickle(ML_MODEL_PATHS["cervical"])
            if model:
                feats = [[cc_age, cc_partners, cc_first_sex, cc_pregnancies,
                          _cc_yn(cc_smokes), cc_smokes_yrs, _cc_yn(cc_hc), cc_hc_yrs,
                          _cc_yn(cc_iud), cc_iud_yrs, _cc_yn(cc_stds), cc_stds_n,
                          _cc_yn(cc_dx_hpv), _cc_yn(cc_dx_cin), _cc_yn(cc_dx_cancer)]]
                pred = model.predict(feats)[0]
                if pred == 1:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#FF453A;'>🔴 High Risk — Biopsy / Colposcopy Recommended Urgently</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#30D158;'>🟢 Low Risk — Routine Screening Advised</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Model not found: {err}")

    # ── TAB 21: HEPATITIS ────────────────────────────────────────────────────
    with sub_tabs[20]:
        st.markdown("#### 🟡 Hepatitis / Liver Fibrosis Prediction")
        st.caption("Model: `pickle_model_hepatitis.pkl` | Dataset: HCV Kaggle | 12 blood enzyme features | Scaler + RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Classify liver condition as <b style="color:#30D158;">Healthy</b> / <b style="color:#FFD60A;">Hepatitis</b> / <b style="color:#FF9F0A;">Fibrosis</b> / <b style="color:#FF453A;">Cirrhosis</b>.
📊 <a href="https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset" target="_blank" style="color:#007AFF;">Kaggle — Hepatitis C Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("hepatitis", "LFT / bilirubin / hepatitis C / liver enzyme blood test")
        c1, c2 = st.columns(2)
        with c1:
            hep_age  = st.number_input("Age", 1, 100, 40, key="hep_age")
            hep_sex  = st.selectbox("Sex", ["Female (0)", "Male (1)"], key="hep_sex")
            hep_alb  = st.number_input("Albumin ALB (g/dL)", 0.0, 10.0, 4.0, step=0.01, key="hep_alb")
            hep_alp  = st.number_input("ALP (IU/L)", 0.0, 500.0, 60.0, step=0.1, key="hep_alp")
            hep_alt  = st.number_input("ALT / SGPT (IU/L)", 0.0, 500.0, 25.0, step=0.1, key="hep_alt")
            hep_ast  = st.number_input("AST / SGOT (IU/L)", 0.0, 500.0, 22.0, step=0.1, key="hep_ast")
        with c2:
            hep_bil  = st.number_input("Bilirubin BIL (mg/dL)", 0.0, 50.0, 0.7, step=0.01, key="hep_bil")
            hep_che  = st.number_input("Cholinesterase CHE (kU/L)", 0.0, 20.0, 8.0, step=0.01, key="hep_che")
            hep_chol = st.number_input("Cholesterol (mmol/L)", 0.0, 15.0, 5.0, step=0.01, key="hep_chol")
            hep_crea = st.number_input("Creatinine (µmol/L)", 0.0, 1500.0, 80.0, step=0.1, key="hep_crea")
            hep_ggt  = st.number_input("GGT (IU/L)", 0.0, 500.0, 25.0, step=0.1, key="hep_ggt")
            hep_prot = st.number_input("Total Protein (g/dL)", 0.0, 15.0, 7.0, step=0.01, key="hep_prot")
        if st.button("🔬 Predict Hepatitis Status", type="primary", use_container_width=True, key="btn_hepatitis"):
            model, err = _load_pickle(ML_MODEL_PATHS["hepatitis"])
            if model:
                feats = [[hep_age, 1 if "Male" in hep_sex else 0,
                          hep_alb, hep_alp, hep_alt, hep_ast,
                          hep_bil, hep_che, hep_chol, hep_crea, hep_ggt, hep_prot]]
                pred = model.predict(feats)[0]
                labels = {0:("🟢 Blood Donor — Healthy","#30D158"),
                          1:("🟡 Hepatitis Detected","#FFD60A"),
                          2:("🟠 Liver Fibrosis","#FF9F0A"),
                          3:("🔴 Cirrhosis","#FF453A")}
                lbl, col = labels.get(int(pred), ("Unknown","#888888"))
                st.markdown(f"<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:{col};'>{lbl}</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Model not found: {err}")

    # ── TAB 22: SEPSIS ───────────────────────────────────────────────────────
    with sub_tabs[21]:
        st.markdown("#### 💊 Sepsis Risk Prediction")
        st.caption("Model: `pickle_model_sepsis.pkl` | Dataset: Kaggle Sepsis | 14 ICU features | Balanced RF")
        st.markdown("""
<div style="background:#1C1C1E;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:13px;color:#A1A1A6;line-height:1.6;">
Predict sepsis from <b style="color:#FFFFFF;">ICU vitals + blood lab values</b>.
📊 <a href="https://www.kaggle.com/datasets/salikhussaini49/sepsis-prediction" target="_blank" style="color:#007AFF;">Kaggle — Sepsis Prediction Dataset</a>
</div>
""", unsafe_allow_html=True)
        _render_report_scanner("sepsis", "ICU chart / blood gas / CBC / metabolic panel report")
        c1, c2 = st.columns(2)
        with c1:
            sp_age    = st.number_input("Age (years)", 1.0, 110.0, 60.0, step=0.5, key="sp_age")
            sp_hr     = st.number_input("Heart Rate (bpm)", 20.0, 200.0, 80.0, step=0.5, key="sp_hr")
            sp_o2     = st.number_input("O₂ Saturation (%)", 50.0, 100.0, 97.0, step=0.1, key="sp_o2")
            sp_temp   = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0, step=0.1, key="sp_temp")
            sp_sbp    = st.number_input("Systolic BP (mmHg)", 40.0, 250.0, 120.0, step=0.5, key="sp_sbp")
            sp_map    = st.number_input("MAP (mmHg)", 20.0, 180.0, 85.0, step=0.5, key="sp_map")
            sp_resp   = st.number_input("Respiration Rate (br/min)", 5.0, 60.0, 16.0, step=0.5, key="sp_resp")
        with c2:
            sp_wbc    = st.number_input("WBC (×10⁹/L)", 0.0, 50.0, 8.0, step=0.1, key="sp_wbc")
            sp_hgb    = st.number_input("Haemoglobin (g/dL)", 0.0, 25.0, 13.0, step=0.1, key="sp_hgb")
            sp_bun    = st.number_input("BUN (mg/dL)", 0.0, 200.0, 15.0, step=0.5, key="sp_bun")
            sp_creat  = st.number_input("Creatinine (mg/dL)", 0.0, 15.0, 1.0, step=0.01, key="sp_creat")
            sp_gluc   = st.number_input("Glucose (mg/dL)", 30.0, 600.0, 110.0, step=0.5, key="sp_gluc")
            sp_lact   = st.number_input("Lactate (mmol/L)", 0.0, 20.0, 1.5, step=0.1, key="sp_lact")
            sp_iculos = st.number_input("ICU Stay (hours)", 0.0, 336.0, 24.0, step=0.5, key="sp_iculos")
        if st.button("🔬 Predict Sepsis Risk", type="primary", use_container_width=True, key="btn_sepsis"):
            model, err = _load_pickle(ML_MODEL_PATHS["sepsis"])
            if model:
                feats = [[sp_hr, sp_o2, sp_temp, sp_sbp, sp_map, sp_resp,
                          sp_wbc, sp_hgb, sp_bun, sp_creat, sp_gluc, sp_lact,
                          sp_age, sp_iculos]]
                pred = model.predict(feats)[0]
                if pred == 1:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#FF453A;'>🔴 Sepsis Risk Detected — Immediate Medical Attention Required</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='background:#1C1C1E;border-radius:12px;padding:16px;text-align:center;font-size:18px;font-weight:700;color:#30D158;'>🟢 Low Sepsis Risk</div>", unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Model not found: {err}")

    # ── TAB 23: ABOUT MODELS ─────────────────────────────────────────────────
    with sub_tabs[22]:
        st.markdown("""
<div style="font-size:13px;color:#A1A1A6;line-height:1.8;">

<b style="color:#FFFFFF;">📦 ML Models (Scikit-Learn / Pickle)</b>

| Model File | Disease | Algorithm | Dataset |
|---|---|---|---|
| `pickle_model_diabetes.pkl` | Diabetes | Random Forest | Pima Indians Diabetes |
| `pickle_model_heart.pkl` | Heart Disease | Random Forest | UCI Heart Disease |
| `pickle_model_disease.pkl` | 41 Diseases | Random Forest | Symptom–Disease (132 features) |
| `pickle_model_breast_cancer.pkl` | Breast Cancer | Scaler + RF | Wisconsin Breast Cancer |
| `pickle_model_kidney.pkl` | Kidney Disease | Imputer + Scaler + RF | UCI Chronic Kidney |
| `pickle_model_liver.pkl` | Liver Disease | Imputer + Scaler + RF | ILPD Indian Liver |
| `pickle_model_parkinsons.pkl` | Parkinson's | Scaler + SVC | UCI Parkinson's Voice |
| `pickle_model_stroke.pkl` | Stroke | Scaler + RF | Kaggle Healthcare Stroke |
| `pickle_model_osteoporosis.pkl` | Osteoporosis | Scaler + RF | Kaggle Lifestyle Factors |
| `pickle_model_pneumonia.pkl` | Pneumonia | PCA + RF | Kaggle Chest X-Ray |
| `pickle_model_malaria.pkl` | Malaria | PCA + RF | Kaggle Cell Images |
| `pickle_model_tuberculosis.pkl` | Tuberculosis | PCA + RF | Kaggle TB X-Ray |
| `pickle_model_retinopathy.pkl` | Diabetic Retinopathy | PCA + RF | Kaggle APTOS 2019 |
| `pickle_model_bone_fracture.pkl` | Bone Fracture | PCA + RF | Kaggle Fracture X-Ray |
| `pickle_model_skin_disease.pkl` | Skin Disease (7 classes) | PCA + RF | HAM10000 |
| `thyroid_rf_model.pkl` | Thyroid (Normal/Hypo/Hyper) | Scaler + RF | UCI Thyroid / Kaggle |
| `pickle_model_anaemia.pkl` | Anaemia | Random Forest | Kaggle Anaemia Dataset |
| `pcos_rf_model.pkl` | PCOS | Scaler + RF | Kaggle PCOS Dataset |
| `pickle_model_heart_failure.pkl` | Heart Failure | GBM + Scaler | Kaggle Heart Failure Clinical |
| `pickle_model_cervical.pkl` | Cervical Cancer Risk | GBM + Imputer | UCI Cervical Cancer |
| `pickle_model_hepatitis.pkl` | Hepatitis / HCV | Scaler + RF | Kaggle HCV Dataset |
| `pickle_model_sepsis.pkl` | Sepsis | Balanced RF | Kaggle Sepsis ICU |

<br>

<b style="color:#FFFFFF;">⚙️ Setup Instructions</b>

</div>
""", unsafe_allow_html=True)

        st.markdown("""
1. Place the `.pkl` model files alongside `app.py`
2. Install: `pip install scikit-learn numpy pandas pillow`
3. Run: `streamlit run app.py`

⚕️ **Disclaimer:** All predictions are for educational and screening purposes only. Not a substitute for professional medical advice.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main Streamlit Application
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="RxPlain", page_icon="💊", layout="centered", initial_sidebar_state="collapsed")

    # Pure Apple WWDC Dark Theme CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp { 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; 
        background: #000000 !important; 
    }
    p, label, h1, h2, h3, h4, h5, h6, li, span, div { color: #FFFFFF !important; }
    .block-container { max-width: 650px !important; padding: 24px 16px !important; }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Buttons - Clean Apple Style */
    .stButton > button {
        border-radius: 12px !important; 
        font-weight: 600 !important;
        padding: 10px 16px !important; 
        font-size: 14px !important;
        transition: all 0.2s ease !important; 
        background: #007AFF !important; 
        border: none !important; 
        color: #FFFFFF !important;
        height: 40px !important;
    }
    .stButton > button:hover { background: #0051D5 !important; }
    .stButton > button[kind="secondary"] { 
        background: #2C2C2E !important; 
        border: 1px solid #3A3A3C !important;
        color: #FFFFFF !important;
    }
    .stButton > button[kind="secondary"]:hover { background: #3A3A3C !important; }
    
    /* File Uploader */
    [data-testid="stFileUploader"] section {
        border: 1px solid #333333 !important; 
        border-radius: 16px !important;
        background: #111111 !important; 
        padding: 32px 20px !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #FFFFFF !important; }
    [data-testid="stFileUploadDropzone"] small { color: #A1A1A6 !important; }
    [data-testid="stFileUploader"] label { display: none; }
    [data-testid="stUploadedFile"] { background-color: #1C1C1E !important; border-radius: 12px !important; }
    [data-testid="stUploadedFile"] div, [data-testid="stUploadedFile"] span { color: #FFFFFF !important; }
    
    /* Text Input */
    .stTextInput input {
        border-radius: 12px !important; 
        border: 1px solid #333333 !important;
        background-color: #1C1C1E !important; 
        color: #FFFFFF !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
    }
    .stTextInput label { font-weight: 600 !important; margin-bottom: 6px !important; color: #A1A1A6 !important; }
    
    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px !important; 
        border: 1px solid #333333 !important;
        background-color: #111111 !important; 
        color: #FFFFFF !important;
        padding: 12px !important;
        font-size: 13px !important;
    }
    .stTextArea label { font-weight: 600 !important; margin-bottom: 8px !important; color: #A1A1A6 !important; }
    
    /* Radio & Selectbox */
    .stRadio label, .stSelectbox label { font-weight: 600 !important; color: #A1A1A6 !important; }
    .stRadio > div { background: transparent !important; }
    
    /* Spinner */
    .stSpinner > div { background: transparent !important; }
    
    /* Expander */
    [data-testid="stExpander"] { border: 1px solid #333333 !important; border-radius: 12px !important; background: #111111 !important; }
    [data-testid="stExpanderDetails"] { background: #0a0a0a !important; }
    
    /* Markdown */
    .stMarkdown { color: #FFFFFF !important; }
    
    /* Tabs — Dark Apple style */
    .stTabs [data-baseweb="tab-list"] {
        background: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 14px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #A1A1A6 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 8px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        background: #007AFF !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 16px !important; }
    
    /* Number Input */
    .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #333333 !important;
        background-color: #1C1C1E !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
    }
    .stNumberInput label { color: #A1A1A6 !important; font-weight: 600 !important; }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background-color: #1C1C1E !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
    }
    .stMultiSelect label { color: #A1A1A6 !important; font-weight: 600 !important; }
    
    @media (max-width: 600px) {
        .block-container { padding: 16px 12px !important; }
        .stButton > button { font-size: 13px !important; padding: 8px 12px !important; height: 36px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize family profiles if not exists
    if "family_profiles" not in st.session_state:
        st.session_state.family_profiles = {"Self": {
            "age": 25, "gender": "Other", "blood_group": "O+",
            "medical_conditions": [], "allergies": [], "medications": [], "health_notes": ""
        }}
    if "current_profile" not in st.session_state:
        st.session_state.current_profile = "Self"

    import streamlit.components.v1 as components
    components.html("""
    <script>
    const doc = window.parent.document;
    
    // 1. Spotlight Effect
    let spotlight = doc.getElementById('global-spotlight');
    if (!spotlight) {
        spotlight = doc.createElement('div');
        spotlight.id = 'global-spotlight';
        spotlight.style.cssText = 'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 9999; background: transparent; transition: opacity 0.3s; opacity: 0; mix-blend-mode: screen;';
        doc.body.appendChild(spotlight);
        
        doc.addEventListener('mousemove', (e) => {
            spotlight.style.opacity = '1';
            spotlight.style.background = `radial-gradient(600px circle at ${e.clientX}px ${e.clientY}px, rgba(150, 200, 255, 0.08), transparent 40%)`;
        });
        doc.addEventListener('mouseleave', () => { spotlight.style.opacity = '0'; });
    }

    // 2. Add SVG Icons to Tabs
    const iconMap = {
        'Prescription Analysis': '<svg style="width:16px;height:16px;margin-right:6px;vertical-align:middle;stroke:currentColor;stroke-width:2;fill:none" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>',
        'Disease Predictor': '<svg style="width:16px;height:16px;margin-right:6px;vertical-align:middle;stroke:currentColor;stroke-width:2;fill:none" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
        'Health Assistant': '<svg style="width:16px;height:16px;margin-right:6px;vertical-align:middle;stroke:currentColor;stroke-width:2;fill:none" viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
        'Health Tools': '<svg style="width:16px;height:16px;margin-right:6px;vertical-align:middle;stroke:currentColor;stroke-width:2;fill:none" viewBox="0 0 24 24"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>'
    };
    const genericIcon = '<svg style="width:14px;height:14px;margin-right:6px;vertical-align:middle;stroke:currentColor;stroke-width:2;fill:none" viewBox="0 0 24 24"><circle cx="12" cy="12" r="6"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/></svg>';
    
    function updateTabs() {
        const tabs = doc.querySelectorAll('button[data-baseweb="tab"] p');
        tabs.forEach(tab => {
            if (!tab.innerHTML.includes('<svg')) {
                const name = tab.innerText.trim();
                const icon = iconMap[name] || genericIcon;
                tab.innerHTML = icon + '<span style="vertical-align:middle;">' + name + '</span>';
            }
        });
    }
    updateTabs();
    
    // Use MutationObserver so icons persist even when Streamlit re-renders React components
    const observer = new MutationObserver(() => updateTabs());
    observer.observe(doc.body, { childList: true, subtree: true });
    </script>
    """, height=0)

    st.markdown("""
<div style="position:relative;text-align:center;margin-bottom:24px;">
<div style="display:inline-flex;align-items:center;gap:8px;background:#111111;border-radius:40px;padding:6px 16px;margin-bottom:16px;border:1px solid #333333;">
<span style="font-weight:600;color:#FFFFFF;font-size:14px;letter-spacing:-0.01em;">RxPlain Intelligence</span>
<span style="font-size:10px;background:#007AFF;color:#FFFFFF !important;padding:2px 8px;border-radius:10px;font-weight:700;">PRO</span>
</div>
<h1 style="font-size:32px;font-weight:700;color:#FFFFFF;margin:0 0 12px;letter-spacing:-0.03em;line-height:1.2;">Healthcare AI Suite</h1>
<p style="color:#8892a6;font-size:15px;margin:0;">Prescription analysis · ML disease prediction · AI health assistant.</p>
</div>
""", unsafe_allow_html=True)

    # ── TOP-LEVEL TABS ──────────────────────────────────────────────────────
    tab_rx, tab_ml, tab_chat, tab_tools, tab_trend = st.tabs(["Prescription Analysis", "Disease Predictor", "Health Assistant", "Health Tools", "Health Trends"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PRESCRIPTION ANALYSIS  (original RxPlain flow)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_rx:
        if "view" not in st.session_state:
            st.session_state.update(view="upload", ocr_text="", ocr_source="", result=None, selected_language="English", prev_language="English")

        # Language selector at top with all Saravam-supported languages
        col_empty, col_lang_select = st.columns([4, 1])
        with col_lang_select:
            selected_lang = st.selectbox(
                "Language",
                ["English", "हिंदी", "Tamil", "Telugu", "Bengali", "Kannada", "Marathi"],
                index=0,
                label_visibility="collapsed",
                key="selected_lang"
            )
    
        # Display selected language info
        lang_display = {
            "English": "🇬🇧 English",
            "हिंदी": "🇮🇳 हिंदी (Hindi)",
            "Tamil": "🇮🇳 தமிழ் (Tamil)",
            "Telugu": "🇮🇳 తెలుగు (Telugu)",
            "Bengali": "🇮🇳 বाংלা (Bengali)",
            "Kannada": "🇮🇳 ಕನ್ನಡ (Kannada)",
            "Marathi": "🇮🇳 मराठी (Marathi)"
        }
        st.caption(f"Content Language: {lang_display.get(selected_lang, selected_lang)}")
    
        # Detect language change and re-run analysis if language was changed
        if selected_lang != st.session_state.get("prev_language", "English"):
            st.session_state.prev_language = selected_lang
            # If we have OCR text and results, re-analyze with new language
            if st.session_state.get("view") == "results" and st.session_state.get("ocr_text"):
                with st.spinner(f"Re-analyzing in {lang_display.get(selected_lang, selected_lang)}..."):
                    try:
                        new_result = execute_analysis_pipeline(st.session_state.ocr_text, language=selected_lang)
                        st.session_state.result = new_result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Language error: {str(e)[:80]}")
            else:
                st.session_state.selected_language = selected_lang

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # VIEW: RESULTS
        if st.session_state.view == "results" and st.session_state.result is not None:
            st.markdown(f"""
    <div style="display:flex;gap:8px;margin-bottom:24px;justify-content:center;">
    {_step_badge(1,"Upload")}
    {_step_badge(2,"Verify")}
    {_step_badge(3,"Analysis",True)}
    </div>
    """, unsafe_allow_html=True)
        
            c_left, c_right = st.columns(2)
            with c_left:
                if st.button("Scan New Prescription", use_container_width=True):
                    st.session_state.update(view="upload", result=None, ocr_text="", prescription_auto_saved=False)
                    st.rerun()
            with c_right:
                if st.button("Refine Transcript", use_container_width=True):
                    st.session_state.view = "review"
                    st.rerun()

            # Export Options Row (Calendar, PDF, WhatsApp, ABHA)
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 0.9])
        
            with exp_col1:
                if st.button("📅 Calendar", use_container_width=True, help="Download .ics"):
                    try:
                        ics_bytes = generate_ics_calendar(st.session_state.result)
                        if ics_bytes:
                            st.download_button(
                                label="📅 Download",
                                data=ics_bytes,
                                file_name=f"RxPlain_{datetime.now().strftime('%Y%m%d')}.ics",
                                mime="text/calendar",
                                use_container_width=True
                            )
                        else:
                            st.info("Calendar export not available")
                    except Exception as e:
                        st.warning(f"Calendar Error: {str(e)[:100]}")
        
            with exp_col2:
                if st.button("📄 PDF", use_container_width=True, help="Download PDF"):
                    try:
                        pdf_bytes = generate_pdf_summary(st.session_state.result, st.session_state.ocr_text)
                        st.download_button(
                            label="📄 Download",
                            data=pdf_bytes,
                            file_name=f"RxPlain_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning(f"PDF Error: {str(e)[:100]}")
        
            with exp_col3:
                whatsapp_link = generate_whatsapp_share_link(st.session_state.result, st.session_state.ocr_text)
                st.markdown(f"""
    <a href="{whatsapp_link}" target="_blank" style="text-decoration:none;">
    <div style="background:#25D366;color:white;border:none;padding:10px;border-radius:12px;font-weight:600;cursor:pointer;text-align:center;height:40px;display:flex;align-items:center;justify-content:center;transition:all 0.2s;font-size:13px;">{SVG_ICONS['whatsapp']} Share</div>
    </a>
    """, unsafe_allow_html=True)

            if st.session_state.ocr_text:
                with st.expander("System Raw Text Extraction Layer"):
                    st.code(st.session_state.ocr_text, language=None)

            # Auto-save prescription to lab reports (only once per analysis)
            if "prescription_auto_saved" not in st.session_state:
                st.session_state.prescription_auto_saved = False
            
            if not st.session_state.prescription_auto_saved and st.session_state.result is not None:
                auto_save_prescription_as_report(st.session_state.result, st.session_state.ocr_text)
                st.session_state.prescription_auto_saved = True
                st.success("💾 Prescription automatically saved to Lab Reports!", icon="✅")
            
            render_results(st.session_state.result, st.session_state.ocr_text)
        
            # NEW: Follow-up Chat Section
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown("""
    <div style="background:#111111;border:1px solid #333333;border-radius:16px;padding:20px;margin-top:24px;">
    <div style="font-weight:600;color:#FFFFFF;margin-bottom:12px;font-size:15px;">❓ Ask About This Prescription</div>
    <p style="font-size:13px;color:#A1A1A6;margin-bottom:12px;">Have questions? Ask anything about your medications, side effects, or instructions.</p>
    </div>
    """, unsafe_allow_html=True)
        
            user_question = st.text_input("Your question:", placeholder="E.g., Can I take this with food? What if I miss a dose?")
            if user_question:
                if st.button("Get Answer", use_container_width=True, type="primary"):
                    with st.spinner("Thinking..."):
                        answer = chat_followup(user_question, st.session_state.ocr_text, st.session_state.result)
                        st.markdown(f"""
    <div style="background:#1C1C1E;border:1px solid #333333;border-radius:12px;padding:16px;margin-top:12px;">
    <div style="font-size:14px;color:#E5E5EA;line-height:1.6;">{answer}</div>
    </div>
    """, unsafe_allow_html=True)
        
            return

        # VIEW: REVIEW
        if st.session_state.view == "review":
            st.markdown(f"""
    <div style="display:flex;gap:8px;margin-bottom:24px;justify-content:center;">
    {_step_badge(1,"Upload")}
    {_step_badge(2,"Verify",True)}
    {_step_badge(3,"Analysis")}
    </div>
    """, unsafe_allow_html=True)

            # NEW: Show flagged uncertain words
            _, flagged_words = flag_uncertain_ocr_words(st.session_state.ocr_text)
            if flagged_words:
                words_html = "".join(
                    f'<span style="background:#FFD60A;color:#000000;padding:4px 10px;border-radius:16px;font-size:12px;font-weight:600;margin-right:4px;">{word}</span>'
                    for word in flagged_words[:8]
                )
                st.markdown(f"""
    <div style="background:#1C1C1E;border:1px solid #FFD60A;border-radius:12px;padding:12px 14px;margin-bottom:16px;">
    <div style="font-weight:600;color:#FFD60A;margin-bottom:8px;font-size:13px;">⚠️ Please Verify These Words:</div>
    <div style="display:flex;flex-wrap:nowrap;gap:8px;overflow-x:auto;">{words_html}</div>
    </div>
    """, unsafe_allow_html=True)

            ocr_corrected = st.text_area(
                "Confirm or make corrections to the interpreted text below:",
                value=st.session_state.ocr_text,
                height=260,
            )

            c_back, c_go = st.columns([1, 2])
            with c_back:
                if st.button("Back", use_container_width=True):
                    st.session_state.update(view="upload", ocr_text="")
                    st.rerun()
            with c_go:
                if st.button("Execute Deep Clinical Analysis", type="primary", use_container_width=True):
                    slot = st.empty()
                    with slot: _loading_card(SVG_ICONS["cpu"], "Processing context via Dual-Engine AI...", "Running Sarvam Primary -> Groq Fallback Protocol")
                    try:
                        res = execute_analysis_pipeline(ocr_corrected.strip(), language=st.session_state.selected_language)
                        st.session_state.update(ocr_text=ocr_corrected.strip(), result=res, view="results")
                        st.rerun()
                    except Exception as e:
                        slot.empty()
                        st.error(f"Execution Error: {e}")
            return

        # VIEW: UPLOAD
        st.markdown(f"""
    <div style="display:flex;gap:8px;margin-bottom:24px;justify-content:center;">
    {_step_badge(1,"Upload",True)}
    {_step_badge(2,"Verify")}
    {_step_badge(3,"Analysis")}
    </div>
    """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📄 Upload Prescription", type=["jpg", "jpeg", "png", "pdf"])

        st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin:20px 0;">
    <div style="flex:1;height:1px;background:#333333;"></div>
    <div style="font-size:11px;color:#A1A1A6;font-weight:600;letter-spacing:0.05em;">OR CAPTURE</div>
    <div style="flex:1;height:1px;background:#333333;"></div>
    </div>
    """, unsafe_allow_html=True)

        camera_image = st.camera_input("📷 Capture Prescription")

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        _, c_demo, _ = st.columns([1, 1.5, 1])
        with c_demo:
            if st.button("Launch Preloaded Demo Mode", use_container_width=True):
                st.session_state.update(result=DEMO_DATA, ocr_text="", view="results")
                st.rerun()

        # Process either uploaded file or camera image
        file_to_process = uploaded_file or camera_image

        if file_to_process:
            suffix = Path(file_to_process.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_to_process.getvalue())
                path = tmp.name
        
            # PRESCRIPTION MODE: OCR -> Review -> Analysis
            slot = st.empty()
            with slot: _loading_card(SVG_ICONS["scan"], "Parsing Document Text Matrix...", "Scanning layout architectures via high-density Vision models")
            try:
                txt, src = run_ocr(path)
                st.session_state.update(ocr_text=txt, ocr_source=src, view="review")
                os.unlink(path)
                st.rerun()
            except Exception as e:
                slot.empty()
                st.error(f"OCR Failure State: {e}")


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DISEASE PREDICTOR  (ML)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ml:
        render_ml_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — HEALTH ASSISTANT  (AI Chatbot + Report Vision Analysis)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_chat:
        # Define SVG Data URIs for avatars
        import base64
        user_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A1A1A6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        ai_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#007AFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/><path d="M12 8V4H8"/></svg>'
        user_avatar = f"data:image/svg+xml;base64,{base64.b64encode(user_svg.encode('utf-8')).decode('utf-8')}"
        ai_avatar = f"data:image/svg+xml;base64,{base64.b64encode(ai_svg.encode('utf-8')).decode('utf-8')}"
        
        st.markdown(f"""
<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:14px;padding:16px 20px;margin-bottom:20px;">
<div style="font-size:16px;font-weight:700;color:#FFFFFF;margin-bottom:4px;display:flex;align-items:center;gap:8px;">
  <img src="{ai_avatar}" width="20" height="20" /> AI Health Assistant
</div>
<div style="font-size:13px;color:#A1A1A6;">Ask about symptoms, diseases, medications, or your prediction results.
This is for <b style="color:#FFD60A;">educational purposes only</b> — not a substitute for professional medical advice.</div>
</div>
""", unsafe_allow_html=True)
        # Embed interactive 3D Spline scene with a wrapper to crop the community header
        import streamlit.components.v1 as components
        components.html("""
        <div style="width: 100%; height: 450px; overflow: hidden; border-radius: 14px; background: #111;">
            <iframe src="https://app.spline.design/community/file/615b9422-9985-43f6-8593-d7d7bc3b0be1" 
                    style="width: 100%; height: 600px; border: none; margin-top: -140px;" 
                    scrolling="no"
                    allow="pointer-lock">
            </iframe>
        </div>
        """, height=450)
        

        # ── Session state init ────────────────────────────────────────────────
        if "health_chat_history" not in st.session_state:
            st.session_state.health_chat_history = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I'm your AI Health Assistant\n\n"
                        "I can help you understand symptoms, diseases, medications, test results, "
                        "or explain what any of the predictions in this app mean.\n\n"
                        "You can also tap **+** below to upload or capture a photo of any medical "
                        "report — I'll read it and tell you what it means in plain language.\n\n"
                        "What would you like to know?"
                    )
                }
            ]
        if "chat_attach_open" not in st.session_state:
            st.session_state.chat_attach_open = False
        if "chat_attach_mode" not in st.session_state:
            st.session_state.chat_attach_mode = "upload"   # "upload" | "camera"

        # ── Chat history ──────────────────────────────────────────────────────
        for msg in st.session_state.health_chat_history:
            with st.chat_message(msg["role"], avatar=ai_avatar if msg["role"] == "assistant" else user_avatar):
                st.markdown(msg["content"])

        # ── Attachment panel (shown when ➕ is active) ─────────────────────────
        REPORT_VISION_PROMPT = """\
You are an expert medical AI assistant. A patient has shared a photo of a medical document.

Carefully analyse this image and do ALL of the following:

1. **Identify the report type** — What kind of test/report is this? (e.g. Complete Blood Count, LFT, Kidney Function Test, ECG, X-Ray, Prescription, etc.)

2. **Extract all values** — List every measurable parameter you can see, its value, unit, and whether it is Normal / Low / High based on standard reference ranges. Format as a table if multiple values are present.

3. **Plain-language summary** — In 3-5 sentences, explain to the patient in simple terms what these results suggest about their health. Use language a non-medical person can understand.

4. **Action items** — Based on any abnormal values, suggest what type of doctor/specialist they should consult and how urgently (routine check-up / within a week / urgent).

5. **Disclaimer** — Remind them this is educational only and not a substitute for professional medical advice.

If this is not a medical document (e.g. a regular photo), politely say so and ask them to upload a medical report instead.
"""

        def _analyse_report_image(img_bytes: bytes, suffix: str) -> str:
            """Send image to Sarvam→Groq vision pipeline and return analysis text."""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            try:
                img_b64, mime_type = load_image_b64(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            return _ocr_with_prompt(img_b64, mime_type, REPORT_VISION_PROMPT)

        if st.session_state.chat_attach_open:
            st.markdown("""
<div style="background:linear-gradient(135deg,#1C1C1E,#111111);border:1px solid #2C2C2E;
border-radius:16px;padding:0px 0px 4px 0px;margin-bottom:12px;overflow:hidden;">
<div style="background:#2C2C2E;padding:10px 16px;display:flex;justify-content:space-between;align-items:center;">
  <span style="font-size:13px;font-weight:700;color:#FFFFFF;display:flex;align-items:center;gap:6px;">
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
    Attach a Medical Report
  </span>
  <span style="font-size:11px;color:#636366;">Powered by Sarvam Vision + Groq</span>
</div>
""", unsafe_allow_html=True)

            # Mode selector
            mode_col1, mode_col2 = st.columns(2)
            with mode_col1:
                if st.button(
                    "Upload",
                    use_container_width=True,
                    type="primary" if st.session_state.chat_attach_mode == "upload" else "secondary",
                    key="attach_mode_upload",
                ):
                    st.session_state.chat_attach_mode = "upload"
                    st.rerun()
            with mode_col2:
                if st.button(
                    "Capture with Camera",
                    use_container_width=True,
                    type="primary" if st.session_state.chat_attach_mode == "camera" else "secondary",
                    key="attach_mode_camera",
                ):
                    st.session_state.chat_attach_mode = "camera"
                    st.rerun()

            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

            attached_img_bytes = None
            attached_suffix    = ".jpg"

            if st.session_state.chat_attach_mode == "upload":
                st.markdown("""
<div style="padding:4px 16px 0px 16px;font-size:12px;color:#A1A1A6;">
Upload any medical report, lab result, prescription, X-Ray, or health document.
</div>""", unsafe_allow_html=True)
                up_file = st.file_uploader(
                    "Upload report",
                    type=["jpg", "jpeg", "png", "pdf"],
                    key="chat_report_upload",
                    label_visibility="collapsed",
                )
                if up_file:
                    attached_img_bytes = up_file.read()
                    attached_suffix    = Path(getattr(up_file, "name", "report.jpg")).suffix or ".jpg"
                    st.image(
                        up_file if attached_suffix.lower() != ".pdf" else None,
                        caption="Report preview",
                        use_container_width=True,
                    ) if attached_suffix.lower() != ".pdf" else st.info("PDF uploaded — AI will analyse the first page.")

            else:  # camera mode
                st.markdown("""
<div style="padding:4px 16px 0px 16px;font-size:12px;color:#A1A1A6;">
Point your camera at the report and take a photo. Make sure it's well-lit and text is clear.
</div>""", unsafe_allow_html=True)
                cam_img = st.camera_input(
                    "Capture report",
                    key="chat_report_camera",
                    label_visibility="collapsed",
                )
                if cam_img:
                    attached_img_bytes = cam_img.read()
                    attached_suffix    = ".jpg"

            # Analyse button — shown when image is ready
            if attached_img_bytes:
                st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                if st.button(
                    "Analyse Report & Send to Chat",
                    type="primary",
                    use_container_width=True,
                    key="btn_analyse_report",
                ):
                    # Add user message to chat
                    st.session_state.health_chat_history.append({
                        "role": "user",
                        "content": "*I've uploaded a medical report. Please analyse it and tell me what it means.*"
                    })

                    # Run vision analysis
                    with st.spinner("AI reading your report…"):
                        try:
                            analysis = _analyse_report_image(attached_img_bytes, attached_suffix)
                        except Exception as e:
                            analysis = f"Sorry, I couldn't read the report right now. ({e})\n\nPlease try again or describe your report in text."

                    # Add AI reply to chat
                    st.session_state.health_chat_history.append({
                        "role": "assistant",
                        "content": analysis
                    })

                    # Close panel and refresh
                    st.session_state.chat_attach_open = False
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Input row:  + button  +  chat_input ─────────────────────────────
        st.markdown("""
<style>
/* Style chat input like ai-prompt-box */
div[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
div[data-testid="stChatInput"] > div {
    background-color: #1F2023 !important;
    border: 1px solid #444444 !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.24) !important;
}
/* Force internal wrappers to be transparent */
div[data-testid="stChatInput"] > div > div, 
div[data-testid="stChatInput"] > div > div > div,
div[data-testid="stChatInput"] .stChatInputTextArea {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #F3F4F6 !important;
}
/* Push the + button to sit flush with the chat input */
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
    height: 52px !important;
    width: 52px !important;
    min-height: 52px !important;
    border-radius: 50% !important;
    font-size: 22px !important;
    padding: 0 !important;
    background: #1F2023 !important;
    border: 1px solid #444444 !important;
    color: #9CA3AF !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s;
}
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button p {
    margin: 0 !important;
}
div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover {
    background: rgba(75, 85, 99, 0.3) !important;
    color: #D1D5DB !important;
}
</style>
""", unsafe_allow_html=True)

        plus_col, input_col = st.columns([1, 10])
        with plus_col:
            if st.button(
                "+" if not st.session_state.chat_attach_open else "x",
                key="toggle_attach_btn",
                help="Attach a medical report photo",
            ):
                st.session_state.chat_attach_open = not st.session_state.chat_attach_open
                st.session_state.chat_attach_mode = "upload"
                st.rerun()

        with input_col:
            if user_input := st.chat_input("Ask anything about health, symptoms or diseases…"):
                st.session_state.health_chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user", avatar=user_avatar):
                    st.markdown(user_input)

                with st.chat_message("assistant", avatar=ai_avatar):
                    with st.spinner("Thinking…"):
                        try:
                            system_prompt = (
                                "You are a knowledgeable and empathetic AI Health Assistant integrated into a Healthcare AI Suite. "
                                "You help users understand medical conditions, symptoms, medications, and health predictions. "
                                "Always be clear, accurate, and concise. Use plain language the user can understand. "
                                "Always remind users that your responses are for educational purposes only and they should "
                                "consult a qualified healthcare professional for diagnosis or treatment. "
                                "When discussing prediction results from the app (diabetes, heart disease, etc.), "
                                "explain what the result means, what factors contribute to it, and what steps they might consider. "
                                "Do not prescribe medications or give specific treatment plans."
                            )
                            messages = [{"role": "system", "content": system_prompt}]
                            for m in st.session_state.health_chat_history[-10:]:
                                messages.append({"role": m["role"], "content": m["content"]})

                            resp = requests.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                                json={
                                    "model": "llama-3.3-70b-versatile",
                                    "messages": messages,
                                    "temperature": 0.5,
                                    "max_tokens": 1024,
                                },
                                timeout=60,
                            )
                            if resp.status_code != 200:
                                raise RuntimeError(f"Groq error {resp.status_code}: {resp.text}")
                            reply = resp.json()["choices"][0]["message"]["content"].strip()
                        except Exception as e:
                            reply = f"Sorry, I couldn't get a response right now. ({e})"

                    st.markdown(reply)
                    st.session_state.health_chat_history.append({"role": "assistant", "content": reply})

        # Clear chat
        if len(st.session_state.health_chat_history) > 1:
            if st.button("Clear conversation", key="clear_chat"):
                st.session_state.health_chat_history = [st.session_state.health_chat_history[0]]
                st.session_state.chat_attach_open = False
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — HEALTH TOOLS  (Pure calculators, no API/model needed)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_tools:
        st.markdown("""
<div style="background:#1C1C1E;border:1px solid #2C2C2E;border-radius:14px;padding:16px 20px;margin-bottom:20px;">
<div style="font-size:16px;font-weight:700;color:#FFFFFF;margin-bottom:4px;">🧮 Health Calculators</div>
<div style="font-size:13px;color:#A1A1A6;">Instant health metrics — no AI required. Results are estimates for general wellness awareness only.</div>
</div>
""", unsafe_allow_html=True)

        tool_tabs = st.tabs(["BMI", "Blood Pressure", "Daily Calories", "Water Intake", "Vaccination Tracker", "Family Profiles", "Cycle Tracker", "Lab Reports"])

        # ── BMI Calculator ────────────────────────────────────────────────────
        with tool_tabs[0]:
            st.markdown("#### ⚖️ Body Mass Index (BMI)")
            c1, c2 = st.columns(2)
            with c1:
                unit_sys = st.radio("Unit system", ["Metric (kg / cm)", "Imperial (lb / in)"], horizontal=True, key="bmi_units")
            with c2:
                bmi_age = st.number_input("Age (years)", 2, 120, 25, key="bmi_age")

            if unit_sys.startswith("Metric"):
                col_h, col_w = st.columns(2)
                with col_h: height_cm = st.number_input("Height (cm)", 50.0, 250.0, 170.0, key="bmi_h_cm")
                with col_w: weight_kg = st.number_input("Weight (kg)", 10.0, 300.0, 70.0, key="bmi_w_kg")
                bmi = weight_kg / ((height_cm / 100) ** 2)
            else:
                col_h, col_w = st.columns(2)
                with col_h: height_in = st.number_input("Height (in)", 20.0, 100.0, 67.0, key="bmi_h_in")
                with col_w: weight_lb = st.number_input("Weight (lb)", 20.0, 660.0, 154.0, key="bmi_w_lb")
                bmi = (weight_lb / (height_in ** 2)) * 703

            if bmi < 18.5:
                cat, col, tip = "Underweight", "#64D2FF", "Consider increasing caloric intake with nutrient-dense foods and consult a dietitian."
            elif bmi < 25:
                cat, col, tip = "Normal weight", "#30D158", "Great! Maintain your healthy lifestyle with balanced diet and regular activity."
            elif bmi < 30:
                cat, col, tip = "Overweight", "#FFD60A", "Moderate exercise and portion control can help bring BMI into the healthy range."
            else:
                cat, col, tip = "Obese", "#FF453A", "Consult a healthcare provider to discuss a safe weight management plan."

            st.markdown(f"""
<div style="background:#1C1C1E;border:2px solid {col};border-radius:14px;padding:20px;margin-top:12px;text-align:center;">
  <div style="font-size:48px;font-weight:800;color:{col};">{bmi:.1f}</div>
  <div style="font-size:18px;font-weight:700;color:#FFFFFF;margin:4px 0;">{cat}</div>
  <div style="font-size:12px;color:#A1A1A6;margin-top:8px;">{tip}</div>
  <div style="display:flex;justify-content:center;gap:8px;margin-top:14px;flex-wrap:wrap;">
    <span style="background:#2C2C2E;border-radius:20px;padding:4px 12px;font-size:11px;color:#64D2FF;">Underweight &lt;18.5</span>
    <span style="background:#2C2C2E;border-radius:20px;padding:4px 12px;font-size:11px;color:#30D158;">Normal 18.5–24.9</span>
    <span style="background:#2C2C2E;border-radius:20px;padding:4px 12px;font-size:11px;color:#FFD60A;">Overweight 25–29.9</span>
    <span style="background:#2C2C2E;border-radius:20px;padding:4px 12px;font-size:11px;color:#FF453A;">Obese ≥30</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Blood Pressure Classifier ─────────────────────────────────────────
        with tool_tabs[1]:
            st.markdown("#### 🩺 Blood Pressure Classifier")
            st.caption("Enter your last blood pressure reading to see what category it falls into.")
            bc1, bc2 = st.columns(2)
            with bc1: systolic  = st.number_input("Systolic (mmHg)  — top number",  60, 300, 120, key="bp_sys")
            with bc2: diastolic = st.number_input("Diastolic (mmHg) — bottom number", 40, 200,  80, key="bp_dia")

            if systolic < 120 and diastolic < 80:
                bp_cat, bp_col, bp_tip = "Normal", "#30D158", "Your blood pressure is in the healthy range. Keep up the good habits!"
            elif systolic < 130 and diastolic < 80:
                bp_cat, bp_col, bp_tip = "Elevated", "#64D2FF", "Slightly above normal. Reduce sodium, exercise regularly, and monitor it."
            elif systolic < 140 or diastolic < 90:
                bp_cat, bp_col, bp_tip = "High — Stage 1", "#FFD60A", "Consult a doctor. Lifestyle changes and possibly medication may be recommended."
            elif systolic >= 140 or diastolic >= 90:
                bp_cat, bp_col, bp_tip = "High — Stage 2", "#FF6B35", "See a doctor soon. This level usually requires medication alongside lifestyle changes."
            if systolic > 180 or diastolic > 120:
                bp_cat, bp_col, bp_tip = "⚠️ Hypertensive Crisis", "#FF453A", "Seek emergency medical care immediately if you have symptoms like chest pain or shortness of breath."

            st.markdown(f"""
<div style="background:#1C1C1E;border:2px solid {bp_col};border-radius:14px;padding:20px;margin-top:12px;text-align:center;">
  <div style="font-size:36px;font-weight:800;color:{bp_col};">{systolic} / {diastolic}</div>
  <div style="font-size:18px;font-weight:700;color:#FFFFFF;margin:6px 0;">{bp_cat}</div>
  <div style="font-size:12px;color:#A1A1A6;margin-top:6px;">{bp_tip}</div>
</div>
<div style="background:#1C1C1E;border-radius:10px;padding:12px 16px;margin-top:12px;font-size:12px;color:#636366;">
  <b style="color:#8E8E93;">Reference ranges (AHA guidelines):</b><br>
  Normal &lt;120/80 · Elevated 120–129/&lt;80 · Stage 1 HBP 130–139/80–89 · Stage 2 HBP ≥140/90 · Crisis &gt;180/120
</div>
""", unsafe_allow_html=True)

        # ── Daily Calorie Calculator ──────────────────────────────────────────
        with tool_tabs[2]:
            st.markdown("#### 🔥 Daily Calorie Needs (TDEE)")
            st.caption("Based on the Mifflin–St Jeor equation, the most accurate for most adults.")
            dc1, dc2 = st.columns(2)
            with dc1:
                cal_gender = st.radio("Sex assigned at birth", ["Male", "Female"], horizontal=True, key="cal_sex")
                cal_age    = st.number_input("Age", 15, 100, 25, key="cal_age")
                cal_weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, key="cal_wt")
            with dc2:
                cal_height = st.number_input("Height (cm)", 100.0, 230.0, 170.0, key="cal_ht")
                activity   = st.selectbox("Activity level", [
                    "Sedentary (desk job, little exercise)",
                    "Lightly active (1–3 days/week)",
                    "Moderately active (3–5 days/week)",
                    "Very active (6–7 days/week)",
                    "Extra active (physical job + training)",
                ], key="cal_act")

            activity_map = {
                "Sedentary (desk job, little exercise)": 1.2,
                "Lightly active (1–3 days/week)": 1.375,
                "Moderately active (3–5 days/week)": 1.55,
                "Very active (6–7 days/week)": 1.725,
                "Extra active (physical job + training)": 1.9,
            }
            if cal_gender == "Male":
                bmr = 10 * cal_weight + 6.25 * cal_height - 5 * cal_age + 5
            else:
                bmr = 10 * cal_weight + 6.25 * cal_height - 5 * cal_age - 161

            tdee      = bmr * activity_map[activity]
            lose_fast = tdee - 500
            lose_slow = tdee - 250
            gain_slow = tdee + 250
            gain_fast = tdee + 500

            st.markdown(f"""
<div style="background:#1C1C1E;border:2px solid #30D158;border-radius:14px;padding:20px;margin-top:12px;text-align:center;">
  <div style="font-size:13px;color:#A1A1A6;margin-bottom:4px;">Your maintenance calories</div>
  <div style="font-size:52px;font-weight:800;color:#30D158;">{tdee:,.0f}</div>
  <div style="font-size:14px;color:#8E8E93;">kcal / day</div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-top:12px;">
  <div style="background:#1C1C1E;border-radius:10px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#FF453A;font-weight:600;">FAST LOSS</div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;">{lose_fast:,.0f}</div>
    <div style="font-size:10px;color:#636366;">~0.5 kg/week</div>
  </div>
  <div style="background:#1C1C1E;border-radius:10px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#FFD60A;font-weight:600;">SLOW LOSS</div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;">{lose_slow:,.0f}</div>
    <div style="font-size:10px;color:#636366;">~0.25 kg/week</div>
  </div>
  <div style="background:#1C1C1E;border-radius:10px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#64D2FF;font-weight:600;">SLOW GAIN</div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;">{gain_slow:,.0f}</div>
    <div style="font-size:10px;color:#636366;">~0.25 kg/week</div>
  </div>
  <div style="background:#1C1C1E;border-radius:10px;padding:12px;text-align:center;">
    <div style="font-size:11px;color:#BF5AF2;font-weight:600;">FAST GAIN</div>
    <div style="font-size:20px;font-weight:700;color:#FFFFFF;">{gain_fast:,.0f}</div>
    <div style="font-size:10px;color:#636366;">~0.5 kg/week</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Water Intake Calculator ───────────────────────────────────────────
        with tool_tabs[3]:
            st.markdown("#### 💧 Daily Water Intake")
            st.caption("General guideline based on weight and activity level.")
            wc1, wc2 = st.columns(2)
            with wc1: water_weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0, key="wat_wt")
            with wc2: water_act    = st.radio("Activity level", ["Low", "Moderate", "High"], horizontal=True, key="wat_act")

            base_water = water_weight * 0.033
            bonus = {"Low": 0.0, "Moderate": 0.35, "High": 0.6}[water_act]
            total_water = base_water + bonus
            glasses = total_water / 0.25

            st.markdown(f"""
<div style="background:#1C1C1E;border:2px solid #64D2FF;border-radius:14px;padding:20px;margin-top:12px;text-align:center;">
  <div style="font-size:52px;font-weight:800;color:#64D2FF;">{total_water:.1f} L</div>
  <div style="font-size:15px;color:#FFFFFF;margin:4px 0;">≈ {glasses:.0f} glasses of 250 ml per day</div>
  <div style="font-size:12px;color:#A1A1A6;margin-top:8px;">
    This increases with heat, illness, or intense exercise. Coffee and tea also count toward intake.
  </div>
</div>
""", unsafe_allow_html=True)

        # ── VACCINATION TRACKER ───────────────────────────────────────────────────
        with tool_tabs[4]:
            st.markdown("#### 💉 Vaccination Tracker (India Schedule)")
            
            # Initialize session state for vaccinations
            if "vaccinations" not in st.session_state:
                st.session_state.vaccinations = {}
            
            # India's standard vaccination schedule
            INDIA_VACCINATION_SCHEDULE = {
                "BCG": {"age": "Birth", "due_days": 0, "description": "Tuberculosis"},
                "OPV Dose 1": {"age": "6 weeks", "due_days": 42, "description": "Polio - Dose 1"},
                "Pentavalent Dose 1": {"age": "6 weeks", "due_days": 42, "description": "DPT + HepB + Hib - Dose 1"},
                "Rotavirus Dose 1": {"age": "6 weeks", "due_days": 42, "description": "Rotavirus - Dose 1"},
                "OPV Dose 2": {"age": "10 weeks", "due_days": 70, "description": "Polio - Dose 2"},
                "Pentavalent Dose 2": {"age": "10 weeks", "due_days": 70, "description": "DPT + HepB + Hib - Dose 2"},
                "Rotavirus Dose 2": {"age": "10 weeks", "due_days": 70, "description": "Rotavirus - Dose 2"},
                "OPV Dose 3": {"age": "14 weeks", "due_days": 98, "description": "Polio - Dose 3"},
                "Pentavalent Dose 3": {"age": "14 weeks", "due_days": 98, "description": "DPT + HepB + Hib - Dose 3"},
                "Rotavirus Dose 3": {"age": "14 weeks", "due_days": 98, "description": "Rotavirus - Dose 3"},
                "PCV Dose 1": {"age": "6 weeks", "due_days": 42, "description": "Pneumococcal - Dose 1"},
                "PCV Dose 2": {"age": "10 weeks", "due_days": 70, "description": "Pneumococcal - Dose 2"},
                "PCV Booster": {"age": "12-15 months", "due_days": 365, "description": "Pneumococcal Booster"},
                "MMR": {"age": "9-12 months", "due_days": 270, "description": "Measles, Mumps, Rubella"},
                "OPV Booster 1": {"age": "18-24 months", "due_days": 540, "description": "Polio Booster 1"},
                "DPT Booster 1": {"age": "18-24 months", "due_days": 540, "description": "DPT Booster 1"},
                "Varicella": {"age": "12-18 months", "due_days": 365, "description": "Chickenpox"},
            }
            
            dob = st.date_input(
                "Child's Date of Birth", 
                value=datetime.now() - timedelta(days=365),
                min_value=datetime(1900, 1, 1).date(),
                max_value=datetime(2099, 12, 31).date()
            )
            
            st.markdown("**Vaccination Status:**")
            
            for vaccine_name, vaccine_info in INDIA_VACCINATION_SCHEDULE.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"💉 {vaccine_name}")
                    st.caption(vaccine_info["description"])
                
                with col2:
                    due_date = dob + timedelta(days=vaccine_info["due_days"])
                    today = datetime.now().date()
                    days_remaining = (due_date - today).days
                    
                    if vaccine_name in st.session_state.vaccinations:
                        st.success(f"✅ Given: {st.session_state.vaccinations[vaccine_name]}")
                    else:
                        if days_remaining > 7:
                            st.info(f"📅 Due: {due_date.strftime('%b %d')}")
                        elif days_remaining > 0:
                            st.warning(f"⚠️ Due soon: {due_date.strftime('%b %d')}")
                        else:
                            st.error(f"⏰ Overdue! {abs(days_remaining)} days")
                
                with col3:
                    if st.checkbox(f"Given", value=vaccine_name in st.session_state.vaccinations, key=f"vacc_{vaccine_name}"):
                        if vaccine_name not in st.session_state.vaccinations:
                            st.session_state.vaccinations[vaccine_name] = datetime.now().strftime("%Y-%m-%d")
                    else:
                        if vaccine_name in st.session_state.vaccinations:
                            del st.session_state.vaccinations[vaccine_name]
            
            if st.session_state.vaccinations:
                st.markdown("---")
                st.markdown(f"**✅ Completed:** {len(st.session_state.vaccinations)} vaccines")
                st.markdown(f"**📋 Remaining:** {len(INDIA_VACCINATION_SCHEDULE) - len(st.session_state.vaccinations)} vaccines")

        # ── FAMILY HEALTH PROFILES (Detailed Editor) ────────────────────────────────
        with tool_tabs[5]:
            st.markdown("#### 👨‍👩‍👧 Family Health Profile Editor")
            st.caption("👆 Switch profiles from the top-right corner | 📝 Edit profile details here")
            
            # Get current profile from header selector
            profile_name = st.session_state.get("current_profile", "Self")
            
            # Create default profile if none exists
            if not st.session_state.family_profiles:
                st.session_state.family_profiles["Self"] = {
                    "age": 25,
                    "gender": "Other",
                    "blood_group": "O+",
                    "medical_conditions": [],
                    "allergies": [],
                    "medications": [],
                    "health_notes": ""
                }
            
            profile = st.session_state.family_profiles.get(profile_name, {})
            
            st.markdown(f"**Editing Profile:** `{profile_name}`")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                profile["age"] = st.number_input("Age", 0, 120, profile.get("age", 25), key=f"{profile_name}_age")
            with col2:
                profile["gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                                 index=["Male", "Female", "Other"].index(profile.get("gender", "Other")),
                                                 key=f"{profile_name}_gender")
            with col3:
                profile["blood_group"] = st.selectbox("Blood Group", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"],
                                                       index=["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"].index(profile.get("blood_group", "O+")),
                                                       key=f"{profile_name}_bg")
            
            st.markdown("**Medical Info:**")
            profile["medical_conditions"] = st.multiselect("Existing Conditions", 
                                                           ["Diabetes", "Hypertension", "Asthma", "Thyroid", "Heart Disease", "Arthritis"],
                                                           default=profile.get("medical_conditions", []),
                                                           key=f"{profile_name}_conditions")
            profile["allergies"] = st.multiselect("Allergies",
                                                  ["Penicillin", "Ibuprofen", "Peanuts", "Shellfish", "Eggs", "Milk"],
                                                  default=profile.get("allergies", []),
                                                  key=f"{profile_name}_allergies")
            
            profile["medications"] = st.text_area("Current Medications (one per line):",
                                                  value="\n".join(profile.get("medications", [])),
                                                  key=f"{profile_name}_meds")
            profile["medications"] = [m.strip() for m in profile["medications"].split("\n") if m.strip()]
            
            profile["health_notes"] = st.text_area("Health Notes:",
                                                   value=profile.get("health_notes", ""),
                                                   key=f"{profile_name}_notes")
            
            st.session_state.family_profiles[profile_name] = profile
            
            # Delete profile button (not available for "Self")
            if profile_name != "Self":
                col_del, col_space = st.columns([1, 4])
                with col_del:
                    if st.button("🗑️ Remove Profile", type="secondary"):
                        del st.session_state.family_profiles[profile_name]
                        st.session_state.current_profile = "Self"
                        st.rerun()

        # ── MENSTRUAL CYCLE TRACKER ────────────────────────────────────────────────
        with tool_tabs[6]:
            st.markdown("#### 🗓️ Menstrual Cycle Tracker")
            
            if "cycle_history" not in st.session_state:
                st.session_state.cycle_history = []
            if "cycle_length" not in st.session_state:
                st.session_state.cycle_length = 28
            
            st.markdown("**Track Your Cycle:**")
            
            col1, col2 = st.columns(2)
            with col1:
                last_period = st.date_input("Last Period Start Date:", value=datetime.now() - timedelta(days=14))
            with col2:
                cycle_length = st.slider("Average Cycle Length (days):", 21, 35, st.session_state.cycle_length)
                st.session_state.cycle_length = cycle_length
            
            # Calculate cycle phases
            today = datetime.now().date()
            cycle_day = (today - last_period).days % cycle_length
            
            menstrual_start = last_period
            menstrual_end = last_period + timedelta(days=5)
            follicular_end = last_period + timedelta(days=13)
            ovulation_day = last_period + timedelta(days=14)
            ovulation_window_start = last_period + timedelta(days=12)
            ovulation_window_end = last_period + timedelta(days=16)
            fertile_window_start = last_period + timedelta(days=8)
            fertile_window_end = last_period + timedelta(days=17)
            luteal_start = last_period + timedelta(days=15)
            
            next_period = last_period + timedelta(days=cycle_length)
            
            # Determine current phase
            if cycle_day <= 5:
                current_phase = "🔴 Menstrual"
                phase_color = "#FF453A"
            elif cycle_day <= 13:
                current_phase = "🟢 Follicular"
                phase_color = "#30D158"
            elif cycle_day <= 16:
                current_phase = "🟡 Ovulation"
                phase_color = "#FFD60A"
            else:
                current_phase = "🟣 Luteal"
                phase_color = "#BF5AF2"
            
            st.markdown(f"""
<div style="background:#1C1C1E;border:2px solid {phase_color};border-radius:14px;padding:20px;margin:12px 0;text-align:center;">
  <div style="font-size:28px;font-weight:800;color:{phase_color};margin-bottom:8px;">{current_phase}</div>
  <div style="font-size:14px;color:#A1A1A6;">Day {cycle_day} of {cycle_length}</div>
  <div style="font-size:12px;color:#8E8E93;margin-top:8px;">Next period: {next_period.strftime('%b %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("**Cycle Phases:**")
            
            phases_data = [
                ("🔴 Menstrual", menstrual_start.strftime("%b %d"), menstrual_end.strftime("%b %d"), "Bleeding, fatigue, cramps", "#FF453A"),
                ("🟢 Follicular", (last_period + timedelta(days=6)).strftime("%b %d"), follicular_end.strftime("%b %d"), "Energy rising, good mood", "#30D158"),
                ("🟡 Ovulation", ovulation_window_start.strftime("%b %d"), ovulation_window_end.strftime("%b %d"), "Highest fertility, energy peak", "#FFD60A"),
                ("🟣 Luteal", luteal_start.strftime("%b %d"), (last_period + timedelta(days=cycle_length - 1)).strftime("%b %d"), "PMS symptoms, energy drops", "#BF5AF2"),
            ]
            
            for phase, start, end, description, color in phases_data:
                st.markdown(f"**{phase}** ({start} to {end})")
                st.caption(description)
            
            st.markdown("---")
            st.markdown("**Fertility Window:**")
            fertile_days = (fertile_window_end - fertile_window_start).days
            st.info(f"🎯 Most fertile: {fertile_window_start.strftime('%b %d')} to {fertile_window_end.strftime('%b %d')} ({fertile_days} days)")
            st.success(f"🔝 Peak ovulation: {ovulation_day.strftime('%b %d, %Y')}")
            
            # Log period
            st.markdown("**Log Period:**")
            log_date = st.date_input("Date started:", value=today, key="log_period_date")
            if st.button("📝 Log Today's Period"):
                st.session_state.cycle_history.append(log_date.isoformat())
                st.success(f"✅ Period logged for {log_date.strftime('%b %d')}")

        # ── LAB REPORT HISTORY ─────────────────────────────────────────────────────
        with tool_tabs[7]:
            st.markdown("#### 📋 Lab Report History")
            
            if "lab_reports" not in st.session_state:
                st.session_state.lab_reports = []
            
            st.markdown("**Add New Report:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                report_type = st.selectbox("Report Type", ["Blood Test", "Urine Test", "Thyroid", "Diabetes", "Liver Function", "Kidney Function", "Lipid Profile", "Other"])
            with col2:
                report_date = st.date_input("Report Date", value=datetime.now())
            with col3:
                test_name = st.text_input("Test Name (e.g., HbA1c, Glucose)", "")
            
            col4, col5 = st.columns(2)
            with col4:
                test_value = st.text_input("Result Value", "")
            with col5:
                reference_range = st.text_input("Reference Range (e.g., 70-100)", "")
            
            report_notes = st.text_area("Notes", "")
            
            if st.button("💾 Save Report"):
                if test_name and test_value:
                    st.session_state.lab_reports.append({
                        "date": report_date.isoformat(),
                        "type": report_type,
                        "test_name": test_name,
                        "value": test_value,
                        "reference": reference_range,
                        "notes": report_notes
                    })
                    st.success("✅ Report saved successfully!")
                    st.rerun()
                else:
                    st.error("Please enter test name and value")
            
            if st.session_state.lab_reports:
                st.markdown("---")
                st.markdown("**Saved Reports:**")
                
                # Sort by date (newest first)
                sorted_reports = sorted(st.session_state.lab_reports, key=lambda x: x["date"], reverse=True)
                
                for idx, report in enumerate(sorted_reports):
                    with st.expander(f"📄 {report['type']} - {report['test_name']} ({report['date']})"):
                        col_info = st.columns(2)
                        with col_info[0]:
                            st.write(f"**Test:** {report['test_name']}")
                            st.write(f"**Result:** {report['value']}")
                        with col_info[1]:
                            st.write(f"**Reference:** {report['reference']}")
                            st.write(f"**Date:** {report['date']}")
                        
                        if report['notes']:
                            st.write(f"**Notes:** {report['notes']}")
                        
                        if st.button("🗑️ Delete", key=f"del_report_{idx}"):
                            st.session_state.lab_reports.pop(idx)
                            st.rerun()
                
                # Trend visualization
                st.markdown("---")
                st.markdown("**Report Trends:**")
                
                # Group reports by test name
                test_trends = {}
                for report in sorted_reports:
                    test = report['test_name']
                    if test not in test_trends:
                        test_trends[test] = []
                    try:
                        test_trends[test].append({
                            "date": datetime.fromisoformat(report['date']),
                            "value": float(report['value'].split()[0]) if report['value'] else 0
                        })
                    except:
                        pass
                
                selected_test = st.selectbox("Select test to view trend:", list(test_trends.keys()))
                
                
                if selected_test and test_trends[selected_test]:
                    trend_data = sorted(test_trends[selected_test], key=lambda x: x['date'])
                    dates = [d['date'].strftime('%b %d') for d in trend_data]
                    values = [d['value'] for d in trend_data]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(dates, values, marker='o', color='#007AFF', linewidth=2, markersize=8)
                    ax.set_title(f'{selected_test} Trend', fontsize=14, fontweight='bold', color='#FFFFFF')
                    ax.set_xlabel('Date', fontsize=11, color='#A1A1A6')
                    ax.set_ylabel('Value', fontsize=11, color='#A1A1A6')
                    ax.grid(True, alpha=0.2)
                    fig.patch.set_facecolor('#1C1C1E')
                    ax.set_facecolor('#111111')
                    ax.tick_params(colors='#A1A1A6')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("📝 No reports saved yet. Add your first lab report above!")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — HEALTH TRENDS  (Longitudinal Time-Series)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_trend:
        st.markdown("## 📈 AI Health Trajectory Analysis")
        st.write("Upload multiple past lab reports (e.g. Lipid Profile, Complete Blood Count) to see how your health parameters have trended over time. Our Vision AI will automatically extract the dates and values.")
        
        uploaded_trend_files = st.file_uploader("Upload Historical Reports", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        report_type_trend = st.selectbox("What type of reports are these?", ["Lipid Profile", "Complete Blood Count (CBC)", "Liver Function Test (LFT)", "Diabetes / Blood Sugar", "Other"])
        if report_type_trend == "Other":
            report_type_trend = st.text_input("Please specify the report type:")
        
        if st.button("🚀 Analyze Trajectory"):
            if not uploaded_trend_files:
                st.warning("Please upload at least one report.")
            else:
                st.info(f"Processing {len(uploaded_trend_files)} reports using Vision AI...")
                
                all_extracted_data = []
                
                with st.spinner("Extracting parameters and dates..."):
                    import json
                    import tempfile, os
                    import pandas as pd
                    
                    for i, file in enumerate(uploaded_trend_files):
                        img_bytes = file.read()
                        
                        prompt = f"""
                        You are a data extraction AI. Look at this {report_type_trend} medical report.
                        Extract the "Collection Date" or "Report Date" in YYYY-MM-DD format.
                        Extract all key numerical parameters and their values.
                        
                        You MUST return ONLY a valid JSON object in this exact format:
                        {{
                            "date": "YYYY-MM-DD",
                            "parameters": {{
                                "Parameter Name": float_value,
                                "Another Parameter": float_value
                            }}
                        }}
                        If you cannot find a date, return "UNKNOWN". Make sure values are numbers, not strings.
                        Do NOT wrap the JSON in markdown blocks like ```json. Return ONLY raw JSON.
                        """
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                                tmp.write(img_bytes)
                                tmp_path = tmp.name
                                
                            try:
                                img_b64, mime = load_image_b64(tmp_path)
                                result_text = _ocr_with_prompt(img_b64, mime, prompt)
                                cleaned = result_text.replace("```json", "").replace("```", "").strip()
                                data = json.loads(cleaned)
                                all_extracted_data.append(data)
                            finally:
                                os.unlink(tmp_path)
                                
                        except Exception as e:
                            st.error(f"Error processing file {file.name}: {e}")
                
                if all_extracted_data:
                    st.success("Data extracted successfully!")
                    
                    records = []
                    for idx, d in enumerate(all_extracted_data):
                        date_str = d.get("date", "UNKNOWN")
                        if date_str == "UNKNOWN":
                            date_str = f"2023-01-0{idx+1}"
                            
                        for param, val in d.get("parameters", {}).items():
                            try:
                                records.append({
                                    "Date": date_str,
                                    "Parameter": str(param),
                                    "Value": float(val)
                                })
                            except:
                                pass
                            
                    df = pd.DataFrame(records)
                    
                    if not df.empty:
                        try:
                            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                            df = df.sort_values("Date")
                            
                            st.markdown("### 📊 Parameter Trends")
                            
                            params = df['Parameter'].unique()
                            for param in params:
                                param_df = df[df['Parameter'] == param].dropna(subset=['Date'])
                                if not param_df.empty:
                                    plot_df = param_df.set_index("Date")[["Value"]]
                                    st.markdown(f"**{param}**")
                                    st.line_chart(plot_df, use_container_width=True)
                                
                            st.markdown("### 🤖 AI Trajectory Summary")
                            summary_prompt = f"You are a medical AI. The patient has uploaded {len(uploaded_trend_files)} {report_type_trend} reports over time. Here is the extracted data:\n{df.to_json(orient='records')}\nWrite a 3-4 sentence plain-language summary of how their health parameters are trending. Are they improving or worsening? What should they focus on? Keep it encouraging. Do not hallucinate."
                            
                            with st.spinner("Generating AI Trajectory Summary..."):
                                try:
                                    summary_resp = analyze_with_sarvam(summary_prompt)
                                    st.info(summary_resp["content"])
                                except Exception as e:
                                    try:
                                        summary_resp = analyze_with_groq(summary_prompt)
                                        st.info(summary_resp["content"])
                                    except:
                                        st.error("Failed to generate summary.")
                                    
                        except Exception as e:
                            st.error(f"Error plotting data: {e}.")

if __name__ == "__main__":
    main()