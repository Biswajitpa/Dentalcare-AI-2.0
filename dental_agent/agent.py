import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from dental_agent.config.settings import MODEL_NAME, TEMPERATURE
from dental_agent.utils import sanitize_messages
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.tools.csv_writer import (
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
)

TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
    book_appointment,
    cancel_appointment,
    reschedule_appointment,
]

SYSTEM_PROMPT = """You are a helpful dental appointment assistant. You help patients with:
1. Checking available appointment slots and doctor information
2. Booking new appointments
3. Cancelling existing appointments
4. Rescheduling appointments

Always use M/D/YYYY H:MM format — e.g. 5/10/2026 9:00.
Always call check_slot_availability before booking to confirm the slot is free.
"""

def _pre_model_hook(state: dict) -> dict:
    sanitized = sanitize_messages(state["messages"])
    return {"llm_input_messages": [SystemMessage(content=SYSTEM_PROMPT)] + sanitized}

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)

dental_graph = create_react_agent(model=llm, tools=TOOLS, pre_model_hook=_pre_model_hook)