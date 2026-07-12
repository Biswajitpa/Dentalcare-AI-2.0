<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0f0f,100:1a2a4a&height=220&section=header&text=DentalCare%20AI%202.0&fontSize=45&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=AI-Powered%20Dental%20Disease%20Analysis,%20Doctor%20Appointments%20and%20Patient%20Management&descAlignY=55&descAlign=50"/>
</p>

## 🦷 Dental Appointment Management System

A conversational AI-powered dental appointment management platform designed to streamline patient scheduling and clinic operations through natural language interactions. Built using LangGraph and Grok-4 (xAI), the system employs a multi-agent architecture in which specialized AI agents collaborate to manage appointment booking, rescheduling, cancellations, doctor availability checks, and patient inquiries.

## 🌟 Highlights

- **Multi-agent orchestration** — A supervisor agent classifies intent and routes conversations to specialized agents, rather than relying on one monolithic prompt.
- **Zero-friction booking** — Patients and staff book, reschedule, and cancel appointments using plain natural language, no forms required.
- **Conflict-safe scheduling** — Every booking and reschedule request is validated against real-time availability to prevent double-booking.
- **Lightweight, transparent data layer** — Appointments live in a simple CSV file, making the system easy to inspect, test, and extend without a database.
- **Educational by design** — Clear separation of agents, tools, and state makes this a practical reference implementation for learning LangGraph-based multi-agent systems.
- **Pluggable LLM backend** — Currently powered by Grok-4 (xAI), with an architecture that isolates model calls so alternate LLMs could be swapped in with minimal changes.

## ✨ Key Features

- 🤖 Multi-agent AI workflow using LangGraph
- 📅 Intelligent appointment booking and scheduling
- 🔄 Appointment rescheduling and cancellation support
- 👨‍⚕️ Doctor availability and department management
- 🗂️ CSV-based appointment database with reservation conflict detection
- 💬 Natural language conversation interface
- ⏰ Automated time-slot validation and booking confirmation
- 🏥 Department-wise doctor assignment and schedule management
- 📊 Real-time appointment tracking and clinic workflow support
- 🔒 Prevention of double-booking through schedule verification

## 📋 Overview

This system provides a chat-based interface for patients and clinic staff to:

- 📅 Check available appointment slots and doctor information
- 👨‍⚕️ View doctor schedules, departments, and availability
- 📝 Book new appointments with preferred doctors and time slots
- ❌ Cancel existing appointments quickly through conversational commands
- 🔄 Reschedule appointments to different dates or available time slots
- 🔍 Verify appointment details using appointment IDs
- ⏰ Prevent double-booking through real-time schedule validation
- 🏥 Assign appointments based on doctor specialization and department
- 📊 Manage clinic schedules using a structured CSV-based appointment database
- 💬 Interact naturally with AI agents without navigating complex forms

The system leverages a multi-agent architecture powered by LangGraph and Grok-4 (xAI), where specialized agents collaborate to handle appointment booking, schedule management, availability checks, cancellations, and patient support. By combining conversational AI with automated scheduling logic, the platform reduces administrative workload, improves appointment accuracy, and delivers a seamless digital experience for both patients and dental clinic staff.

The system uses a supervisor agent that intelligently routes user requests to the appropriate specialized agent based on the detected intent, making it an excellent educational example of multi-agent AI systems.

## Architecture

### Multi-Agent Design

The system follows a supervisor pattern, where a central coordinator analyzes user messages and routes them to the most appropriate specialized agent:

```
                    ┌──────────────┐
                    │   Supervisor │ ← Intent classification & routing
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐ ┌─────────────┐ ┌───────────────┐
   │ Info Agent  │ │   Booking   │ │  Cancellation │
   │             │ │    Agent    │ │    Agent      │
   └─────────────┘ └─────────────┘ └───────────────┘
          │
          ▼
   ┌───────────────┐
   │   Reschedule  │
   │    Agent      │
   └───────────────┘
```

### Agent Responsibilities

| Agent | Responsibility |
|-------|-----------------|
| **Supervisor** | Analyzes user input, classifies intent (`get_info`, `book`, `cancel`, `reschedule`, `end`), and routes to the appropriate agent |
| **Info Agent** | Handles queries about available slots, doctor schedules, and patient appointment lookups |
| **Booking Agent** | Collects booking details and creates new appointments |
| **Cancellation Agent** | Handles appointment cancellation requests |
| **Rescheduling Agent** | Manages moving appointments to different time slots |

### Technology Stack

| Component | Role |
|-----------|------|
| **LangGraph** | Orchestrates the multi-agent workflow and state management |
| **LangChain** | Provides the LLM integration and tool framework |
| **Grok-4 (xAI)** | Powers the conversational AI capabilities |
| **Pandas** | Manages the CSV-based data storage |
| **Pydantic** | Handles structured data validation |

## 🏗️ System Design

This section goes beyond the high-level architecture diagram above and describes how requests actually flow through the system, how state is shared between agents, and the design decisions behind them.

<p align="center">
  <img src="./dental_ai_system_architecture.svg" alt="DentalCare AI system architecture diagram" width="700"/>
</p>

### 1. End-to-End Request Flow

```
 User Input
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                        main.py (CLI)                         │
│  Reads user message → appends to conversation state          │
└───────────────────────────────┬───────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow (graph.py)              │
│                                                                │
│   ┌──────────────┐   intent    ┌──────────────────────────┐  │
│   │  Supervisor  │────────────▶│  Conditional Edge Router  │  │
│   │    Node      │             └──────────────┬────────────┘  │
│   └──────────────┘                             │               │
│                       ┌─────────────────────────┼───────────┐  │
│                       ▼             ▼            ▼           ▼  │
│                 ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌────┐│
│                 │   Info   │ │ Booking  │ │Cancellation│ │Res-││
│                 │  Agent   │ │  Agent   │ │   Agent    │ │ch. ││
│                 └────┬─────┘ └────┬─────┘ └─────┬──────┘ └─┬──┘│
│                      │            │             │          │   │
│                      ▼            ▼             ▼          ▼   │
│                 ┌──────────────────────────────────────────┐   │
│                 │        Tool Layer (csv_reader/writer)     │   │
│                 └────────────────────┬─────────────────────┘   │
└──────────────────────────────────────┼─────────────────────────┘
                                        ▼
                        doctor_availability.csv (data store)
                                        │
                                        ▼
                        Response formatted & returned to user
```

**Flow summary:**

1. The CLI in `main.py` captures raw user text and appends it to the shared LangGraph state as a new message.
2. The **Supervisor node** runs first on every turn. It performs structured-output intent classification (`get_info`, `book`, `cancel`, `reschedule`, `end`) and writes the decision back into state.
3. A **conditional edge** inspects `state.next_agent` and routes execution to exactly one specialized agent node — agents never call each other directly, which keeps the graph a strict tree instead of a mesh.
4. The selected agent may invoke one or more **tools** (read-only or mutating) to gather information or persist changes.
5. The agent formats a natural-language response, which is appended to the message history and returned to the CLI for display.
6. Control returns to the Supervisor for the next turn, so every message is independently re-classified rather than assuming the previous agent should keep handling the conversation.

### 2. State Design

LangGraph's shared state object is the backbone of the system. It is intentionally kept flat and serializable so it can later be persisted (e.g., to Redis or a database) with minimal changes:

| State Field | Type | Purpose |
|---|---|---|
| `messages` | `list[BaseMessage]` | Full conversation history, used for LLM context |
| `intent` | `str` | Last classified intent from the Supervisor |
| `next_agent` | `str` | Routing target used by the conditional edge |
| `reasoning` | `str` | Supervisor's explanation, useful for debugging/logging |
| `collected_params` | `dict` | In-progress booking/reschedule fields (patient ID, doctor, date/time, specialization) collected across multiple turns |
| `tool_results` | `list` | Raw outputs from the most recent tool calls |
| `final_response` | `str` | The message ultimately shown to the user |

Because `collected_params` persists across turns, agents can ask clarifying follow-up questions ("Which time slot works for you?") without losing previously supplied information — this is what allows multi-turn booking flows to feel conversational instead of form-like.

### 3. Design Principles

- **Single Responsibility per Agent** — Each agent owns exactly one capability (info, booking, cancellation, rescheduling). This keeps prompts small and focused, which improves classification and response accuracy compared to one large multi-purpose prompt.
- **Least-Privilege Tool Access** — The Info Agent only has read tools (`csv_reader.py`); only Booking, Cancellation, and Rescheduling agents can access write tools (`csv_writer.py`). This mirrors production-grade access control and prevents an info query from accidentally mutating data.
- **Validate Before Mutate** — Every write path (book/reschedule) first re-reads current availability before writing, closing the race condition where a slot could be booked twice between the time it was displayed and the time it was reserved.
- **Idempotent-Friendly Tools** — Tool functions accept explicit identifiers (patient ID + date/time) rather than relying on implicit "last mentioned" context, so a repeated call produces the same, predictable result instead of silently duplicating an action.
- **Separation of Orchestration and Data Access** — `workflows/graph.py` only knows about agents and routing; it has no knowledge of CSV structure. `tools/csv_reader.py` and `tools/csv_writer.py` are the only modules that touch `doctor_availability.csv`, so the storage format can change without touching the graph or agents.
- **Stateless Model Calls, Stateful Graph** — Each call to Grok-4 is stateless; all continuity comes from the LangGraph state passed in as context. This makes the system easy to reason about, test with fixed inputs, and scale horizontally.

### 4. Data Flow & Conflict Prevention

```
Booking Request
     │
     ▼
[1] Parse requested date/time, doctor, specialization
     │
     ▼
[2] csv_reader.check_availability(doctor, date_slot)
     │
     ├── is_available == False ──▶ Reject & suggest alternatives
     │
     ▼ is_available == True
[3] csv_writer.book_appointment(patient_id, doctor, date_slot)
     │      → sets is_available = False
     │      → sets patient_to_attend = patient_id
     ▼
[4] Persist to doctor_availability.csv
     ▼
[5] Return confirmation to user
```

The same check-then-write pattern is reused for rescheduling (release old slot → validate new slot → reserve new slot) and is what guarantees the "conflict-safe scheduling" and "no double-booking" guarantees described in the Highlights section.

### 5. Extensibility Points

The system was designed with a few clear seams for growth:

| Extension | Where to change |
|---|---|
| Swap the LLM provider | `dental_agent/config/settings.py` + the LLM client instantiation, since agents call a shared wrapper rather than the xAI SDK directly |
| Replace CSV with a real database | Only `tools/csv_reader.py` and `tools/csv_writer.py` need new implementations; agent and graph code is storage-agnostic |
| Add a new capability (e.g., billing) | Add a new agent module under `agents/`, register it as a graph node, and extend the Supervisor's intent enum |
| Add persistence for state across sessions | `models/state.py` is already a flat, serializable schema, making it straightforward to checkpoint to a store such as Redis or Postgres |
| Expose over an API instead of CLI | Replace `main.py` with a thin API layer (e.g., FastAPI) that feeds the same LangGraph workflow — no agent/tool changes required |

### 6. Known Limitations (by design, for an educational reference)

- CSV storage is not safe for concurrent writers; a production deployment should move to a transactional database.
- There is no authentication/authorization layer; anyone using the CLI can act as any patient ID.
- The Supervisor re-classifies intent every turn rather than maintaining a "current flow" concept, which is simple and robust but can occasionally re-ask for information mid-booking if a message is ambiguous.

## Project Structure

```
dental_agent_project/
├── main.py                          # Entry point - interactive CLI
├── doctor_availability.csv          # Data store for appointments
├── requirements.txt                 # Python dependencies
├── dental_agent/
│   ├── agent.py                     # Main agent definition & tools
│   ├── config/
│   │   └── settings.py              # Configuration & environment
│   ├── models/
│   │   └── state.py                 # State schema definitions
│   ├── tools/
│   │   ├── csv_reader.py            # Read operations (query tools)
│   │   └── csv_writer.py            # Write operations (mutation tools)
│   ├── agents/
│   │   ├── supervisor.py            # Intent classification & routing
│   │   ├── info_agent.py            # Information queries
│   │   ├── booking_agent.py         # Appointment booking
│   │   ├── cancellation_agent.py    # Appointment cancellation
│   │   └── rescheduling_agent.py    # Appointment rescheduling
│   └── workflows/
│       └── graph.py                 # LangGraph workflow definition
```

## Installation

### Prerequisites

- Python 3.10 or higher
- An xAI API key (Grok-4 model)

### Steps

1. **Clone the repository** and navigate to the project directory:
   ```bash
   cd dental_agent_project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:

   Create a `.env` file in the project root with your xAI API key:
   ```
   XAI_API_KEY=your_api_key_here
   MODEL_NAME=grok-4
   TEMPERATURE=0
   ```

   You can obtain an API key from the [xAI Console](https://console.x.ai/).

## Usage

### Running the System

Start the interactive appointment management system:

```bash
python main.py
```

### Example Interactions

**Check available slots:**
```
You: Show available slots for an orthodontist
Agent: Here are the available orthodontist appointments:
1. 5/10/2026 9:00 - Dr. Emily Johnson
2. 5/10/2026 10:00 - Dr. Emily Johnson
3. 5/12/2026 14:00 - Dr. Emily Johnson
...
```

**Book an appointment:**
```
You: Book patient 1000082 with Emily Johnson on 5/10/2026 9:00
Agent: Let me check that slot first... The slot is available!
I've booked the appointment:
- Patient ID: 1000082
- Doctor: Emily Johnson
- Date/Time: 5/10/2026 9:00
- Specialization: orthodontist
```

**Check patient appointments:**
```
You: What appointments does patient 1000048 have?
Agent: Patient 1000048 has the following appointments:
1. 5/8/2026 9:00 - Dr. John Doe (general_dentist)
```

**Cancel an appointment:**
```
You: Cancel appointment for patient 1000082 at 5/10/2026 9:00
Agent: I've cancelled the appointment for patient 1000082 on 5/10/2026 at 9:00.
```

**Reschedule an appointment:**
```
You: Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00
Agent: Let me verify the new slot is available... It's available!
I've rescheduled the appointment:
- Patient ID: 1000082
- New Date/Time: 5/12/2026 10:00
- Doctor: Emily Johnson
```

## Available Specializations

The system supports the following dental specializations:

- General Dentist
- Oral Surgeon
- Orthodontist
- Cosmetic Dentist
- Prosthodontist
- Pediatric Dentist
- Emergency Dentist

## Data Model

The appointment data is stored in `doctor_availability.csv` with the following structure:

| Field | Description |
|-------|-------------|
| `date_slot` | Appointment date and time (M/D/YYYY H:MM) |
| `specialization` | Type of dental specialist |
| `doctor_name` | Name of the dentist |
| `is_available` | Boolean indicating if the slot is open |
| `patient_to_attend` | Patient ID if booked, empty if available |

## For Students: How the Flow Works

Understanding this system helps demonstrate several key AI engineering concepts:

### 1. Intent Classification

When a user sends a message, the Supervisor agent analyzes the text to determine what action the user wants. This is done using structured output parsing, where the LLM returns a JSON object with:

- `intent`: The type of request (`get_info`, `book`, `cancel`, `reschedule`, `end`)
- `next_agent`: Which specialized agent should handle it
- `reasoning`: Explanation of the decision

### 2. Tool Use in Agents

Each specialized agent has access to specific tools. For example, the Info Agent can query available slots but cannot book appointments. This demonstrates the principle of least privilege in agent design.

### 3. State Management

LangGraph maintains conversation state across all agents. The state includes:

- Message history (for context)
- Current intent and routing decision
- Parameters collected during booking (patient ID, doctor, date)
- Tool execution results
- Final responses

### 4. Conditional Routing

The graph uses conditional edges to determine flow:

- **After supervisor**: Route based on classified intent
- **After agent**: Continue to tools if needed, or end if the response is complete

### 5. Data Layer Abstraction

Tools provide an abstraction layer over the CSV data, making it easy to:

- Change the data source (e.g., to a database)
- Add validation
- Modify query logic without changing agent code

## Configuration

Environment variables can be set in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | Your xAI API key | Required |
| `MODEL_NAME` | LLM model to use | `grok-4` |
| `TEMPERATURE` | LLM creativity (0 = deterministic) | `0` |

## 👤 Created & Maintained By

<p align="center">
  <img src="https://img.shields.io/badge/Author-Biswajit%20Pattanaik-0f0f0f?style=for-the-badge&logo=github&logoColor=white" alt="Author Badge"/>
</p>

<p align="center">
  <b>Biswajit Pattanaik</b><br/>
  <i>Building AI-powered agentic systems, one workflow at a time 🚀</i>
</p>

<p align="center">
  ⭐ If this project helped you, consider giving it a star — it goes a long way!<br/>
  🐛 Found a bug or have an idea? Issues and pull requests are always welcome.<br/>
  🤝 Open to feedback, collaboration, and discussion.
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a2a4a,100:0f0f0f&height=100&section=footer"/>
</p>
