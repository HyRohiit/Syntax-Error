# SynQ

SynQ is a local prototype for classroom doubt sharing. Students and teachers (registered as admins) join a shared live chat; the app groups similar doubts into topic cards so a teacher can see which subjects are drawing attention. A separate voice-notes view transcribes short microphone recordings and turns them into class notes.

> **Project status:** Local prototype, not production-ready. Accounts, sessions, chat messages, and clusters are held in server memory and disappear when the server restarts. Voice transcription and AI summaries use the Groq API.

## The problem

Questions in a class group chat are easy to miss. Similar questions appear at different times, useful context is mixed with general discussion, and teachers have to scan the entire conversation to see where students need help.

## The approach

SynQ provides one shared, real-time classroom chat and an automatically updated doubt dashboard. It uses sentence embeddings to group related messages, then asks a Groq chat model to name and summarize each group. The optional voice-notes view records a teacher or student, transcribes the recording in short chunks, and produces a summary with key points.

The clustering dashboard is a triage aid: it surfaces repeated topics and their relative volume. It does not verify answers or determine whether a doubt has been resolved.

## Features

### Implemented in the SynQ prototype (`synQ/`)

- Authenticated WebSocket classroom chat with sender identity and timestamps.
- Admin badges on messages sent by admin accounts.
- Topic clustering using `all-MiniLM-L6-v2` sentence embeddings and cosine distance.
- Groq-generated topic labels and short summaries for groups of doubts; updates are broadcast to connected clients.
- Live microphone recording in the browser, submitted as short audio chunks for English transcription with Groq Whisper.
- A separate class-notes view that summarizes the completed transcript and lists key points.
- A fallback that keeps chat available and separates messages into individual groups if the local embedding model cannot be loaded.

### Planned or not implemented in this repository

- Automatic spam or off-topic filtering in SynQ's shared chat. The current SynQ chat accepts validated text messages but does not classify or remove spam.
- Persistent database storage, so accounts and class data survive restarts.
- Multiple classrooms, teacher-managed membership, and classroom-level permissions.
- Deployment hardening and production operations, such as secure-cookie configuration for HTTPS, rate limits, account recovery, and managed secret storage.

## Architecture

SynQ is a small full-stack application served by FastAPI:

1. Chat messages are kept in memory. After enough new messages accumulate, a background worker embeds them with Sentence Transformers and groups them with scikit-learn's agglomerative clustering.
2. The backend sends each group to Groq for a concise topic and summary, then broadcasts the updated clusters to connected clients.
3. For voice notes, the browser records audio and sends chunks to the backend. The backend forwards each chunk to Groq speech-to-text; after recording ends, the browser sends the transcript for Groq summarization.

```text
Browser (served by FastAPI)
  ├── WebSocket: authenticated classroom messages and cluster updates
  └── HTTP: audio transcription and transcript summarization
              │
              ▼
        FastAPI backend
          ├── In-memory accounts, sessions, chat, clusters
          ├── Sentence Transformers + scikit-learn for doubt grouping
          └── Groq API for topic summaries, speech transcription, class notes
```

## Technology stack

| Area | Technology |
| --- | --- |
| User interface | HTML, CSS, browser JavaScript |
| Web server and API | Python, FastAPI, Uvicorn |
| Live chat | WebSockets |
| Doubt grouping | Sentence Transformers (`all-MiniLM-L6-v2`), scikit-learn |
| AI summaries and speech transcription | Groq API; configurable chat and speech-to-text models |
| Configuration | Environment variables and `python-dotenv` |

The standalone SynQ app does not require Node.js. The repository also contains a separate React + Vite chat-recap interface, which does use Node.js.

## Requirements and prerequisite packages

Before installing the Python packages, have these prerequisites ready:

- Windows 10/11 (the commands below use PowerShell); macOS and Linux can run the app with equivalent commands.
- Python 3.10 or newer, with `pip` available.
- Internet access for Groq requests and the first download of the sentence-embedding model.
- A current browser. Microphone access requires `localhost`, `127.0.0.1`, or HTTPS and user permission.
- A Groq API key for AI summaries, speech transcription, and generated class notes. The shared chat can be opened without one, but AI features need the key.

SynQ's Python dependencies are listed in `collegium-chat-app/requirements.txt` and installed together with `pip install -r requirements.txt`:

| Package | What it is used for |
| --- | --- |
| `fastapi` | Web server and HTTP API |
| `uvicorn[standard]` | Runs the FastAPI application |
| `requests` | Sends requests to Groq |
| `python-dotenv` | Loads settings from `.env` |
| `python-multipart` | Accepts uploaded audio chunks |
| `sentence-transformers` | Creates embeddings to group similar doubts |
| `scikit-learn` | Clusters the doubt embeddings |

No separate Node.js install is needed for SynQ. The standalone root React recap experiment does use Node.js, but it is not required to run `collegium-chat-app/`.

## Setup on Windows

Open PowerShell in the repository root, then run:

```powershell
Set-Location .\synQ
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

In `.env`, set your own Groq key and a private admin registration code:

Add this under Configuration and API key safety:
1. Sign in to the Groq Console.
2. Open API Keys, choose Create API Key, and copy the key.
3. Add it as GROQ_API_KEY in synQ/.env. Keep the file private and never commit the key to GitHub.

```dotenv
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_STT_MODEL=whisper-large-v3-turbo
ADMIN_REGISTRATION_CODE=choose-a-long-private-code
```

Save the file and start SynQ:

```powershell
python main.py
```

Open <http://127.0.0.1:8000> in your browser. Keep the terminal open while using the app. The embedding model may download the first time doubt clustering runs.

If PowerShell blocks virtual-environment activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window, then activate the environment again.


The admin code is shared by this local prototype; do not publish it or use it as a substitute for managed school access control.

## Configuration and API key safety

SynQ reads configuration from `synQ/.env` (or the process environment):

| Variable | Purpose | Default |
| --- | --- | --- |
| `GROQ_API_KEY` | Server-side credential for Groq requests | Empty; AI features require configuration |
| `GROQ_MODEL` | Groq chat-completion model for cluster descriptions and class notes | `openai/gpt-oss-120b` |
| `GROQ_STT_MODEL` | Groq speech-to-text model | `whisper-large-v3-turbo` |

Keep `.env` private. Do not put API keys in the HTML, JavaScript, screenshots, issue reports, or GitHub commits. The browser sends requests to the local backend; the backend attaches the key when calling Groq. Only submit audio or classroom messages you are permitted to share with that service. If a key has ever been committed or shared, revoke it and create a replacement.

## Using SynQ

1. Start the FastAPI app and visit `http://127.0.0.1:8000`.
2. Register as a student or admin, then sign in.
3. Send a question or doubt in the shared chat. Other connected users see new messages in real time.
4. As messages arrive, inspect the topic cards in the doubt dashboard. Grouping and summaries are approximate and may take time while the local embedding model initializes or the AI service responds.
5. To make class notes, open the voice-notes view, allow microphone access, start recording, and stop when finished. SynQ transcribes short chunks during recording and creates a summary with key points at the end.
6. Sign out when finished. Restarting the server clears all prototype accounts and conversation data.

## API and AI workflow

The SynQ backend exposes these routes:

| Route | Purpose |
| --- | --- |
| `GET /` | Serve the SynQ interface |
| `GET /health` | Report backend and model configuration status |
| `POST /api/transcribe` | Transcribe an uploaded audio chunk; requires an authenticated session |
| `POST /api/summarize-transcript` | Generate class notes from a transcript; requires an authenticated session |
| `WS /ws` | Authenticated live classroom chat and cluster updates |

The chat currently stores up to 5,000 messages in process memory. New doubts are analyzed in batches. When the embedding dependency is available, the backend groups semantically similar text using cosine distance, then requests a topic and summary from Groq. If embeddings fail to initialize, each message is left as its own group so chat can continue. Groq failures produce an unavailable-summary message.

Voice recording is a separate notes workflow: the browser records in 15-second intervals, sends audio chunks for English transcription, then submits the combined transcript for summary and 3–8 key points. Audio uploads are limited to 25 MB per chunk and transcript input to 120,000 characters.

## Repository layout

```text
.
├── collegium-chat-app/     # SynQ app: page, server, dependencies, and config example
│   ├── index(1).html       # User interface
│   ├── main.py             # Web server, accounts, chat, clustering, and AI routes
│   ├── requirements.txt
│   └── .env
└── README.md
```

SynQ does not have a separate frontend and backend directory: its interface and server code are together in `synQ/`. The `main.py` file still runs a backend server for login, shared chat, clustering, and AI requests. The other folders contain separate experiments and are not needed to run SynQ.
## Limitations

- **Ephemeral data:** SynQ's users, sessions, messages, and clusters are held in memory. A restart clears them; multiple server workers would not share state.
- **Shared admin invite code:** Admin registration relies on a single environment variable rather than institution-managed invitations or permissions.
- **External processing:** Audio chunks and transcripts sent for transcription or summarization are processed by Groq. Follow your school's privacy and consent requirements.
- **First-run model download:** Sentence Transformers downloads its embedding model on first use; without it, clustering falls back to separate groups.
- **No deployment setup:** This repository does not include production database migrations, HTTPS termination, deployment manifests, monitoring, or backup procedures.

## Roadmap

- Add persistent, classroom-scoped storage and teacher-managed classroom membership.
- Add configurable moderation and spam/off-topic flags, with teacher review before content is hidden.
- Improve cluster refresh behavior and let teachers merge, rename, or mark topics resolved.
- Connect speech-derived questions to the shared doubt dashboard as an explicit option, alongside the current class-notes workflow.
- Add privacy controls, retention settings, and production session/security configuration.
- Add deployment documentation and operational monitoring.

## Contributing

Contributions are welcome. Before opening a pull request:

1. Describe the classroom problem the change addresses.
2. Keep SynQ-specific changes inside `collegium-chat-app/` unless the change intentionally targets one of the separate prototypes.
3. Never commit API keys, `.env` files, real student messages, or identifying classroom data.
4. Document new configuration variables and API behavior in this README.
5. Include clear manual verification steps for changes to chat, authentication, clustering, or microphone behavior.

## License

No license has been selected yet. Add a license file before redistributing this project, and replace this placeholder with the chosen license and copyright details.
