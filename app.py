from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional, List
from jose import JWTError, jwt
from passlib.context import CryptContext
import sqlite3
import spacy
from huggingface_hub import InferenceClient
from fastapi.middleware.cors import CORSMiddleware

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origin, Currently set to all origins (Not recommended for Production)
    allow_credentials=True,  # Allow sending cookies and credentials
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (e.g., Authorization)
)

HUGGINGFACE_API_KEY = "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"
HUGGINGFACE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Secret key for JWT
SECRET_KEY = "d9d41e7391af3dc0868618f136b94f7d"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database setup
DATABASE = "app.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            prompt TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    topic: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(email: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user or not verify_password(password, user[3]):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        user = get_user(email)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def resolve_pronouns(user_prompt: str, latest_response: str) -> str:
    doc_prompt = nlp(user_prompt)
    doc_response = nlp(latest_response)
    entities = {ent.label_: ent.text for ent in doc_response.ents}
    noun_chunks = [chunk.text for chunk in doc_response.noun_chunks]

    pronoun_mapping = {
        "he": entities.get("PERSON", ""),
        "she": entities.get("PERSON", ""),
        "it": _find_relevant_noun("it", noun_chunks),
        "they": entities.get("ORG", entities.get("GPE", "")),
        "his": entities.get("PERSON", "") + "'s",
        "her": entities.get("PERSON", "") + "'s",
        "their": entities.get("ORG", "") + "'s",
    }

    resolved_prompt = " ".join(
        pronoun_mapping.get(token.lower_, token.text) for token in doc_prompt
    )

    if not any(token.lower_ in pronoun_mapping for token in doc_prompt):
        topic = _find_relevant_noun("it", noun_chunks)
        if topic:
            resolved_prompt += f" (referring to {topic})"

    return resolved_prompt

def _find_relevant_noun(pronoun: str, noun_chunks: List[str]) -> Optional[str]:
    for chunk in noun_chunks:
        if pronoun.lower() in chunk.lower():
            return chunk
    return noun_chunks[0] if noun_chunks else None

def detect_topic(response: str) -> str:
    doc = nlp(response)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT"}]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    return entities[0] if entities else (noun_chunks[0] if noun_chunks else "General")

def call_huggingface_api(prompt: str) -> str:
    client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
    messages = [
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=HUGGINGFACE_MODEL,
        messages=messages,
        max_tokens=1000
    )
    return completion.choices[0].message

# Endpoints
@app.post("/signup", response_model=Token)
def signup(user: UserCreate):
    hashed_password = get_password_hash(user.password)
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (user.username, user.email, hashed_password))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user[2]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT response FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1
    """, (current_user[0],))
    latest_chat = cursor.fetchone()
    context = latest_chat[0] if latest_chat else ""

    resolved_prompt = resolve_pronouns(request.prompt, context)
    response = call_huggingface_api(resolved_prompt)
    topic = detect_topic(response)

    cursor.execute("INSERT INTO chat_history (user_id, prompt, response) VALUES (?, ?, ?)",
                   (current_user[0], request.prompt, response))
    conn.commit()
    conn.close()

    return {"response": response, "topic": topic}

@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cutoff_date = datetime.utcnow() - timedelta(days=4)
    cursor.execute("""
        SELECT id, prompt, timestamp FROM chat_history
        WHERE user_id = ? AND timestamp >= ? ORDER BY timestamp DESC
    """, (current_user[0], cutoff_date))
    history = cursor.fetchall()
    conn.close()
    return [
        {"id": chat[0], "prompt": chat[1][:50], "timestamp": chat[2]} for chat in history
    ]

@app.get("/history/{chat_id}")
def get_full_history(chat_id: int, current_user: dict = Depends(get_current_user)):
    conn = sqlite3.connect(DATABASE)
