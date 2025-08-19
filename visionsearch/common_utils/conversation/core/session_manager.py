import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AnalysisType(Enum):
    GENERAL = "general"
    OBJECT_DETECTION = "object_detection"
    TEXT_EXTRACTION = "text_extraction"
    SCENE_UNDERSTANDING = "scene_understanding"

@dataclass
class DetectedObject:
    name: str
    confidence: float
    bbox: Optional[List[float]] = None

@dataclass
class ExtractedText:
    text: str
    confidence: float
    bbox: Optional[List[float]] = None

@dataclass
class VLMResponse:
    """Standardized response format for VLM operations"""
    success: bool
    response_text: str
    confidence: ConfidenceLevel
    analysis_type: AnalysisType
    detected_objects: Optional[List[DetectedObject]] = None
    extracted_text: Optional[List[ExtractedText]] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None
    model_info: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

@dataclass
class ConversationTurn:
    """Single conversation exchange"""
    question: str
    response: VLMResponse
    timestamp: str
    turn_number: int

@dataclass
class ChatSession:
    """Complete chat session with optional image context"""
    session_id: str
    image_data: Optional[str]  # base64 encoded image (None for text-only chat)
    image_metadata: Optional[Dict[str, Any]]
    conversation_history: List[ConversationTurn]
    created_at: str
    last_activity: str
    session_type: str  # "image_chat", "text_chat", or "mixed_chat"
    is_active: bool = True
    image_upload_turn: Optional[int] = None  # Which turn number the image was uploaded
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for storage/serialization"""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type,
            "image_metadata": self.image_metadata,
            "image_upload_turn": self.image_upload_turn,
            "conversation_history": [
                {
                    "question": turn.question,
                    "response": asdict(turn.response),
                    "timestamp": turn.timestamp,
                    "turn_number": turn.turn_number
                }
                for turn in self.conversation_history
            ],
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "is_active": self.is_active
        }

class SessionManager:
    """Manages chat sessions with VLM conversations"""
    
    def __init__(self, session_timeout_hours: int = 1, max_conversations_per_session: int = 20):
        self.sessions: Dict[str, ChatSession] = {}
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.max_conversations = max_conversations_per_session
    
    def create_session(self, image_data: Optional[str] = None, image_metadata: Optional[Dict] = None) -> str:
        """Create a new chat session (with or without image)"""
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        
        # Determine session type
        if image_data:
            session_type = "image_chat"
            if image_metadata is None:
                image_metadata = {
                    "upload_time": current_time,
                    "size_bytes": len(image_data)
                }
        else:
            session_type = "text_chat"
            image_metadata = None
        
        session = ChatSession(
            session_id=session_id,
            image_data=image_data,
            image_metadata=image_metadata,
            conversation_history=[],
            created_at=current_time,
            last_activity=current_time,
            session_type=session_type
        )
        
        self.sessions[session_id] = session
        print(f"Created new {session_type} session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a session by ID"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # Check if session has expired
        if not self._is_session_active(session):
            self._deactivate_session(session_id)
            return None
            
        return session
    
    def create_text_chat_session(self) -> str:
        """Create a text-only chat session (no image)"""
        return self.create_session(image_data=None)
    
    def create_image_chat_session(self, image_data: str, image_metadata: Optional[Dict] = None) -> str:
        """Create an image-based chat session"""
        return self.create_session(image_data=image_data, image_metadata=image_metadata)
    
    def add_image_to_session(self, session_id: str, image_data: str, image_metadata: Optional[Dict] = None) -> bool:
        """Add an image to an existing session (converts text_chat to mixed_chat)"""
        session = self.get_session(session_id)
        if not session:
            print(f"Session {session_id} not found")
            return False
        
        current_time = datetime.now().isoformat()
        
        # Prepare image metadata
        if image_metadata is None:
            image_metadata = {
                "upload_time": current_time,
                "size_bytes": len(image_data)
            }
        
        # Update session with image
        session.image_data = image_data
        session.image_metadata = image_metadata
        session.image_upload_turn = len(session.conversation_history) + 1
        session.last_activity = current_time
        
        # Update session type
        if session.session_type == "text_chat":
            session.session_type = "mixed_chat"
        elif session.session_type == "image_chat":
            # Replace existing image
            print(f"Replacing existing image in session {session_id}")
        
        print(f"Added image to session {session_id} at turn {session.image_upload_turn}")
        return True
    
    def remove_image_from_session(self, session_id: str) -> bool:
        """Remove image from session (converts back to text_chat if it was mixed_chat)"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.image_data = None
        session.image_metadata = None
        session.image_upload_turn = None
        session.last_activity = datetime.now().isoformat()
        
        # Update session type
        if session.session_type in ["mixed_chat", "image_chat"]:
            session.session_type = "text_chat"
        
        print(f"Removed image from session {session_id}")
        return True

    def add_conversation_turn(self, session_id: str, question: str, vlm_response: VLMResponse) -> bool:
        """Add a new conversation turn to the session"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Check conversation limit
        if len(session.conversation_history) >= self.max_conversations:
            print(f"Session {session_id} has reached maximum conversations ({self.max_conversations})")
            return False
        
        turn = ConversationTurn(
            question=question,
            response=vlm_response,
            timestamp=datetime.now().isoformat(),
            turn_number=len(session.conversation_history) + 1
        )
        
        session.conversation_history.append(turn)
        session.last_activity = datetime.now().isoformat()
        
        print(f"Added turn #{turn.turn_number} to session {session_id}")
        return True
    
    def get_conversation_history(self, session_id: str) -> List[ConversationTurn]:
        """Get the complete conversation history for a session"""
        session = self.get_session(session_id)
        return session.conversation_history if session else []
    
    def _is_session_active(self, session: ChatSession) -> bool:
        """Check if session is still within timeout window"""
        if not session.is_active:
            return False
            
        last_activity = datetime.fromisoformat(session.last_activity)
        return datetime.now() - last_activity < self.session_timeout
    
    def _deactivate_session(self, session_id: str) -> None:
        """Mark session as inactive"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            print(f"Deactivated expired session: {session_id}")
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count of cleaned sessions"""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not self._is_session_active(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            print(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict]:
        """Get statistics about a session"""
        session = self.get_session(session_id)
        if not session:
            return None
            
        return {
            "session_id": session_id,
            "session_type": session.session_type,
            "total_turns": len(session.conversation_history),
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "is_active": session.is_active,
            "time_since_creation": str(datetime.now() - datetime.fromisoformat(session.created_at)),
            "has_image": session.image_data is not None,
            "image_upload_turn": session.image_upload_turn,
            "image_size_bytes": session.image_metadata.get("size_bytes", 0) if session.image_metadata else 0
        }
    
    def list_active_sessions(self) -> List[str]:
        """Get list of all active session IDs"""
        active_sessions = []
        for session_id, session in self.sessions.items():
            if self._is_session_active(session):
                active_sessions.append(session_id)
        return active_sessions
    
    def list_sessions_by_type(self, session_type: str) -> List[str]:
        """Get list of sessions by type ('image_chat', 'text_chat', or 'mixed_chat')"""
        sessions_of_type = []
        for session_id, session in self.sessions.items():
            if self._is_session_active(session) and session.session_type == session_type:
                sessions_of_type.append(session_id)
        return sessions_of_type

# Test the Session Manager
if __name__ == "__main__":
    # Create session manager
    session_mgr = SessionManager()
    
    # Test 1: Create text session and add image later
    print("=== Test 1: Text Session → Add Image Mid-Chat ===")
    text_session_id = session_mgr.create_text_chat_session()
    print(f"Created text session: {text_session_id}")
    
    # Add some text conversation
    text_response = VLMResponse(
        success=True,
        response_text="Hello! I'm ready to help you with any questions.",
        confidence=ConfidenceLevel.HIGH,
        analysis_type=AnalysisType.GENERAL,
        processing_time_ms=100,
        model_info={"model": "gemma3", "version": "1.0"}
    )
    session_mgr.add_conversation_turn(text_session_id, "Hello, how are you?", text_response)
    
    print(f"Session type before image: {session_mgr.get_session(text_session_id).session_type}")
    
    # Now add an image mid-conversation
    fake_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    success = session_mgr.add_image_to_session(text_session_id, fake_image_data, {"filename": "uploaded.jpg"})
    print(f"Added image to session: {success}")
    print(f"Session type after image: {session_mgr.get_session(text_session_id).session_type}")
    
    # Add image-based conversation
    image_response = VLMResponse(
        success=True,
        response_text="Now I can see the image you uploaded. It appears to be a small test image.",
        confidence=ConfidenceLevel.HIGH,
        analysis_type=AnalysisType.GENERAL,
        processing_time_ms=150,
        model_info={"model": "gemma3", "version": "1.0"}
    )
    session_mgr.add_conversation_turn(text_session_id, "What do you see in this image?", image_response)
    
    # Test 2: Check context information
    print("\n=== Test 2: Context Information ===")
    context_info = session_mgr.get_session_context_info(text_session_id)
    print(f"Context info: {context_info}")
    
    # Test 3: Create image session from start
    print("\n=== Test 3: Image Session from Start ===")
    image_session_id = session_mgr.create_image_chat_session(fake_image_data, {"filename": "initial.jpg"})
    print(f"Created image session: {image_session_id}")
    
    initial_image_response = VLMResponse(
        success=True,
        response_text="I can see the image you provided from the beginning.",
        confidence=ConfidenceLevel.HIGH,
        analysis_type=AnalysisType.GENERAL,
        processing_time_ms=140,
        model_info={"model": "gemma3", "version": "1.0"}
    )
    session_mgr.add_conversation_turn(image_session_id, "Analyze this image", initial_image_response)
    
    image_context = session_mgr.get_session_context_info(image_session_id)
    print(f"Image session context: {image_context}")
    
    # Test 4: Session statistics
    print("\n=== Test 4: Session Statistics ===")
    mixed_stats = session_mgr.get_session_stats(text_session_id)
    image_stats = session_mgr.get_session_stats(image_session_id)
    
    print(f"Mixed session stats: {mixed_stats}")
    print(f"Image session stats: {image_stats}")
    
    # Test 5: List sessions by type
    print("\n=== Test 5: Sessions by Type ===")
    text_sessions = session_mgr.list_sessions_by_type("text_chat")
    image_sessions = session_mgr.list_sessions_by_type("image_chat")
    mixed_sessions = session_mgr.list_sessions_by_type("mixed_chat")
    
    print(f"Text sessions: {len(text_sessions)}")
    print(f"Image sessions: {len(image_sessions)}")
    print(f"Mixed sessions: {len(mixed_sessions)}")
    
    # Test 6: Remove image from mixed session
    print("\n=== Test 6: Remove Image from Mixed Session ===")
    print(f"Before removal - Session type: {session_mgr.get_session(text_session_id).session_type}")
    session_mgr.remove_image_from_session(text_session_id)
    print(f"After removal - Session type: {session_mgr.get_session(text_session_id).session_type}")
    
    print("\n=== Enhanced Session Manager with Mid-Chat Image Upload Tests Complete ===")