from typing import Optional
from typing_extensions import TypedDict


class State(TypedDict):
    raw_conversation: str         
    user_id: Optional[str]        

    domain: Optional[str]         

    goal: Optional[str]           
    decisions: Optional[list[str]] 
    dead_ends: Optional[list[str]] 
    current_state: Optional[str]   
    next_steps: Optional[list[str]]

    briefing: Optional[str]        

    critic_score: Optional[int]   
    critic_feedback: Optional[str] 

    brain_id: Optional[str]       

    retry_count: int              
    error: Optional[str]           