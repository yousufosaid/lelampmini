import sqlite3
import ollama

DB_NAME = "spatial_memory.db"

def query_spatial_memory(object_label):
    """Searches the SQLite database for the last known position of an object."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Pull the most recent entry for the requested object label
        cursor.execute('''
            SELECT timestamp, bbox_x_center, bbox_y_center, gaze_state_at_detection 
            FROM object_logs 
            WHERE object_label LIKE ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (f"%{object_label}%",))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "found": True,
                "timestamp": row[0],
                "x": row[1],
                "y": row[2],
                "state": row[3]
            }
        return {"found": False}
    except Exception as e:
        print(f"[Brain Error] Database query failed: {e}")
        return {"found": False}

async def think_and_respond(user_prompt):
    """Passes user intent to Llama 3.2 and checks if a database lookup is required."""
    print(f"\n[Brain] Processing user request: '{user_prompt}'")
    
    # Define a clean system prompt to keep our 1B model focused
    system_instruction = (
        "You are LeLamp, a helpful robotic desk lamp. Your job is to tell the user "
        "where their items are based *only* on the database telemetry provided to you. "
        "Be extremely direct, short, and factual."
    )
    
    try:
        # Check if the user is asking to find an object
        lowercase_prompt = user_prompt.lower()
        target_object = None
        
        # Simple, fast keyword extraction tailored for small edge models
        for word in ["keys", "phone", "mug", "wallet", "glasses"]:
            if word in lowercase_prompt:
                target_object = word
                break
                
        if target_object:
            print(f"🔍 [Brain] Target detected ('{target_object}'). Querying spatial memory...")
            memory_result = query_spatial_memory(target_object)
            
            if memory_result["found"]:
                # Inject the real physical coordinates right into the context!
                context_prompt = (
                   f"DATA READOUT FROM TELEMETRY SYSTEM:\n"
                    f"- Target Object: {target_object}\n"
                    f"- Live Grid Position: X={memory_result['x']}, Y={memory_result['y']}\n"
                    f"- Timestamp: {memory_result['timestamp']}\n\n"
                    f"INSTRUCTION: The user just asked '{user_prompt}'. State the object's position "
                    f"clearly using the data above. Do not give warnings or apologize."
                )
            else:
                context_prompt = f"The user is asking: '{user_prompt}'. Tell them you haven't scanned that object in the room yet."
        else:
            context_prompt = user_prompt

        # Run local inference via Ollama
        response = ollama.generate(
            model='llama3.2:1b',
            system=system_instruction,
            prompt=context_prompt
        )
        
        return response['response']
        
    except Exception as e:
        return f"Brain connection hiccup: {e}"

# Standalone test block
if __name__ == "__main__":
    import asyncio
    
    async def test_brain():
        # Test 1: General conversation
        res1 = await think_and_respond("Who are you?")
        print(f" LeLamp: {res1}\n")
        
        # Test 2: Spatial memory recall lookup (pulls from our previous step's mock data!)
        res2 = await think_and_respond("Where did I leave my keys?")
        print(f" LeLamp: {res2}\n")
        
    asyncio.run(test_brain())