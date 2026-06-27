import sqlite3
import asyncio
import os

DB_NAME = "spatial_memory.db"

def init_db():
    """Initializes the SQLite database and sets up the tracking schema."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS object_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            object_label TEXT NOT NULL,
            confidence REAL,
            bbox_x_center REAL,
            bbox_y_center REAL,
            gaze_state_at_detection TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("[Memory] Database initialized and schema verified.")

async def log_detected_object(label, confidence, x, y, system_state):
    """Asynchronously logs an object detection instance to the database."""
    # Run the blocking SQLite write operation inside a separate thread executor
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _db_write_worker, label, confidence, x, y, system_state)

def _db_write_worker(label, confidence, x, y, system_state):
    """Synchronous worker thread to handle the physical write command."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO object_logs (object_label, confidence, bbox_x_center, bbox_y_center, gaze_state_at_detection)
            VALUES (?, ?, ?, ?, ?)
        ''', (label, confidence, x, y, system_state))
        conn.commit()
        conn.close()
        print(f"[Memory Entry] Logged: '{label}' (Conf: {confidence:.2f}) at position ({x}, {y})")
    except Exception as e:
        print(f"[Memory Error] Failed to write to database: {e}")

# Quick standalone test to verify compilation
if __name__ == "__main__":
    init_db()
    # Simulate a quick test log
    async def test():
        print("Simulating test logs...")
        await log_detected_object("keys", 0.92, 0.45, 0.67, "ENGAGED")
        await log_detected_object("mug", 0.88, 0.12, 0.34, "IDLE_ATTENTION")
    
    asyncio.run(test())