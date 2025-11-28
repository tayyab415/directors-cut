"""
Director: The "Logic" component.
Uses DeepSeek-V3 to write the editing script based on verified hotspots.
"""

import os
import google.generativeai as genai
import json
import re
from typing import List, Dict, Any

class Director:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash-lite-preview-02-05" # Latest Flash Lite
        
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not found. Director will fail.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def create_edit_plan(self, hotspots: List[Dict], video_info: Dict) -> List[Dict]:
        """
        Generate an editing plan.
        
        Args:
            hotspots: List of verified hotspots.
            video_info: Metadata about the video.
            
        Returns:
            List of edit instructions (JSON).
        """
        print("  - Director is planning the cut...")
        
        prompt = f"""
        You are a professional video editor creating viral short-form content.
        
        Video: "{video_info.get('title', 'Unknown')}" (Duration: {video_info.get('duration', 0)}s)
        
        Available Hotspots (pre-downloaded clips):
        {json.dumps(hotspots, indent=2)}
        
        **CRITICAL EDITING RULES:**
        
        1. **Clip Duration**: Each clip MUST be 8-15 seconds long (MINIMUM 8 seconds!)
        
        2. **Timestamp Constraints**: Your start/end times MUST stay within the hotspot boundaries!
           - For each hotspot, the downloaded clip spans: [start-2s, end+2s]
           - Your edit times must fit within this range
           - Example: If hotspot is 100-110s, you can use 98-112s range
        
        3. **Selection**: Pick 3-5 of the BEST hotspots based on:
           - High scores (>0.8 are excellent)
           - Semantic triggers (these are gold!)
           - Variety across the video timeline
        
        4. **Flow**: Arrange clips chronologically unless reordering tells a better story
        
        5. **Target**: 30-60 seconds total duration
        
        **OUTPUT FORMAT:**
        Return ONLY a valid JSON array (no markdown, no explanations):
        
        [
            {{"start": <float>, "end": <float>, "description": "<why include>"}}
        ]
        
        Example:
        [
            {{"start": 100.0, "end": 112.0, "description": "Controversial opinion about AI"}},
            {{"start": 455.5, "end": 468.0, "description": "Personal story that went viral"}}
        ]
        """
        
        if not self.api_key:
            print("  ! Director failed: GEMINI_API_KEY not set")
            return []

        try:
            response = self.model.generate_content(prompt)
            content = response.text
            
            print(f"  [DEBUG] Raw Director Output:\n{content}\n")
            
            # Clean up JSON
            content = content.replace("```json", "").replace("```", "").strip()
            
            plan = json.loads(content)
            return plan
            
        except Exception as e:
            print(f"  ! Director failed: {e}")
            return []

if __name__ == "__main__":
    d = Director()
    print("Director initialized.")
