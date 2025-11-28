"""
Verifier: The "Vision AI" component.
Extracts frames from hotspots and uses Qwen2.5-VL (via OpenRouter) to verify them.
"""

import os
import cv2
import google.generativeai as genai
from typing import Dict, Any, Optional
import PIL.Image

from src.paths import ASSETS_TEMP_DIR, ensure_runtime_dirs

class Verifier:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-2.0-flash-lite-preview-02-05"
        
        if not self.api_key:
            print("⚠️ GEMINI_API_KEY not found. Verifier will fail.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def verify(self, video_path: str, timestamp: float) -> Dict[str, Any]:
        """
        Verify if a specific moment in the video is interesting.
        
        Args:
            video_path: Path to video file.
            timestamp: Time in seconds to extract frame from.
            
        Returns:
            Dict with verification result (is_interesting, score, reason).
        """
        print(f"  - Verifying frame at {timestamp}s...")
        
        # 1. Extract Frame
        frame_path = self._extract_frame(video_path, timestamp)
        if not frame_path:
            return {"error": "Failed to extract frame"}

        # 2. Load Image (PIL)
        # Gemini prefers PIL images or file paths for upload, but PIL is easy here
        try:
            image = PIL.Image.open(frame_path)
        except Exception as e:
             return {"error": f"Failed to load image: {e}"}
        
        # 3. Call Vision AI
        if not self.api_key:
            return {"error": "GEMINI_API_KEY not set"}

        try:
            prompt = """
            Analyze this video frame. Is it visually interesting or important?
            Is it a slide, a demo, a person talking, or just noise?
            Rate it 1-10 on 'Viral Potential'.
            Return JSON: { "score": int, "reason": str, "type": str }
            """
            
            response = self.model.generate_content([prompt, image])
            content = response.text
            
            # Clean up code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Simple parsing
            import json
            try:
                result = json.loads(content)
                return result
            except:
                return {"raw_response": content, "score": 0}

        except Exception as e:
            print(f"  ! Vision AI failed: {e}")
            return {"error": str(e)}
        finally:
            # Cleanup
            if os.path.exists(frame_path):
                os.remove(frame_path)

    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[str]:
        """Extracts a single frame and saves it to temp."""
        try:
            ensure_runtime_dirs()
            cap = cv2.VideoCapture(video_path)
            # Set position
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, image = cap.read()
            if success:
                temp_path = os.path.join(ASSETS_TEMP_DIR, f"frame_{timestamp}.jpg")
                cv2.imwrite(temp_path, image)
                cap.release()
                return temp_path
            cap.release()
            return None
        except Exception as e:
            print(f"  ! Frame extraction failed: {e}")
            return None

    # Helper _encode_image removed as Gemini uses PIL directly

if __name__ == "__main__":
    # Test stub
    v = Verifier()
    print("Verifier initialized.")
