import json
import os
from collections import Counter
import re

class JsonPaser:
    def __init__(self):
        pass
    def extract_json_from_text(self,text):
        """
        Extract JSON content from text that may contain other content.
        
        Args:
            text (str): Text containing JSON content within markdown code blocks or other content
            
        Returns:
            dict or None: Parsed JSON object if found, None otherwise
        """
        # Pattern to match JSON content in markdown code blocks
        # Looks for:
        # 1. ```json ... ``` - JSON in markdown code blocks
        # 2. {... } - stand-alone JSON object
        # 3. json\s*{... } - text that starts with 'json' followed by JSON object
        json_pattern = r'```json\s*([\s\S]*?)\s*```|(json\s*\{[\s\S]*\})|(\{[\s\S]*\})'
        
        # Find all matches
        matches = re.findall(json_pattern, text)
        
        # Process matches
        for match in matches:
            # Check which capture group has content
            if match[0]:  # ```json ... ```
                json_str = match[0]
            elif match[1]:  # json{...}
                # Remove the "json" prefix and extract just the JSON part
                json_str = re.sub(r'^json\s*', '', match[1])
            else:  # {...}
                json_str = match[2]
            
            # Clean up the string
            json_str = json_str.strip()
            
            # Try to parse as JSON
            try:
                json_obj = json.loads(json_str)
                return json_obj
            except json.JSONDecodeError:
                continue
        
        return None
