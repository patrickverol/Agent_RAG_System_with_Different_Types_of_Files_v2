#!/usr/bin/env python3
"""
Test script for Google GenAI image generation
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_google_genai():
    """Test Google GenAI image generation"""
    
    # Get API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables")
        return False
    
    try:
        # Import Google GenAI
        from google import genai
        from google.genai import types
        from PIL import Image
        from io import BytesIO
        import base64
        
        print("Testing Google GenAI image generation...")
        
        # Configure Google GenAI
        genai.configure(api_key=google_api_key)
        client = genai.Client()
        
        # Test prompt
        test_prompt = "Create a simple line chart showing stock prices: 30, 40, 50, 60, 70, 100. Make it professional and clean."
        
        print(f"Generating image with prompt: {test_prompt}")
        
        # Generate the image
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=test_prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # Check response
        if response.candidates and len(response.candidates) > 0:
            print("✓ Response received successfully")
            
            # Check for text and image parts
            has_text = False
            has_image = False
            
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"✓ Text response: {part.text[:100]}...")
                    has_text = True
                elif part.inline_data is not None:
                    print("✓ Image data received")
                    has_image = True
                    
                    # Test image processing
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        print(f"✓ Image opened successfully: {image.size} pixels")
                        
                        # Test base64 conversion
                        img_buffer = BytesIO()
                        image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        img_str = base64.b64encode(img_buffer.getvalue()).decode()
                        print(f"✓ Base64 conversion successful: {len(img_str)} characters")
                        
                    except Exception as e:
                        print(f"✗ Error processing image: {e}")
                        return False
            
            if has_text and has_image:
                print("✓ SUCCESS: Both text and image generated successfully!")
                return True
            else:
                print("✗ ERROR: Missing text or image in response")
                return False
        else:
            print("✗ ERROR: No candidates in response")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_google_genai()
    if success:
        print("\n🎉 Google GenAI test passed! The image generation should work in the main application.")
    else:
        print("\n❌ Google GenAI test failed! Check your API key and configuration.") 