"""
Interactive script to help users set up their social media API credentials.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def setup_platform_credentials(platform: str, required_fields: list) -> dict:
    """
    Helper function to collect platform-specific credentials.
    """
    print(f"\n📝 Setting up {platform.title()} credentials...")
    credentials = {}
    
    for field in required_fields:
        value = input(f"Enter your {field} (press Enter to skip): ").strip()
        if value:
            credentials[field] = value
    
    return credentials

def main():
    """Main setup function."""
    env_file = Path(".env")
    
    # Load existing environment variables
    load_dotenv()
    
    print("🔑 Social Media Credentials Setup")
    print("=================================")
    print("This script will help you set up your social media API credentials.")
    print("You can skip any platform by pressing Enter when prompted.")
    
    # Platform configurations
    platforms = {
        "twitter": [
            "TWITTER_API_KEY",
            "TWITTER_API_SECRET",
            "TWITTER_ACCESS_TOKEN",
            "TWITTER_ACCESS_TOKEN_SECRET"
        ],
        "facebook": [
            "FACEBOOK_APP_ID",
            "FACEBOOK_APP_SECRET",
            "FACEBOOK_ACCESS_TOKEN"
        ],
        "linkedin": [
            "LINKEDIN_CLIENT_ID",
            "LINKEDIN_CLIENT_SECRET",
            "LINKEDIN_ACCESS_TOKEN"
        ],
        "instagram": [
            "INSTAGRAM_APP_ID",
            "INSTAGRAM_APP_SECRET",
            "INSTAGRAM_ACCESS_TOKEN"
        ]
    }
    
    # Collect credentials for each platform
    for platform, fields in platforms.items():
        credentials = setup_platform_credentials(platform, fields)
        
        # Update .env file with new credentials
        if credentials:
            for field, value in credentials.items():
                set_key(env_file, field, value)
            print(f"✅ {platform.title()} credentials saved successfully!")
        else:
            print(f"⏩ Skipped {platform.title()} setup")
    
    print("\n🎉 Setup completed!")
    print("You can update these credentials at any time by:")
    print("1. Running this script again")
    print("2. Directly editing the .env file")
    print("\n⚠️ Note: Keep your API credentials secure and never share them!")

if __name__ == "__main__":
    main()
