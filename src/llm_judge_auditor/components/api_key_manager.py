"""
API Key Manager for managing and validating API keys for external judge services.

This component handles:
- Loading API keys from environment variables or config
- Validating API keys with lightweight test calls
- Generating setup instructions for obtaining free API keys
- Displaying formatted setup guides with validation status
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIKeyStatus:
    """Status of an API key."""
    
    service: str
    available: bool
    validated: bool = False
    error_message: Optional[str] = None


class APIKeyManager:
    """
    Manages API keys for external judge services (Groq, Gemini).
    
    Responsibilities:
    - Load API keys from environment variables
    - Validate API keys
    - Provide setup instructions when keys are missing
    """
    
    def __init__(self):
        """Initialize the API key manager."""
        self.groq_key: Optional[str] = None
        self.gemini_key: Optional[str] = None
        self._key_status: Dict[str, APIKeyStatus] = {}
    
    def load_keys(self) -> Dict[str, bool]:
        """
        Load API keys from environment variables.
        
        Checks for:
        - GROQ_API_KEY
        - GEMINI_API_KEY
        
        Returns:
            Dict mapping service name to whether key is available
        """
        # Load Groq API key
        self.groq_key = os.environ.get("GROQ_API_KEY")
        groq_available = self.groq_key is not None and len(self.groq_key.strip()) > 0
        self._key_status["groq"] = APIKeyStatus(
            service="groq",
            available=groq_available
        )
        
        # Load Gemini API key
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        gemini_available = self.gemini_key is not None and len(self.gemini_key.strip()) > 0
        self._key_status["gemini"] = APIKeyStatus(
            service="gemini",
            available=gemini_available
        )
        
        if groq_available:
            logger.info("Groq API key loaded from environment")
        else:
            logger.warning("Groq API key not found in environment (GROQ_API_KEY)")
        
        if gemini_available:
            logger.info("Gemini API key loaded from environment")
        else:
            logger.warning("Gemini API key not found in environment (GEMINI_API_KEY)")
        
        return {
            "groq": groq_available,
            "gemini": gemini_available
        }
    
    def validate_groq_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validate Groq API key with a lightweight test call.
        
        Args:
            api_key: Optional API key to validate. If None, uses loaded key.
        
        Returns:
            True if key is valid, False otherwise
        """
        key_to_validate = api_key or self.groq_key
        
        if not key_to_validate:
            self._key_status["groq"].error_message = "No API key provided"
            return False
        
        try:
            # Import here to avoid requiring groq if not used
            from groq import Groq
            
            client = Groq(api_key=key_to_validate)
            
            # Make a minimal test call
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            self._key_status["groq"].validated = True
            self._key_status["groq"].error_message = None
            logger.info("Groq API key validated successfully")
            return True
            
        except ImportError:
            error_msg = "groq package not installed. Run: pip install groq"
            self._key_status["groq"].error_message = error_msg
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Groq API key validation failed: {str(e)}"
            self._key_status["groq"].error_message = error_msg
            logger.error(error_msg)
            return False
    
    def validate_gemini_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validate Gemini API key with a lightweight test call.
        
        Args:
            api_key: Optional API key to validate. If None, uses loaded key.
        
        Returns:
            True if key is valid, False otherwise
        """
        key_to_validate = api_key or self.gemini_key
        
        if not key_to_validate:
            self._key_status["gemini"].error_message = "No API key provided"
            return False
        
        try:
            # Import here to avoid requiring google-generativeai if not used
            import google.generativeai as genai
            
            genai.configure(api_key=key_to_validate)
            model = genai.GenerativeModel("gemini-2.0-flash-exp")
            
            # Make a minimal test call
            response = model.generate_content(
                "test",
                generation_config=genai.types.GenerationConfig(max_output_tokens=1)
            )
            
            self._key_status["gemini"].validated = True
            self._key_status["gemini"].error_message = None
            logger.info("Gemini API key validated successfully")
            return True
            
        except ImportError:
            error_msg = "google-generativeai package not installed. Run: pip install google-generativeai"
            self._key_status["gemini"].error_message = error_msg
            logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Gemini API key validation failed: {str(e)}"
            self._key_status["gemini"].error_message = error_msg
            logger.error(error_msg)
            return False
    
    def has_any_keys(self) -> bool:
        """
        Check if any API keys are available.
        
        Returns:
            True if at least one API key is available
        """
        return any(status.available for status in self._key_status.values())
    
    def get_available_services(self) -> list[str]:
        """
        Get list of services with available API keys.
        
        Returns:
            List of service names with available keys
        """
        return [
            service 
            for service, status in self._key_status.items() 
            if status.available
        ]
    
    def get_key_status(self, service: str) -> Optional[APIKeyStatus]:
        """
        Get status of a specific API key.
        
        Args:
            service: Service name ("groq" or "gemini")
        
        Returns:
            APIKeyStatus object or None if service not found
        """
        return self._key_status.get(service)
    
    def validate_all_keys(self, verbose: bool = True) -> Dict[str, bool]:
        """
        Validate all available API keys with lightweight test calls.
        
        Args:
            verbose: If True, log validation progress
        
        Returns:
            Dict mapping service name to validation success
        """
        results = {}
        
        # Validate Groq key if available
        if self._key_status.get("groq") and self._key_status["groq"].available:
            if verbose:
                logger.info("Validating Groq API key...")
            results["groq"] = self.validate_groq_key()
        else:
            results["groq"] = False
        
        # Validate Gemini key if available
        if self._key_status.get("gemini") and self._key_status["gemini"].available:
            if verbose:
                logger.info("Validating Gemini API key...")
            results["gemini"] = self.validate_gemini_key()
        else:
            results["gemini"] = False
        
        return results
    
    def get_validation_summary(self) -> str:
        """
        Get a formatted summary of API key validation status.
        
        Returns:
            Formatted string showing validation status for each service
        """
        summary = "\n"
        summary += "╔══════════════════════════════════════════════════════════════╗\n"
        summary += "║  🔍 API Key Validation Status                                ║\n"
        summary += "╠══════════════════════════════════════════════════════════════╣\n"
        summary += "║                                                              ║\n"
        
        # Groq status
        groq_status = self._key_status.get("groq")
        if groq_status:
            if groq_status.validated:
                summary += "║  ✅ Groq API Key: VALID                                      ║\n"
            elif groq_status.available:
                summary += "║  ❌ Groq API Key: INVALID                                    ║\n"
                if groq_status.error_message:
                    # Truncate error message to fit in box
                    error = groq_status.error_message[:50]
                    summary += f"║     Error: {error:<50} ║\n"
            else:
                summary += "║  ⚠️  Groq API Key: NOT FOUND                                 ║\n"
        
        summary += "║                                                              ║\n"
        
        # Gemini status
        gemini_status = self._key_status.get("gemini")
        if gemini_status:
            if gemini_status.validated:
                summary += "║  ✅ Gemini API Key: VALID                                    ║\n"
            elif gemini_status.available:
                summary += "║  ❌ Gemini API Key: INVALID                                  ║\n"
                if gemini_status.error_message:
                    # Truncate error message to fit in box
                    error = gemini_status.error_message[:50]
                    summary += f"║     Error: {error:<50} ║\n"
            else:
                summary += "║  ⚠️  Gemini API Key: NOT FOUND                               ║\n"
        
        summary += "║                                                              ║\n"
        summary += "╚══════════════════════════════════════════════════════════════╝\n"
        
        return summary
    
    def get_setup_instructions(self, show_validation: bool = False) -> str:
        """
        Generate setup instructions for obtaining free API keys.
        
        Args:
            show_validation: If True, include validation status in output
        
        Returns:
            Formatted string with setup instructions
        """
        instructions = "\n"
        instructions += "╔══════════════════════════════════════════════════════════════╗\n"
        instructions += "║  🔑 API Key Setup Required                                   ║\n"
        instructions += "╠══════════════════════════════════════════════════════════════╣\n"
        instructions += "║                                                              ║\n"
        instructions += "║  This system uses FREE API-based judges for evaluation:     ║\n"
        instructions += "║                                                              ║\n"
        instructions += "║  1️⃣  Groq Llama 3.3 70B (FREE)                              ║\n"
        instructions += "║     • Sign up: https://console.groq.com                     ║\n"
        instructions += "║     • Get API key: https://console.groq.com/keys            ║\n"
        instructions += "║     • Set: export GROQ_API_KEY=\"your-key\"                   ║\n"
        
        groq_status = self._key_status.get("groq")
        if groq_status:
            if show_validation and groq_status.validated:
                instructions += "║     ✅ Groq key: VALID                                       ║\n"
            elif groq_status.available and show_validation and not groq_status.validated:
                instructions += "║     ❌ Groq key: INVALID                                     ║\n"
                if groq_status.error_message:
                    # Show truncated error
                    error = groq_status.error_message[:50]
                    instructions += f"║        {error:<54} ║\n"
            elif groq_status.available:
                instructions += "║     ⚠️  Groq key detected (not validated)                    ║\n"
            else:
                instructions += "║     ❌ Groq key not found                                    ║\n"
        
        instructions += "║                                                              ║\n"
        instructions += "║  2️⃣  Google Gemini Flash (FREE)                             ║\n"
        instructions += "║     • Sign up: https://aistudio.google.com                  ║\n"
        instructions += "║     • Get API key: https://aistudio.google.com/app/apikey   ║\n"
        instructions += "║     • Set: export GEMINI_API_KEY=\"your-key\"                 ║\n"
        
        gemini_status = self._key_status.get("gemini")
        if gemini_status:
            if show_validation and gemini_status.validated:
                instructions += "║     ✅ Gemini key: VALID                                     ║\n"
            elif gemini_status.available and show_validation and not gemini_status.validated:
                instructions += "║     ❌ Gemini key: INVALID                                   ║\n"
                if gemini_status.error_message:
                    # Show truncated error
                    error = gemini_status.error_message[:50]
                    instructions += f"║        {error:<54} ║\n"
            elif gemini_status.available:
                instructions += "║     ⚠️  Gemini key detected (not validated)                  ║\n"
            else:
                instructions += "║     ❌ Gemini key not found                                  ║\n"
        
        instructions += "║                                                              ║\n"
        instructions += "║  💡 Both APIs are completely FREE!                          ║\n"
        instructions += "║  💡 You need at least ONE key to use API judges             ║\n"
        instructions += "║  💡 Using BOTH keys gives better evaluation accuracy        ║\n"
        instructions += "║                                                              ║\n"
        
        # Add troubleshooting section if there are errors
        if show_validation and any(
            status.available and not status.validated 
            for status in self._key_status.values()
        ):
            instructions += "╠══════════════════════════════════════════════════════════════╣\n"
            instructions += "║  🔧 Troubleshooting                                          ║\n"
            instructions += "║                                                              ║\n"
            instructions += "║  If validation fails:                                       ║\n"
            instructions += "║  • Check that your API key is correct                       ║\n"
            instructions += "║  • Ensure you have internet connectivity                    ║\n"
            instructions += "║  • Verify the API service is not experiencing issues        ║\n"
            instructions += "║  • Check that required packages are installed:              ║\n"
            instructions += "║    - pip install groq google-generativeai                   ║\n"
            instructions += "║                                                              ║\n"
        
        instructions += "╚══════════════════════════════════════════════════════════════╝\n"
        
        return instructions
    
    def display_setup_guide_with_validation(self, validate: bool = True) -> None:
        """
        Display setup guide and optionally validate available keys.
        
        This is a convenience method that:
        1. Loads API keys from environment
        2. Optionally validates them with test calls
        3. Displays a formatted setup guide with status
        
        Args:
            validate: If True, validate keys before displaying guide
        """
        # Load keys
        self.load_keys()
        
        # Validate if requested
        if validate and self.has_any_keys():
            logger.info("Validating API keys...")
            self.validate_all_keys(verbose=True)
        
        # Display setup guide
        print(self.get_setup_instructions(show_validation=validate))
        
        # If validation was performed, show summary
        if validate and self.has_any_keys():
            print(self.get_validation_summary())
    
    def get_error_details(self, service: str) -> Optional[str]:
        """
        Get detailed error message for a specific service.
        
        Args:
            service: Service name ("groq" or "gemini")
        
        Returns:
            Error message or None if no error
        """
        status = self._key_status.get(service)
        if status and status.error_message:
            return status.error_message
        return None
    
    def get_troubleshooting_guide(self) -> str:
        """
        Get a detailed troubleshooting guide for API key issues.
        
        Returns:
            Formatted troubleshooting guide
        """
        guide = "\n"
        guide += "╔══════════════════════════════════════════════════════════════╗\n"
        guide += "║  🔧 API Key Troubleshooting Guide                            ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  Common Issues and Solutions:                               ║\n"
        guide += "║                                                              ║\n"
        guide += "║  1. \"Invalid API Key\" or \"Authentication Failed\"            ║\n"
        guide += "║     • Double-check your API key is copied correctly         ║\n"
        guide += "║     • Ensure no extra spaces or quotes in the key           ║\n"
        guide += "║     • Verify the key is active in your account              ║\n"
        guide += "║                                                              ║\n"
        guide += "║  2. \"Rate Limit Exceeded\"                                   ║\n"
        guide += "║     • Wait a few minutes before trying again                ║\n"
        guide += "║     • Free tier limits: Groq (30/min), Gemini (15/min)     ║\n"
        guide += "║                                                              ║\n"
        guide += "║  3. \"Package Not Installed\"                                 ║\n"
        guide += "║     • Run: pip install groq google-generativeai            ║\n"
        guide += "║     • Ensure you're in the correct virtual environment      ║\n"
        guide += "║                                                              ║\n"
        guide += "║  4. \"Network Error\" or \"Connection Timeout\"                ║\n"
        guide += "║     • Check your internet connection                        ║\n"
        guide += "║     • Try again in a few moments                            ║\n"
        guide += "║     • Check if the API service is experiencing issues       ║\n"
        guide += "║                                                              ║\n"
        guide += "║  5. Environment Variables Not Set                           ║\n"
        guide += "║     • Make sure to export the variables:                    ║\n"
        guide += "║       export GROQ_API_KEY=\"your-key\"                        ║\n"
        guide += "║       export GEMINI_API_KEY=\"your-key\"                      ║\n"
        guide += "║     • Restart your terminal after setting variables         ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Need More Help?                                            ║\n"
        guide += "║     • Groq Docs: https://console.groq.com/docs              ║\n"
        guide += "║     • Gemini Docs: https://ai.google.dev/docs               ║\n"
        guide += "║                                                              ║\n"
        guide += "╚══════════════════════════════════════════════════════════════╝\n"
        
        return guide
