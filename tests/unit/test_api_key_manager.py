"""
Unit tests for APIKeyManager component.

Tests key loading, validation, and setup guide display functionality.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from llm_judge_auditor.components.api_key_manager import APIKeyManager, APIKeyStatus


class TestAPIKeyManagerLoading:
    """Test API key loading from environment."""
    
    def test_load_keys_both_present(self):
        """Test loading when both API keys are present."""
        with patch.dict(os.environ, {
            'GROQ_API_KEY': 'test-groq-key',
            'GEMINI_API_KEY': 'test-gemini-key'
        }):
            manager = APIKeyManager()
            result = manager.load_keys()
            
            assert result['groq'] is True
            assert result['gemini'] is True
            assert manager.groq_key == 'test-groq-key'
            assert manager.gemini_key == 'test-gemini-key'
    
    def test_load_keys_groq_only(self):
        """Test loading when only Groq key is present."""
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-groq-key'}, clear=True):
            manager = APIKeyManager()
            result = manager.load_keys()
            
            assert result['groq'] is True
            assert result['gemini'] is False
            assert manager.groq_key == 'test-groq-key'
            assert manager.gemini_key is None
    
    def test_load_keys_gemini_only(self):
        """Test loading when only Gemini key is present."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-gemini-key'}, clear=True):
            manager = APIKeyManager()
            result = manager.load_keys()
            
            assert result['groq'] is False
            assert result['gemini'] is True
            assert manager.groq_key is None
            assert manager.gemini_key == 'test-gemini-key'
    
    def test_load_keys_none_present(self):
        """Test loading when no API keys are present."""
        with patch.dict(os.environ, {}, clear=True):
            manager = APIKeyManager()
            result = manager.load_keys()
            
            assert result['groq'] is False
            assert result['gemini'] is False
            assert manager.groq_key is None
            assert manager.gemini_key is None
    
    def test_load_keys_empty_strings(self):
        """Test loading when API keys are empty strings."""
        with patch.dict(os.environ, {
            'GROQ_API_KEY': '',
            'GEMINI_API_KEY': '   '
        }):
            manager = APIKeyManager()
            result = manager.load_keys()
            
            assert result['groq'] is False
            assert result['gemini'] is False


class TestAPIKeyManagerValidation:
    """Test API key validation functionality."""
    
    def test_validate_groq_key_success(self):
        """Test successful Groq key validation."""
        with patch('groq.Groq') as mock_groq_class:
            # Setup mock
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_client
            
            manager = APIKeyManager()
            manager.groq_key = 'test-groq-key'
            manager._key_status['groq'] = APIKeyStatus(service='groq', available=True)
            
            result = manager.validate_groq_key()
            
            assert result is True
            assert manager._key_status['groq'].validated is True
            assert manager._key_status['groq'].error_message is None
    
    def test_validate_groq_key_failure(self):
        """Test failed Groq key validation."""
        with patch('groq.Groq') as mock_groq_class:
            # Setup mock to raise exception
            mock_groq_class.side_effect = Exception("Invalid API key")
            
            manager = APIKeyManager()
            manager.groq_key = 'invalid-key'
            manager._key_status['groq'] = APIKeyStatus(service='groq', available=True)
            
            result = manager.validate_groq_key()
            
            assert result is False
            assert manager._key_status['groq'].validated is False
            assert manager._key_status['groq'].error_message is not None
    
    def test_validate_groq_key_no_key(self):
        """Test Groq validation with no key."""
        manager = APIKeyManager()
        manager.groq_key = None
        manager._key_status['groq'] = APIKeyStatus(service='groq', available=False)
        
        result = manager.validate_groq_key()
        
        assert result is False
        assert "No API key provided" in manager._key_status['groq'].error_message
    
    def test_validate_gemini_key_success(self):
        """Test successful Gemini key validation."""
        with patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            # Setup mock
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            manager = APIKeyManager()
            manager.gemini_key = 'test-gemini-key'
            manager._key_status['gemini'] = APIKeyStatus(service='gemini', available=True)
            
            result = manager.validate_gemini_key()
            
            assert result is True
            assert manager._key_status['gemini'].validated is True
            assert manager._key_status['gemini'].error_message is None
    
    def test_validate_gemini_key_failure(self):
        """Test failed Gemini key validation."""
        with patch('google.generativeai.configure') as mock_configure:
            # Setup mock to raise exception
            mock_configure.side_effect = Exception("Invalid API key")
            
            manager = APIKeyManager()
            manager.gemini_key = 'invalid-key'
            manager._key_status['gemini'] = APIKeyStatus(service='gemini', available=True)
            
            result = manager.validate_gemini_key()
            
            assert result is False
            assert manager._key_status['gemini'].validated is False
            assert manager._key_status['gemini'].error_message is not None
    
    def test_validate_gemini_key_no_key(self):
        """Test Gemini validation with no key."""
        manager = APIKeyManager()
        manager.gemini_key = None
        manager._key_status['gemini'] = APIKeyStatus(service='gemini', available=False)
        
        result = manager.validate_gemini_key()
        
        assert result is False
        assert "No API key provided" in manager._key_status['gemini'].error_message


class TestAPIKeyManagerUtilities:
    """Test utility methods."""
    
    def test_has_any_keys_both(self):
        """Test has_any_keys with both keys."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True),
            'gemini': APIKeyStatus(service='gemini', available=True)
        }
        
        assert manager.has_any_keys() is True
    
    def test_has_any_keys_one(self):
        """Test has_any_keys with one key."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        assert manager.has_any_keys() is True
    
    def test_has_any_keys_none(self):
        """Test has_any_keys with no keys."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=False),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        assert manager.has_any_keys() is False
    
    def test_get_available_services(self):
        """Test getting available services."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        services = manager.get_available_services()
        
        assert 'groq' in services
        assert 'gemini' not in services
        assert len(services) == 1
    
    def test_get_key_status(self):
        """Test getting key status."""
        manager = APIKeyManager()
        status = APIKeyStatus(service='groq', available=True, validated=True)
        manager._key_status['groq'] = status
        
        result = manager.get_key_status('groq')
        
        assert result == status
        assert result.service == 'groq'
        assert result.available is True
        assert result.validated is True
    
    def test_get_key_status_not_found(self):
        """Test getting status for non-existent service."""
        manager = APIKeyManager()
        
        result = manager.get_key_status('nonexistent')
        
        assert result is None
    
    def test_get_error_details(self):
        """Test getting error details."""
        manager = APIKeyManager()
        manager._key_status['groq'] = APIKeyStatus(
            service='groq',
            available=True,
            validated=False,
            error_message='Test error'
        )
        
        error = manager.get_error_details('groq')
        
        assert error == 'Test error'
    
    def test_get_error_details_no_error(self):
        """Test getting error details when no error."""
        manager = APIKeyManager()
        manager._key_status['groq'] = APIKeyStatus(
            service='groq',
            available=True,
            validated=True
        )
        
        error = manager.get_error_details('groq')
        
        assert error is None


class TestAPIKeyManagerSetupGuide:
    """Test setup guide generation."""
    
    def test_get_setup_instructions_no_keys(self):
        """Test setup instructions when no keys are present."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=False),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        instructions = manager.get_setup_instructions()
        
        assert '🔑 API Key Setup Required' in instructions
        assert 'https://console.groq.com' in instructions
        assert 'https://aistudio.google.com' in instructions
        assert 'export GROQ_API_KEY=' in instructions
        assert 'export GEMINI_API_KEY=' in instructions
        assert '❌ Groq key not found' in instructions
        assert '❌ Gemini key not found' in instructions
    
    def test_get_setup_instructions_with_keys(self):
        """Test setup instructions when keys are present."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True, validated=True),
            'gemini': APIKeyStatus(service='gemini', available=True, validated=True)
        }
        
        instructions = manager.get_setup_instructions(show_validation=True)
        
        assert '✅ Groq key: VALID' in instructions
        assert '✅ Gemini key: VALID' in instructions
    
    def test_get_setup_instructions_with_invalid_keys(self):
        """Test setup instructions when keys are invalid."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(
                service='groq',
                available=True,
                validated=False,
                error_message='Invalid API key'
            ),
            'gemini': APIKeyStatus(
                service='gemini',
                available=True,
                validated=False,
                error_message='Authentication failed'
            )
        }
        
        instructions = manager.get_setup_instructions(show_validation=True)
        
        assert '❌ Groq key: INVALID' in instructions
        assert '❌ Gemini key: INVALID' in instructions
        assert '🔧 Troubleshooting' in instructions
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True, validated=True),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        summary = manager.get_validation_summary()
        
        assert '🔍 API Key Validation Status' in summary
        assert '✅ Groq API Key: VALID' in summary
        assert '⚠️  Gemini API Key: NOT FOUND' in summary
    
    def test_get_troubleshooting_guide(self):
        """Test troubleshooting guide generation."""
        manager = APIKeyManager()
        
        guide = manager.get_troubleshooting_guide()
        
        assert '🔧 API Key Troubleshooting Guide' in guide
        assert 'Invalid API Key' in guide
        assert 'Rate Limit Exceeded' in guide
        assert 'Package Not Installed' in guide
        assert 'Network Error' in guide
        assert 'Environment Variables Not Set' in guide
        assert 'https://console.groq.com/docs' in guide
        assert 'https://ai.google.dev/docs' in guide


class TestAPIKeyManagerValidateAll:
    """Test validate_all_keys functionality."""
    
    @patch.object(APIKeyManager, 'validate_groq_key')
    @patch.object(APIKeyManager, 'validate_gemini_key')
    def test_validate_all_keys_both_available(
        self,
        mock_validate_gemini,
        mock_validate_groq
    ):
        """Test validating all keys when both are available."""
        mock_validate_groq.return_value = True
        mock_validate_gemini.return_value = True
        
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True),
            'gemini': APIKeyStatus(service='gemini', available=True)
        }
        
        results = manager.validate_all_keys(verbose=False)
        
        assert results['groq'] is True
        assert results['gemini'] is True
        assert mock_validate_groq.called
        assert mock_validate_gemini.called
    
    @patch.object(APIKeyManager, 'validate_groq_key')
    @patch.object(APIKeyManager, 'validate_gemini_key')
    def test_validate_all_keys_one_available(
        self,
        mock_validate_gemini,
        mock_validate_groq
    ):
        """Test validating all keys when only one is available."""
        mock_validate_groq.return_value = True
        
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=True),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        results = manager.validate_all_keys(verbose=False)
        
        assert results['groq'] is True
        assert results['gemini'] is False
        assert mock_validate_groq.called
        assert not mock_validate_gemini.called
    
    @patch.object(APIKeyManager, 'validate_groq_key')
    @patch.object(APIKeyManager, 'validate_gemini_key')
    def test_validate_all_keys_none_available(
        self,
        mock_validate_gemini,
        mock_validate_groq
    ):
        """Test validating all keys when none are available."""
        manager = APIKeyManager()
        manager._key_status = {
            'groq': APIKeyStatus(service='groq', available=False),
            'gemini': APIKeyStatus(service='gemini', available=False)
        }
        
        results = manager.validate_all_keys(verbose=False)
        
        assert results['groq'] is False
        assert results['gemini'] is False
        assert not mock_validate_groq.called
        assert not mock_validate_gemini.called
