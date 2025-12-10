"""
Utility functions for Azure AI dataset generation examples.
"""

from azure.identity import DefaultAzureCredential


def get_azure_openai_token_provider():
    """
    Get token provider for Azure OpenAI authentication.
    
    Returns:
        Callable that returns access token for Azure Cognitive Services.
    """
    credential = DefaultAzureCredential()
    
    def token_provider():
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token
    
    return token_provider
