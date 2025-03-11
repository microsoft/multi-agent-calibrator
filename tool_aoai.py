from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI

async def call_aoai_withoutowndata_o1(messages: any, azureopenai_endpoint: str, model_deploy: str = "o1-mini-deploy", api_version: str = "2025-01-01-preview" ) -> str:

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    azure_oai_client = AsyncAzureOpenAI(
        api_version=api_version,
        azure_endpoint=azureopenai_endpoint,
        azure_ad_token_provider=token_provider
    )

    response = await azure_oai_client.chat.completions.create(
        model=model_deploy,
        messages=messages
    )

    response_content = response.choices[0].message.content
    return response_content

