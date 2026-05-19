import pytest
from httpx import ASGITransport, AsyncClient
from auto_gateway.core.server import create_app
from auto_gateway.core.router import ProviderRouter
from auto_gateway.strategies.sequential import SequentialStrategy
from auto_gateway.providers.base import BaseProvider, ProviderCallResult

# 1. Define a minimal Mock Provider
class SimpleMockProvider(BaseProvider):
    async def call(self, *, key, model, messages, timeout, tools, tool_choice, extra_body=None) -> ProviderCallResult:
        return {
            "text": "Hello! I am the mock provider.",
            "reasoning": None,
            "tool_calls": None,
            "usage": {"prompt_tokens": 1, "completion_tokens": 5, "total_tokens": 6}
        }

# 2. Setup the application fixture
@pytest.fixture
def test_app():
    # Initialize your components
    providers = {"mock": SimpleMockProvider("mock", [], {"gpt-4": []})}
    router = ProviderRouter(providers)
    
    # Use sequential strategy for simple testing
    all_models = {p: prov.get_all_models() for p, prov in providers.items()}
    strategy = SequentialStrategy(providers, all_models)
    
    return create_app(router=router, strategy=strategy)

# 3. Write an asynchronous test case
@pytest.mark.asyncio
async def test_chat_completions_endpoint(test_app):
    # Use ASGITransport to test the FastAPI app directly without starting a server
    transport = ASGITransport(app=test_app)
    
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello world!"}],
            "stream": False
        })
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello! I am the mock provider."
    assert "usage" in data