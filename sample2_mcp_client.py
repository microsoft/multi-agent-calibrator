import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import the configuration loader
from sk_calibrator_config import load_config

# Load endpoints from YAML configuration
config = load_config("sample_sk_orchestrator_config.yaml")
azure_openai_endpoint = config.get("azureopenai_endpoint")
azure_openai_model = config.get("azureopenai_model")

class MCPClient:
    def __init__(self):
        # Initialize session and Azure OpenAI client
        self.session = None
        self.exit_stack = AsyncExitStack()
        # Store endpoint and deployment on self so other methods can use them
        self.azure_endpoint = azure_openai_endpoint
        self.azure_deployment = azure_openai_model
        # Create token provider for Azure AD authentication
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        # Instantiate Async Azure OpenAI client with specified endpoint and deployment
        self.openai = AsyncAzureOpenAI(
            api_version="2024-06-01",  # API version supporting tools (GA)
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            azure_ad_token_provider=token_provider
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server given the server script path (Python or Node)."""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        # Prepare server parameters for stdio connection
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        # Launch the server and establish stdio connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        # Create an MCP client session over the stdio transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        # Initialize the MCP session (handshake with server)
        await self.session.initialize()
        # Log available tools from the server
        response = await self.session.list_tools()
        tool_names = [tool.name for tool in response.tools]
        print("\nConnected to server with tools:", tool_names)

    async def process_query(self, query: str) -> str:
        """Process a single user query using the Azure OpenAI model and available tools."""
        # Start conversation with the user message
        messages = [{"role": "user", "content": query}]
        # Fetch available tools from the MCP server and format them for the OpenAI API
        response = await self.session.list_tools()
        available_tools = []
        for tool in response.tools:
            # Each tool is provided as type 'function' with its schema
            tool_spec = {
                "name": tool.name,
                "description": tool.description
            }
            # Include tool parameters schema if available
            if hasattr(tool, "inputSchema"):
                tool_spec["parameters"] = tool.inputSchema  # JSON schema for inputs
            elif hasattr(tool, "parameters"):
                tool_spec["parameters"] = tool.parameters
            available_tools.append({"type": "function", "function": tool_spec})
        # Send initial request to Azure OpenAI with tools enabled
        response = await self.openai.chat.completions.create(
            model=self.azure_deployment,      # use stored deployment name
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            max_tokens=800,
            temperature=0.7
        )
        # Parse the response to detect if a tool call is required
        completion = response.model_dump()  # convert to dict for inspection
        while True:
            choice_msg = completion["choices"][0]["message"]
            # Check if the model's response includes any tool call requests
            if choice_msg.get("tool_calls"):
                # Append the assistant message (tool request) to conversation history
                messages.append({
                    "role": choice_msg["role"],
                    "content": choice_msg.get("content"),
                    "tool_calls": choice_msg["tool_calls"]
                })
                # Execute each requested tool call
                for tool_call in choice_msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    # Parse function arguments from JSON
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = tool_call["function"]["arguments"]
                    # Call the tool via MCP server and get the result
                    try:
                        result = await self.session.call_tool(tool_name, args)
                        # Extract tool output content (assuming result has .content or .output)
                        if hasattr(result, "content"):
                            result_content = result.content
                        elif hasattr(result, "output"):
                            result_content = result.output
                        else:
                            result_content = str(result)
                    except Exception as err:
                        # Handle tool execution errors by returning an error message
                        result_content = json.dumps({"error": str(err)})
                    # Append the tool result as a message for the model, with the tool_call_id
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result_content
                    })
                # Send the updated conversation (including tool results) back to the model
                response = await self.openai.chat.completions.create(
                    model=self.azure_deployment,  # use stored deployment name
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=800,
                    temperature=0.7
                )
                completion = response.model_dump()
                # Loop again to check if further tool calls are needed
                continue
            else:
                # No tool calls; the assistant's answer is ready
                return choice_msg.get("content", "")

    async def chat_loop(self):
        """Run an interactive chat loop with the user (for testing purposes)."""
        print("\nMCP Client Started! Type your query or 'quit' to exit.")
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                if not user_input or user_input.lower() in ("quit", "exit"):
                    break
                answer = await self.process_query(user_input)
                print("\nAssistant:", answer)
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self):
        """Clean up resources and close connections."""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

#  python .\sample2_mcp_client.py .\sample2_mcp\sample2_mcp_server.py