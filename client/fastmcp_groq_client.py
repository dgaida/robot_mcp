# fastmcp_groq_client.py

import asyncio
import os
import json
from typing import List, Dict, Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport
from groq import Groq
from dotenv import load_dotenv

import logging

# Client can use stdout since it doesn't communicate via stdio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotMCPClient")


class RobotFastMCPClient:
    def __init__(self, groq_api_key: str, model: str = "moonshotai/kimi-k2-instruct-0905"):
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = model
        # self.server_script = server_script
        self.available_tools: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []

        # System prompt for the LLM
        self.system_prompt = """You are a helpful robot control assistant. You have access to various tools to control a robotic arm and detect objects in its workspace.

        Key capabilities:
        - Pick and place objects using coordinates
        - Detect and query objects in the workspace
        - Move the robot to observation poses
        - Get information about workspaces

        When the user asks you to do something:
        1. Use the available tools to accomplish the task
        2. Always call get_detected_objects first to understand what's in the workspace
        3. Use exact coordinates from detected objects for pick and place operations
        4. Provide clear feedback about what you're doing
        5. If something fails, explain what went wrong

        Coordinate system:
        - Coordinates are in meters [x, y]
        - X-axis: forward/backward from robot base
        - Y-axis: left/right from robot base

        Location options for placement:
        - "left next to" - places object to the left
        - "right next to" - places object to the right
        - "above" - places above
        - "below" - places below
        - "on top of" - stacks on top
        - "close to" - near the coordinate

        Always be precise and verify object positions before attempting to manipulate them."""

        transport = SSETransport("http://127.0.0.1:8000/sse")

        self.client = Client(transport)

    async def connect(self):
        print("🤖 Connecting to FastMCP server...")
        await self.client.__aenter__()

        self.available_tools = await self.client.list_tools()
        print(f"Connected! Found tools: {[t.name for t in self.available_tools]}")

    async def disconnect(self):
        if hasattr(self, "client"):
            await self.client.__aexit__(None, None, None)
        print("✓ Disconnected from MCP server")

    def _convert_tools_to_groq_format(self) -> List[Dict[str, Any]]:
        """Convert FastMCP tool definitions to Groq/OpenAI format."""
        tools = []
        for tool in self.available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema  # fastmcp liefert schon JSON schema
                }
            })
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call a tool via MCP.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result as string
        """
        print(f"🔧 Calling tool: {tool_name}")
        print(f"   Arguments: {json.dumps(arguments, indent=2)}")

        try:
            result = await self.client.call_tool(tool_name, arguments)

            # print(result)

            # Extract text content from result
            if result.content:
                text_results = [
                    item.text for item in result.content
                    if hasattr(item, 'text')
                ]
                result_text = "\n".join(text_results)
                print(f"✓ Result: {result_text}\n")
                return result_text
            else:
                return "Tool executed successfully (no output)"

        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            print(f"✗ {error_msg}\n")
            return error_msg

    async def process_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """
        Process tool calls from the LLM response.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of tool results for the next LLM call
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            # Parse arguments
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Call the tool via MCP
            result = await self.call_tool(tool_name, arguments)

            # Format result for Groq
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": result
            })

        return tool_results

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the assistant's response.

        Args:
            user_message: User's input message

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        max_iterations = 4  # 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Prepare messages for Groq
            messages = [
                           {"role": "system", "content": self.system_prompt}
                       ] + self.conversation_history

            # Call Groq API
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self._convert_tools_to_groq_format(),
                    tool_choice="auto",
                    max_tokens=4096,
                    temperature=0.7
                )

                assistant_message = response.choices[0].message

                # Check if the model wants to call tools
                if assistant_message.tool_calls:
                    # Add assistant's tool call request to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    # Process tool calls
                    tool_results = await self.process_tool_calls(assistant_message.tool_calls)

                    # Add tool results to history
                    self.conversation_history.extend(tool_results)

                    # Continue loop to get final response
                    continue

                else:
                    # No more tool calls, return final response
                    final_response = assistant_message.content or "I completed the task."

                    # Add to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })

                    return final_response

            except Exception as e:
                error_msg = f"Error calling Groq API: {str(e)}"
                print(f"✗ {error_msg}")
                return error_msg

        return "Maximum iterations reached. Task may be incomplete."

    def print_available_tools(self):
        """Print all available tools."""
        print("\n📋 Available Tools:")
        print("=" * 60)
        for tool in self.available_tools:
            func = tool["function"]
            print(f"\n🔧 {func['name']}")
            print(f"   {func['description'][:80]}...")
        print("=" * 60 + "\n")

    async def interactive_mode(self):
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("🤖 ROBOT CONTROL ASSISTANT")
        print("=" * 60)
        print("\nType your commands in natural language.")
        print("Examples:")
        print("  - 'What objects do you see?'")
        print("  - 'Pick up the pencil and place it at [0.2, 0.1]'")
        print("  - 'Move the red cube to the right of the blue square'")
        print("  - 'Show me the largest object'")
        print("  - 'Pick all pens or pencils and place them left next to the chocolate bar.'")
        print("  - 'Swap positions of pencil and chocolate bar.'")
        print("\nType 'quit' or 'exit' to stop.")
        print("Type 'tools' to see available tools.")
        print("Type 'clear' to clear conversation history.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'tools':
                    self.print_available_tools()
                    continue

                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation history cleared.\n")
                    continue

                print()  # Empty line for readability

                # Process the message
                response = await self.chat(user_input)

                print(f"\n🤖 Assistant: {response}\n")
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")

    async def run_command(self, command: str) -> str:
        """
        Run a single command and return the response.

        Args:
            command: Natural language command

        Returns:
            Assistant's response
        """
        response = await self.chat(command)
        return response


async def main():
    load_dotenv(dotenv_path="secrets.env")

    api_key = os.getenv("GROQ_API_KEY")

    client = RobotFastMCPClient(groq_api_key=api_key)

    await client.connect()

    # Beispiel: Tool direkt aufrufen
    # result = await client.call_tool("pick_place_object",
    #                                 {"object_name": "pencil", "pick_coordinate": [0.24, 0.02],
    #                                  "place_coordinate": [0.1, 0.11], "location": "right next to"})
    # print("Tool result:", result)

    await client.interactive_mode()

    await client.disconnect()


# python "client/fastmcp_groq_client.py"
if __name__ == "__main__":
    asyncio.run(main())
