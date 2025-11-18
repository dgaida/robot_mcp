"""
Example Scripts for Universal FastMCP Client with Multi-LLM Support

Demonstrates robot control using OpenAI, Groq, Gemini, or Ollama.

Usage:
    python examples/universal_client_examples.py [example_name] [--api PROVIDER]

Examples:
    # Auto-detect available API
    python examples/universal_client_examples.py workspace_scan

    # Use specific provider
    python examples/universal_client_examples.py sort_by_size --api openai
    python examples/universal_client_examples.py create_pattern --api groq
    python examples/universal_client_examples.py --api gemini

    # Compare providers
    python examples/universal_client_examples.py compare_providers
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.fastmcp_universal_client import RobotUniversalMCPClient
from examples.robot_examples_base import EXAMPLE_MAP, RobotExamplesBase


class UniversalRobotExamples(RobotExamplesBase):
    """Robot control examples using Universal MCP client with multi-LLM support."""

    def __init__(
        self,
        api_choice: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """
        Initialize with Universal MCP client.

        Args:
            api_choice: LLM provider ('openai', 'groq', 'gemini', 'ollama', or None for auto)
            model: Specific model name (None for provider default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        client = RobotUniversalMCPClient(api_choice=api_choice, model=model, temperature=temperature, max_tokens=max_tokens)
        super().__init__(client)
        self.api_choice = api_choice
        self.model = model

    async def connect(self):
        """Connect to the FastMCP server."""
        await self.client.connect()

    async def disconnect(self):
        """Disconnect from the FastMCP server."""
        await self.client.disconnect()

    # ==================== NEW EXAMPLE: Compare Providers ====================

    async def compare_providers(self):
        """Compare performance across different LLM providers."""
        print("\n🔬 MULTI-LLM COMPARISON TEST")
        print("=" * 70)

        command = "What objects do you see in the workspace? List them with positions."

        # Determine which providers to test
        providers_to_test = []

        if os.getenv("OPENAI_API_KEY"):
            providers_to_test.append(("openai", "gpt-4o-mini"))
        if os.getenv("GROQ_API_KEY"):
            providers_to_test.append(("groq", "moonshotai/kimi-k2-instruct-0905"))
        if os.getenv("GEMINI_API_KEY"):
            providers_to_test.append(("gemini", "gemini-2.0-flash"))

        # Always test Ollama if available
        try:
            import ollama

            ollama.list()  # Test if Ollama is running
            providers_to_test.append(("ollama", "llama3.2:1b"))
        except Exception:
            pass

        if not providers_to_test:
            print("❌ No LLM providers available!")
            print("Add at least one API key to secrets.env or install Ollama")
            return

        print(f"\n📋 Testing {len(providers_to_test)} provider(s):\n")

        results = {}

        for api, model in providers_to_test:
            print(f"\n{'=' * 70}")
            print(f"Testing: {api.upper()} - {model}")
            print("=" * 70)

            try:
                # Create new client for this provider
                test_client = RobotUniversalMCPClient(api_choice=api, model=model, temperature=0.7)

                await test_client.connect()

                # Measure response time
                start_time = time.time()
                response = await test_client.chat(command)
                elapsed = time.time() - start_time

                await test_client.disconnect()

                # Store results
                results[api] = {
                    "model": model,
                    "response": response[:200] + "..." if len(response) > 200 else response,
                    "time": elapsed,
                    "success": True,
                }

                print(f"\n✅ Response ({elapsed:.2f}s):")
                print(response[:300] + ("..." if len(response) > 300 else ""))

            except Exception as e:
                results[api] = {"model": model, "error": str(e), "success": False}

                print(f"\n❌ Failed: {e}")

            # Small delay between tests
            await asyncio.sleep(2)

        # Summary
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 70)

        successful = [r for r in results.items() if r[1].get("success")]

        if successful:
            # Sort by response time
            successful.sort(key=lambda x: x[1]["time"])

            print("\n🏆 Rankings (fastest to slowest):\n")
            for rank, (api, data) in enumerate(successful, 1):
                print(f"{rank}. {api.upper()} - {data['model']}")
                print(f"   Time: {data['time']:.2f}s")

            fastest = successful[0]
            print(f"\n⚡ Fastest: {fastest[0].upper()} ({fastest[1]['time']:.2f}s)")

        failed = [api for api, data in results.items() if not data.get("success")]
        if failed:
            print(f"\n❌ Failed providers: {', '.join(f.upper() for f in failed)}")

        print("\n" + "=" * 70)

    # ==================== NEW EXAMPLE: Provider Switching ====================

    async def demonstrate_switching(self):
        """Demonstrate switching between providers."""
        print("\n🔄 PROVIDER SWITCHING DEMONSTRATION")
        print("=" * 70)

        providers = []
        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", "gpt-4o-mini"))
        if os.getenv("GROQ_API_KEY"):
            providers.append(("groq", "moonshotai/kimi-k2-instruct-0905"))
        if os.getenv("GEMINI_API_KEY"):
            providers.append(("gemini", "gemini-2.0-flash"))

        if len(providers) < 2:
            print("❌ Need at least 2 providers with API keys for this demo")
            print("Add more API keys to secrets.env")
            return

        commands = [
            "What objects do you see?",
            "Which object is largest?",
            "Sort objects by size",
        ]

        for i, (api, model) in enumerate(providers):
            if i >= len(commands):
                break

            print(f"\n{'=' * 70}")
            print(f"Using: {api.upper()} - {model}")
            print("=" * 70)

            # Switch to this provider
            self.client.llm_client = self.client.LLMClient(api_choice=api, model=model, temperature=0.7)

            # Execute command
            command = commands[i]
            print(f"\n📝 Command: {command}")

            response = await self.client.chat(command)
            print(f"\n🤖 Response: {response}\n")

            await asyncio.sleep(1)

        print("\n✅ Successfully demonstrated switching between providers!")


# Add new examples to the map
EXAMPLE_MAP_EXTENDED = {
    **EXAMPLE_MAP,
    "compare_providers": "Compare LLM Provider Performance",
    "demonstrate_switching": "Demonstrate Provider Switching",
}


async def main():
    """Main entry point for examples."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robot Control Examples with Universal Multi-LLM Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Examples:
  workspace_scan          - Scan and analyze workspace
  sort_by_size            - Sort objects by size
  create_pattern          - Create geometric pattern
  compare_providers       - Compare LLM provider performance
  demonstrate_switching   - Demonstrate provider switching
  all                     - Run all examples

LLM Providers:
  --api openai            - Use OpenAI (GPT-4o, GPT-4o-mini)
  --api groq              - Use Groq (Kimi, Llama, Mixtral)
  --api gemini            - Use Google Gemini
  --api ollama            - Use Ollama (local)
  (default: auto-detect)

Examples:
  # Auto-detect API
  python examples/universal_client_examples.py workspace_scan

  # Use specific provider
  python examples/universal_client_examples.py sort_by_size --api openai

  # Compare all available providers
  python examples/universal_client_examples.py compare_providers

  # Custom model
  python examples/universal_client_examples.py --api groq --model llama-3.3-70b-versatile
        """,
    )

    parser.add_argument(
        "example",
        nargs="?",
        default="workspace_scan",
        help="Example to run (default: workspace_scan)",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "groq", "gemini", "ollama"],
        help="LLM provider (auto-detected if not specified)",
    )
    parser.add_argument("--model", help="Specific model name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens (default: 4096)")

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(dotenv_path="secrets.env")

    # Create examples instance
    print("\n🤖 Initializing Universal Robot Examples")
    print("=" * 70)

    examples = UniversalRobotExamples(
        api_choice=args.api, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens
    )

    try:
        await examples.connect()

        print(f"\nUsing: {examples.client.llm_client.api_choice.upper()}")
        print(f"Model: {examples.client.llm_client.llm}")
        print("=" * 70)

        if args.example == "all":
            # Run all examples
            for name, description in EXAMPLE_MAP_EXTENDED.items():
                if name == "compare_providers":
                    continue  # Skip comparison in "all" mode
                example_func = getattr(examples, name)
                await examples.run_example(example_func, description)
                await asyncio.sleep(3)
        elif args.example in EXAMPLE_MAP_EXTENDED:
            # Run specific example
            description = EXAMPLE_MAP_EXTENDED[args.example]
            example_func = getattr(examples, args.example)
            await examples.run_example(example_func, description)
        else:
            print(f"Unknown example: {args.example}")
            print(f"Available: {', '.join(EXAMPLE_MAP_EXTENDED.keys())}, all")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await examples.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
