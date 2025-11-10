#!/bin/bash

# Launch script for Robot Control GUI
# Usage: ./launch_gui.sh [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ROBOT="niryo"
SIMULATION=true
MODEL="moonshotai/kimi-k2-instruct-0905"
SHARE=false

# Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Print banner
print_banner() {
    print_message "$BLUE" "=================================="
    print_message "$BLUE" "  Robot Control GUI Launcher"
    print_message "$BLUE" "=================================="
    echo ""
}

# Check dependencies
check_dependencies() {
    print_message "$YELLOW" "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_message "$RED" "✗ Python 3 not found"
        exit 1
    fi
    print_message "$GREEN" "✓ Python 3 found: $(python3 --version)"

    # Check required packages
    local packages=("gradio" "torch" "fastmcp" "groq")
    for package in "${packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_message "$GREEN" "✓ $package installed"
        else
            print_message "$RED" "✗ $package not installed"
            print_message "$YELLOW" "  Install with: pip install $package"
            exit 1
        fi
    done

    echo ""
}

# Check API keys
check_api_keys() {
    print_message "$YELLOW" "Checking API keys..."

    # Check for secrets.env file
    if [ ! -f "secrets.env" ]; then
        print_message "$RED" "✗ secrets.env not found"
        print_message "$YELLOW" "  Creating template..."
        cat > secrets.env << EOF
GROQ_API_KEY=your_groq_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
EOF
        print_message "$YELLOW" "  Please edit secrets.env and add your API keys"
        exit 1
    fi

    # Source and check GROQ_API_KEY
    source secrets.env

    if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" = "your_groq_api_key_here" ]; then
        print_message "$RED" "✗ GROQ_API_KEY not set in secrets.env"
        exit 1
    fi
    print_message "$GREEN" "✓ GROQ_API_KEY found"

    echo ""
}

# Show configuration
show_config() {
    print_message "$BLUE" "Configuration:"
    echo "  Robot:      $ROBOT"
    echo "  Mode:       $([ "$SIMULATION" = true ] && echo "Simulation" || echo "Real Robot")"
    echo "  Model:      $MODEL"
    echo "  Share:      $([ "$SHARE" = true ] && echo "Yes" || echo "No")"
    echo ""
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --robot)
                ROBOT="$2"
                shift 2
                ;;
            --real)
                SIMULATION=false
                shift
                ;;
            --model)
                MODEL="$2"
                shift 2
                ;;
            --share)
                SHARE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_message "$RED" "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_help() {
    cat << EOF
Usage: ./launch_gui.sh [options]

Options:
  --robot ROBOT       Robot type (niryo or widowx) [default: niryo]
  --real              Use real robot instead of simulation
  --model MODEL       Groq model to use [default: moonshotai/kimi-k2-instruct-0905]
  --share             Create public Gradio link
  --help, -h          Show this help message

Examples:
  ./launch_gui.sh
  ./launch_gui.sh --robot widowx --real
  ./launch_gui.sh --model llama-3.1-8b-instant --share

EOF
}

# Launch GUI
launch_gui() {
    print_message "$GREEN" "🚀 Launching Robot Control GUI..."
    echo ""

    # Build command
    local cmd="python3 robot_gui/mcp_app.py --robot $ROBOT --model $MODEL"

    if [ "$SIMULATION" = false ]; then
        cmd="$cmd --no-simulation"
    fi

    if [ "$SHARE" = true ]; then
        cmd="$cmd --share"
    fi

    # Execute
    print_message "$YELLOW" "Command: $cmd"
    echo ""

    $cmd
}

# Cleanup on exit
cleanup() {
    echo ""
    print_message "$YELLOW" "Shutting down..."

    # Kill any remaining processes
    pkill -f "main_server.py" 2>/dev/null || true

    print_message "$GREEN" "✓ Cleanup complete"
}

# Main
main() {
    print_banner

    # Parse command line arguments
    parse_args "$@"

    # Check dependencies and API keys
    check_dependencies
    check_api_keys

    # Show configuration
    show_config

    # Set up cleanup trap
    trap cleanup EXIT INT TERM

    # Launch GUI
    launch_gui
}

# Run main
main "$@"
