#!/bin/bash

###############################################################################
# AutoCoEv v2 - Two-Tier Workflow CLI Wrapper
#
# Simplifies execution of the two-tier workflow (ESM-2 + CAPS2)
# Provides common presets and error handling
###############################################################################

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# Functions
###############################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}AutoCoEv v2 - Two-Tier Workflow${NC}"
    echo -e "${BLUE}ESM-2 Fast Screening + CAPS2 Validation${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_usage() {
    cat << EOF
Usage: ./run_two_tier.sh [OPTIONS] --input PROTEINS.fasta

OPTIONS:
    --input, -i FILE        Input FASTA file (required)
    --output, -o DIR        Output directory (default: ./results/two_tier)
    --top-n, -n NUMBER      Number of top candidates for CAPS2 (default: 20)
    --preset PRESET         Use preset configuration (see below)
    --config FILE           Custom configuration file
    --help, -h              Show this help message

PRESETS:
    quick               Fast screening (10 candidates, threshold 0.7)
    standard            Balanced analysis (20 candidates, threshold 0.6)
    comprehensive       Thorough validation (30 candidates, threshold 0.5)
    novel-only          Focus on novel discoveries only

EXAMPLES:
    # Quick analysis (10 top candidates)
    ./run_two_tier.sh --input proteins.fasta --preset quick

    # Standard analysis (20 candidates)
    ./run_two_tier.sh --input proteins.fasta

    # Comprehensive analysis (30 candidates)
    ./run_two_tier.sh --input proteins.fasta --preset comprehensive

    # Focus on novel discoveries
    ./run_two_tier.sh --input proteins.fasta --preset novel-only

    # Custom number of candidates
    ./run_two_tier.sh --input proteins.fasta --top-n 25

WORKFLOW STEPS:
    1. ESM-2 screens all protein pairs (~1-2 sec per protein)
    2. STRING validates against known interactions
    3. Top N candidates selected for CAPS2
    4. CAPS2 performs rigorous MSA-based coevolution analysis
    5. Results merged into comprehensive report

REQUIREMENTS:
    - Python 3.10+ with dependencies installed
    - For CAPS2: OrthoDB databases, BLAST, MUSCLE, PRANK, PhyML
    - GPU recommended for ESM-2 (10x speedup)

For more information, see TWO_TIER_WORKFLOW_GUIDE.md
EOF
}

check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python 3 not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python 3 found${NC}"

    # Check required Python packages
    python3 -c "import torch, esm, pandas, yaml" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠ Some Python dependencies missing${NC}"
        echo -e "${YELLOW}  Run: pip install -r requirements.txt${NC}"
    else
        echo -e "${GREEN}✓ Python dependencies found${NC}"
    fi

    echo ""
}

###############################################################################
# Parse Arguments
###############################################################################

INPUT=""
OUTPUT="./results/two_tier"
TOP_N=""
PRESET=""
CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --top-n|-n)
            TOP_N="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

###############################################################################
# Validation
###############################################################################

if [ -z "$INPUT" ]; then
    echo -e "${RED}Error: Input file required${NC}"
    print_usage
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT${NC}"
    exit 1
fi

###############################################################################
# Apply Presets
###############################################################################

if [ -n "$PRESET" ]; then
    echo -e "${BLUE}Applying preset: $PRESET${NC}\n"

    case $PRESET in
        quick)
            TOP_N=10
            ESM2_THRESHOLD="0.7"
            ;;
        standard)
            TOP_N=20
            ESM2_THRESHOLD="0.6"
            ;;
        comprehensive)
            TOP_N=30
            ESM2_THRESHOLD="0.5"
            ;;
        novel-only)
            TOP_N=15
            ESM2_THRESHOLD="0.6"
            PRIORITIZE_NOVEL="--novel-only"
            ;;
        *)
            echo -e "${RED}Unknown preset: $PRESET${NC}"
            exit 1
            ;;
    esac
fi

###############################################################################
# Main Execution
###############################################################################

print_header
check_dependencies

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Input:  $INPUT"
echo -e "  Output: $OUTPUT"
echo -e "  Top N:  ${TOP_N:-from config}"
echo -e "  Preset: ${PRESET:-none}"
echo ""

# Build command
CMD="python3 two_tier_workflow.py --input \"$INPUT\" --output \"$OUTPUT\""

if [ -n "$TOP_N" ]; then
    CMD="$CMD --top-n $TOP_N"
fi

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config \"$CONFIG\""
fi

if [ -n "$ESM2_THRESHOLD" ]; then
    CMD="$CMD --esm2-threshold $ESM2_THRESHOLD"
fi

# Run workflow
echo -e "${BLUE}Starting two-tier workflow...${NC}\n"
echo -e "${YELLOW}Command: $CMD${NC}\n"

eval $CMD
EXIT_CODE=$?

# Check result
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Two-tier workflow completed successfully${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Results are in: ${BLUE}$OUTPUT${NC}"
    echo -e "View the report: ${BLUE}$OUTPUT/autocoev_analysis_*.md${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Workflow failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    exit $EXIT_CODE
fi
