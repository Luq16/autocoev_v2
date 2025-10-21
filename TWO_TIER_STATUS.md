# Two-Tier Workflow Status Report

**Generated**: October 14, 2025
**Test Run**: `results/test_run/`

---

## ✅ Completed Steps

### 1. Tier 1: ESM-2 Fast Screening (COMPLETE)

**Status**: ✅ Successfully completed in 0.3 minutes
**Results File**: `results/test_run/tier1_esm2_results.csv`

#### Performance Summary:
- **Proteins Analyzed**: 5 (EGFR, GRB2, TP53, MDM2, SHC1)
- **Total Pairs**: 10 (all combinations)
- **Threshold**: 0.5
- **Interactions Found**: 10/10 (100%)
- **STRING Validation**: 8 known + 2 novel

#### Top Interactions (by confidence):
| Rank | Protein A | Protein B | AutoCoEv | STRING | Combined | Novel |
|------|-----------|-----------|----------|--------|----------|-------|
| 1 | TP53 | MDM2 | 0.692 | 0.999 | 0.876 | False |
| 2 | GRB2 | SHC1 | 0.676 | 0.999 | 0.870 | False |
| 3 | EGFR | GRB2 | 0.636 | 0.999 | 0.854 | False |
| 4 | EGFR | SHC1 | 0.571 | 0.999 | 0.809 | False |
| 5 | TP53 | SHC1 | 0.531 | 0.876 | 0.778 | False |

---

### 2. Tier 2: CAPS2 Preparation (COMPLETE)

**Status**: ✅ All input files generated successfully

#### Generated Files:

**Configuration File:**
```
✅ results/test_run/caps2_validation/settings.conf (2.2 KB)
   - Organism: 10090 (Mouse)
   - Taxonomic Level: 40674 (Mammalia)
   - Working Directory: /var/tmp/autocoev_twotier
   - Database Directory: /var/tmp/DB10v1
```

**Protein Lists:**
```
✅ results/test_run/caps2_validation/proteins/tier2_candidates.tsv (25 B)
   Contains:
   - EGFR
   - GRB2
   - MDM2
   - SHC1
   - TP53
```

**FASTA Sequences:**
```
✅ results/test_run/caps2_validation/caps2_input_proteins.fasta (2.7 KB)
   Format: CAPS2-compatible (60-character lines)
   Proteins: 5 unique proteins from top candidates
```

**Convenience Copy:**
```
✅ settings.conf (2.2 KB) - Copied to project root for easy access
```

---

## ⏸️ Pending Step: CAPS2 Execution

### Why CAPS2 Requires Manual Execution

CAPS2 validation requires:
1. **OrthoDB databases** (~50 GB total, ~10 GB after trimming)
2. **Interactive menu selection** (cannot be fully automated)
3. **Manual verification** of each step's completion

---

## 📋 Next Steps: Running CAPS2

### Prerequisites Check

**Database Status:**
```
❌ /var/tmp/DB10v1/ does not exist
```

**Required Databases:**
- odb10v1_gene_xrefs.tab.gz (MD5: 3ab6d2efdc43ed051591514a3cc9044e)
- odb10v1_OG2genes.tab.gz (MD5: 33e63fa97ee420707cd3cddcb5e282a6)
- odb10v1_all_fasta.tab.gz (MD5: 831ef830fff549857a4c8d1639a760cb)

---

## 🚀 Complete CAPS2 Execution Guide

### Step 1: Prepare Environment

```bash
# Set environment variables
export TMP=/var/tmp/autocoev_twotier
export DTB=/var/tmp/DB10v1

# Create working directories
mkdir -p $TMP
mkdir -p $DTB

# Navigate to AutoCoEv directory
cd /Users/luqmanawoniyi_1/Documents/autoCoev_mordenization/autocoev_v2
```

### Step 2: Launch CAPS2 Script

```bash
# Copy settings to current directory (already done)
cp results/test_run/caps2_validation/settings.conf ./settings_caps2.conf

# Launch CAPS2
./start.sh
```

### Step 3: Database Preparation (Interactive Menu)

When `start.sh` launches, you'll see a menu. Select these options **in order**:

#### Database Download & Preparation:
1. **Option 1**: Download odb10v1_gene_xrefs
2. **Option 2**: Download odb10v1_OG2genes
3. **Option 3**: Download odb10v1_all_fasta
4. **Option 4**: Check MD5sum of databases (verify integrity)
5. **Option 5**: Extract odb10v1_gene_xrefs
6. **Option 6**: Extract odb10v1_OG2genes
7. **Option 7**: Extract odb10v1_all_fasta
8. **Option 8**: Index odb10v1_all_fasta
9. **Option 9**: Trim gene_xrefs for organism 10090 (Mouse)
10. **Option 10**: Trim OG2genes for taxonomic level 40674 (Mammalia)
11. **Option 11**: [DONE AND CONTINUE]

**Note**: Database download is ~50 GB. This step only needs to be done **once** per system.

### Step 4: Configure Analysis (Interactive Prompts)

After database setup, you'll be prompted:

```bash
# Working directory
Press ENTER to accept: /var/tmp/autocoev_twotier

# Proteins folder
Change to: results/test_run/caps2_validation/proteins/
(or press ENTER if protein list is already configured)

# Species list
Press ENTER to accept: placental.tsv
```

### Step 5: Run CAPS2 Analysis Steps (Main Menu)

Select these pipeline steps **in order**:

#### Essential Steps:
1. **Option 1**: Pair UniProt <-> OrthoDB <-> OGuniqueID
2. **Option 2**: Get orthologues (level: 40674)
3. **Option 3**: Download sequences from UniProt (organism: 10090)
4. **Option 4**: BLAST orthologues against UniProt sequence
5. **Option 5**: Get FASTA sequences of the best hits
6. **Option 6**: [MSA] Exclude too divergent sequences (GUIDANCE)
7. **Option 7**: [MSA] Create MSA with selected method (prank)
8. **Option 8**: [TRE] Prepare trees (auto - CAPS will generate)
9. **Option 9**: [RUN] Create pairs (all combinations)
10. **Option 10**: [RUN] CAPS run (alpha: 0.01)

**Note**: Each step validates the previous step's output. Follow the sequence exactly.

---

## ⏱️ Estimated Timeline

### First-Time Setup (with database download):
- **Database Download**: 2-4 hours (depends on internet speed)
- **Database Extraction**: 30-60 minutes
- **Database Indexing**: 10-15 minutes
- **Total Setup**: ~3-5 hours (one-time only)

### CAPS2 Analysis (5 proteins, 10 pairs):
- **UniProt Pairing**: ~5 minutes
- **Ortholog Retrieval**: ~15 minutes
- **BLAST Analysis**: ~30 minutes
- **FASTA Extraction**: ~10 minutes
- **GUIDANCE (MSA QC)**: ~20 minutes
- **MSA Generation (prank)**: ~45 minutes
- **Tree Preparation**: ~10 minutes (if auto)
- **Pair Creation**: ~2 minutes
- **CAPS Execution**: ~60 minutes
- **Total Analysis**: ~3-4 hours

**Note**: After database setup, future analyses only require the analysis time (~3-4 hours for this dataset).

---

## 🔍 Expected Output Files

After successful CAPS2 execution, you'll find:

```
/var/tmp/autocoev_twotier/results/coev/
├── [Protein pairs with significant coevolution]
│   ├── TP53-MDM2/
│   │   ├── allResidues
│   │   ├── coevolving_pairs.tsv
│   │   └── statistics.txt
│   ├── EGFR-GRB2/
│   └── ...
```

---

## 🔄 Resuming Two-Tier Workflow

After CAPS2 completes, resume the two-tier workflow:

### Option 1: Manual Merging

```bash
# Parse CAPS2 results
python3 -c "
from modern.utils.caps2_parser import CAPS2ResultsParser
from pathlib import Path

parser = CAPS2ResultsParser('/var/tmp/autocoev_twotier/results/coev')
results = parser.parse_all_results()
print(f'Found {len(results)} protein pairs with coevolution')
for pair in results:
    print(f'{pair[\"protein_a\"]}-{pair[\"protein_b\"]}: {pair[\"coev_count\"]} coevolving residues')
"

# Merge with ESM-2 results
python3 -c "
from modern.utils.caps2_integration import merge_caps2_with_esm2
import pandas as pd

merged = merge_caps2_with_esm2(
    esm2_csv='results/test_run/tier1_esm2_results.csv',
    caps2_results_dir='/var/tmp/autocoev_twotier/results/coev',
    output_csv='results/test_run/merged_results.csv'
)
print(f'Merged results saved: {len(merged)} interactions')
"
```

### Option 2: Re-run Two-Tier Workflow (Skip to Merge)

```bash
# Modify configuration to skip tiers
python3 two_tier_workflow.py \
  --input test_data/test_proteins.fasta \
  --output results/test_run \
  --skip-tier1 \
  --skip-tier2 \
  --merge-only
```

---

## 📊 Alternative: Skip CAPS2 (ESM-2 Only)

If you want to proceed without CAPS2 validation:

```bash
# Generate report from Tier 1 only
python3 autocoev_modern.py \
  --input test_data/test_proteins.fasta \
  --output results/test_run_esm2_only \
  --threshold 0.5 \
  --no-llm

# Results will use ESM-2 + STRING scoring only
# Weights: ESM-2 (60%) + STRING (40%)
```

---

## 💡 Recommendations

### For Quick Testing (Today):
- ✅ **Use ESM-2 results only** (already complete)
- ✅ **Analyze Tier 1 results** (`tier1_esm2_results.csv`)
- ✅ **Review top candidates** (already identified)

### For Full Validation (When Time Permits):
- 📦 **Set up databases** (one-time, ~3-5 hours)
- 🧬 **Run CAPS2 analysis** (~3-4 hours per analysis)
- 📈 **Compare ESM-2 vs CAPS2** (residue-level validation)

### For Production Use:
1. **Run Tier 1 (ESM-2)** on large datasets (fast screening)
2. **Select top N candidates** (e.g., top 20-50)
3. **Run Tier 2 (CAPS2)** only on selected candidates (deep validation)
4. **Merge results** for comprehensive confidence scoring

---

## 📁 File Locations Summary

```
autocoev_v2/
├── settings.conf                              # ✅ CAPS2 config (copied)
├── settings_twotier.conf                      # ✅ CAPS2 config (backup)
├── two_tier_workflow.py                       # ✅ Main orchestrator
├── run_two_tier.sh                           # ✅ CLI wrapper
├── TWO_TIER_WORKFLOW_GUIDE.md                # 📚 Full documentation
├── TWO_TIER_STATUS.md                        # 📄 This file
│
├── results/test_run/
│   ├── tier1_esm2_results.csv                # ✅ Tier 1 complete (10 pairs)
│   └── caps2_validation/
│       ├── settings.conf                      # ✅ CAPS2 config
│       ├── caps2_input_proteins.fasta         # ✅ FASTA sequences
│       └── proteins/
│           └── tier2_candidates.tsv           # ✅ Protein list (5 proteins)
│
├── modern/
│   ├── utils/
│   │   ├── caps2_integration.py              # 🔧 Format converters
│   │   └── caps2_parser.py                   # 🔧 Results parser
│   └── config/
│       └── two_tier_config.yaml              # ⚙️ Configuration
│
└── test_data/
    └── test_proteins.fasta                    # 📝 Test dataset
```

---

## ✅ Success Criteria

**Tier 1 (ESM-2)**: ✅ COMPLETE
- [x] All protein pairs analyzed
- [x] STRING validation performed
- [x] Results CSV generated
- [x] Top candidates identified

**Tier 2 (CAPS2)**: ⏸️ PENDING
- [x] Configuration generated
- [x] FASTA sequences prepared
- [x] Protein list created
- [ ] Databases downloaded/installed
- [ ] CAPS2 analysis executed
- [ ] Results parsed and merged

---

## 🆘 Troubleshooting

### Database Download Fails
```bash
# Manually download from OrthoDB
wget https://v101.orthodb.org/download/odb10v1_gene_xrefs.tab.gz
wget https://v101.orthodb.org/download/odb10v1_OG2genes.tab.gz
wget https://v101.orthodb.org/download/odb10v1_all_fasta.tab.gz

# Move to database directory
mv odb10v1_*.tab.gz /var/tmp/DB10v1/
```

### Disk Space Issues
```bash
# Check available space
df -h /var/tmp

# Required: ~50 GB for downloads, ~10 GB after trimming
# Free space by deleting .gz archives after extraction
```

### Permission Errors
```bash
# Create directories with proper permissions
sudo mkdir -p /var/tmp/autocoev_twotier
sudo mkdir -p /var/tmp/DB10v1
sudo chown -R $(whoami) /var/tmp/autocoev_twotier
sudo chown -R $(whoami) /var/tmp/DB10v1
```

---

## 📚 Additional Resources

- **Two-Tier Workflow Guide**: `TWO_TIER_WORKFLOW_GUIDE.md`
- **Modern Pipeline README**: `README_MODERN.md`
- **Legacy AutoCoEv Docs**: `doc/`
- **OrthoDB Documentation**: https://www.orthodb.org/

---

**Status**: Ready for CAPS2 execution
**Next Action**: Run `./start.sh` and follow database setup steps
**Estimated Time**: 3-5 hours (first time), 3-4 hours (subsequent runs)
