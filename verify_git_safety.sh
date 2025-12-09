#!/bin/bash

# Git Safety Verification Script
# Run this before pushing to ensure no secrets are exposed

echo "🔍 Git Safety Verification"
echo "=========================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ISSUES_FOUND=0

# Check 1: Look for .env files (not examples)
echo "1. Checking for .env files..."
ENV_FILES=$(find . -name ".env" -not -name "*.example" -not -path "*/node_modules/*" -not -path "*/.venv/*" 2>/dev/null)
if [ -z "$ENV_FILES" ]; then
    echo -e "${GREEN}✅ No .env files found${NC}"
else
    echo -e "${RED}❌ Found .env files:${NC}"
    echo "$ENV_FILES"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 2: Look for database files
echo "2. Checking for database files..."
DB_FILES=$(find . -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" | grep -v node_modules | grep -v .venv 2>/dev/null)
if [ -z "$DB_FILES" ]; then
    echo -e "${GREEN}✅ No database files found${NC}"
else
    echo -e "${YELLOW}⚠️  Found database files (verify they're not tracked):${NC}"
    echo "$DB_FILES"
fi
echo ""

# Check 3: Check git status for sensitive files
echo "3. Checking git status for sensitive patterns..."
# Exclude legitimate files that manage API keys but don't contain them
SENSITIVE_STAGED=$(git status --porcelain 2>/dev/null | grep -E "\.env$|\.db$|\.sqlite" | grep -v "\.example" | grep -v "\.md" | grep -v "_manager\.py" | grep -v "_example\.py" | grep -v "test_")
if [ -z "$SENSITIVE_STAGED" ]; then
    echo -e "${GREEN}✅ No sensitive files in git status${NC}"
else
    echo -e "${RED}❌ Found sensitive files in git status:${NC}"
    echo "$SENSITIVE_STAGED"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 4: Look for hardcoded API keys in staged files
echo "4. Checking for hardcoded API keys in staged files..."
if git diff --cached --name-only &>/dev/null; then
    STAGED_FILES=$(git diff --cached --name-only 2>/dev/null)
    if [ -n "$STAGED_FILES" ]; then
        # Check for patterns like sk-..., api_key = "...", etc.
        POTENTIAL_KEYS=$(git diff --cached | grep -E "(sk-[a-zA-Z0-9]{20,}|api[_-]?key['\"]?\s*[:=]\s*['\"][a-zA-Z0-9_-]{20,})" | grep -v "your-key" | grep -v "example" | grep -v "placeholder")
        if [ -z "$POTENTIAL_KEYS" ]; then
            echo -e "${GREEN}✅ No hardcoded API keys found in staged files${NC}"
        else
            echo -e "${RED}❌ Potential API keys found in staged files:${NC}"
            echo "$POTENTIAL_KEYS"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    else
        echo -e "${GREEN}✅ No staged files to check${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Not a git repository or no staged files${NC}"
fi
echo ""

# Check 5: Verify .gitignore exists and has key patterns
echo "5. Checking .gitignore configuration..."
if [ -f ".gitignore" ]; then
    if grep -q "\.env" .gitignore && grep -q "\.db" .gitignore; then
        echo -e "${GREEN}✅ .gitignore properly configured${NC}"
    else
        echo -e "${YELLOW}⚠️  .gitignore may be missing important patterns${NC}"
    fi
else
    echo -e "${RED}❌ .gitignore file not found${NC}"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi
echo ""

# Check 6: Look for node_modules or __pycache__ in git
echo "6. Checking for build artifacts in git..."
BUILD_ARTIFACTS=$(git ls-files 2>/dev/null | grep -E "node_modules/|__pycache__|\.pyc$|build/|dist/" | head -5)
if [ -z "$BUILD_ARTIFACTS" ]; then
    echo -e "${GREEN}✅ No build artifacts tracked in git${NC}"
else
    echo -e "${YELLOW}⚠️  Found build artifacts in git (should be in .gitignore):${NC}"
    echo "$BUILD_ARTIFACTS"
    echo "..."
fi
echo ""

# Summary
echo "=========================="
echo "Summary"
echo "=========================="
if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Safe to push.${NC}"
    echo ""
    echo "You can now safely run:"
    echo "  git add ."
    echo "  git commit -m 'Your message'"
    echo "  git push origin main"
    exit 0
else
    echo -e "${RED}❌ Found $ISSUES_FOUND issue(s). Please fix before pushing.${NC}"
    echo ""
    echo "Actions to take:"
    echo "1. Remove or move sensitive files outside the repository"
    echo "2. Ensure .gitignore includes all sensitive patterns"
    echo "3. Run 'git rm --cached <file>' to unstage sensitive files"
    echo "4. Run this script again to verify"
    exit 1
fi
