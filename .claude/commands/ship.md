Pre-ship checklist. Run all steps in order. Stop and report if any fail.

1. Run full test suite (check CLAUDE.md for command).
   Report: pass/fail, number of tests, any failures.

2. Use reviewer agent on: git diff --name-only HEAD
   Report: any must-fix findings. Stop if any found.

3. Check for hardcoded secrets:
   grep -rn "api_key\|apikey\|password\|secret\|token" \
     --include="*.py" --include="*.ts" --include="*.js" \
     --exclude-dir=".git" --exclude-dir="node_modules" \
     --exclude-dir="tests" .
   Report any hits. Stop if secrets found in non-test code.

4. Verify env vars:
   Check that every var used in code exists in .env.example.
   Report any missing.

5. Run: git diff --stat HEAD
   Show what is being committed.

6. Use claude-md-updater agent to update CLAUDE.md.

7. Suggest a conventional commit message for the work done.
   Format: type(scope): description
   Types: feat / fix / refactor / test / docs / chore
