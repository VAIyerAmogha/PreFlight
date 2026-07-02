End of session wrap-up. Run all steps in order.

1. Run test suite (check CLAUDE.md for command).
   Report result.

2. Use reviewer agent on: git diff --name-only HEAD
   Report any must-fix findings.

3. Use claude-md-updater agent.
   Update CLAUDE.md with everything completed this session.

4. Show: git diff CLAUDE.md
   Confirm the update was made.

5. Suggest a git commit message for work done this session.
